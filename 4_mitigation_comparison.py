"""
Mitigation Technique Evaluation

Tests 11 class overlap mitigation techniques (SMOTE, BorderlineSMOTE, ADASYN, CBO, Tomek Links, ENN, RENN, NCR, SMOTE+Tomek, SMOTE+ENN, Random Oversampling) combined with each classifier.

"""

import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ClusterBasedOversampling:
    """Custom implementation of Cluster-Based Oversampling (CBO)"""
    
    def __init__(self, k_clusters=5, sampling_strategy='auto', random_state=42):
        self.k_clusters = k_clusters
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        """Fit and resample the dataset"""
        np.random.seed(self.random_state)
        
        # Separate minority and majority classes
        X_min = X[y == 1]
        X_maj = X[y == 0]
        
        if len(X_min) < 2:
            return X, y
        
        # Determine number of clusters (at least 2, at most k_clusters)
        n_clusters = min(self.k_clusters, max(2, len(X_min) // 10))
        
        # Cluster minority samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_min)
        
        # Calculate borderline ratio for each cluster
        cluster_borderline_ratios = {}
        knn = NearestNeighbors(n_neighbors=min(6, len(X)))
        knn.fit(X)
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = X_min[cluster_mask]
            
            borderline_count = 0
            for sample in cluster_samples:
                distances, indices = knn.kneighbors([sample])
                neighbor_classes = y[indices[0][1:]]  # Exclude self
                if np.sum(neighbor_classes == 0) > 0:  # Has majority neighbors
                    borderline_count += 1
            
            ratio = borderline_count / len(cluster_samples) if len(cluster_samples) > 0 else 0
            cluster_borderline_ratios[cluster_id] = ratio
        
        # Generate synthetic samples proportional to borderline ratio
        synthetic_samples = []
        n_synthetic_total = len(X_maj) - len(X_min)  # Balance the dataset
        
        total_ratio = sum(cluster_borderline_ratios.values())
        if total_ratio == 0:
            total_ratio = 1
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = X_min[cluster_mask]
            
            if len(cluster_samples) < 2:
                continue
                
            # Samples to generate for this cluster
            weight = cluster_borderline_ratios[cluster_id]
            n_synthetic = int(n_synthetic_total * weight / total_ratio)
            
            for _ in range(n_synthetic):
                # Select two random samples from cluster
                idx1, idx2 = np.random.choice(len(cluster_samples), 2, replace=False)
                sample1, sample2 = cluster_samples[idx1], cluster_samples[idx2]
                
                # Generate synthetic sample along the line
                alpha = np.random.uniform(0, 1)
                synthetic = sample1 + alpha * (sample2 - sample1)
                synthetic_samples.append(synthetic)
        
        # Combine all samples
        if synthetic_samples:
            X_synthetic = np.array(synthetic_samples)
            X_resampled = np.vstack([X, X_synthetic])
            y_resampled = np.hstack([y, np.ones(len(X_synthetic))])
        else:
            X_resampled = X
            y_resampled = y
            
        return X_resampled, y_resampled


class MitigationComparison:
    """Framework for comparing mitigation strategies on overlapping imbalanced datasets"""
    
    def __init__(self, datasets_path, results_path='mitigation_results'):
        self.datasets_path = datasets_path
        self.results_path = results_path
        self.results = []
        
        # Create results directory
        os.makedirs(self.results_path, exist_ok=True)
        
        # Define mitigation techniques
        self.mitigation_techniques = {
            # Baseline
            'None': None,
            
            # Oversampling
            'RandomOverSampler': RandomOverSampler(random_state=42),
            'SMOTE': SMOTE(random_state=42, k_neighbors=5),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=5, kind='borderline-1'),
            'ADASYN': ADASYN(random_state=42, n_neighbors=5),
            'CBO': ClusterBasedOversampling(k_clusters=5, random_state=42),
            
            # Undersampling/Cleaning
            'TomekLinks': TomekLinks(),
            'ENN': EditedNearestNeighbours(n_neighbors=3),
            'RENN': RepeatedEditedNearestNeighbours(n_neighbors=3),
            'NCR': NeighbourhoodCleaningRule(n_neighbors=3),
            
            # Hybrid
            'SMOTEENN': SMOTEENN(random_state=42),
            'SMOTETomek': SMOTETomek(random_state=42),
        }
        
        # ALL 12 classifiers matching benchmark_overlap_classifiers.py
        self.classifiers = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                           random_state=42, early_stopping=True),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                                   class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, 
                                         use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced', verbose=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, 
                                                     class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, 
                                               class_weight='balanced'),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), 
                                        n_estimators=50, random_state=42),
            'Stacking': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('svm', SVC(kernel='linear', probability=True, random_state=42))
                ],
                final_estimator=LogisticRegression(random_state=42),
                cv=3
            )
        }
        
    def compute_overlap_metrics(self, X, y):
        """Compute overlap metrics for the dataset"""
        metrics = {}
        
        n_neighbors = min(6, len(X) - 1)
        if n_neighbors < 2:
            return {'N1': np.nan, 'N3': np.nan, 'imbalance_ratio': np.nan, 'minority_ratio': np.nan}
        
        # N1: Fraction of borderline points
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X)
        
        borderline_count = 0
        for i in range(len(X)):
            distances, indices = knn.kneighbors([X[i]])
            neighbor_classes = y[indices[0][1:]]  # Exclude self
            if len(np.unique(neighbor_classes)) > 1:
                borderline_count += 1
        
        metrics['N1'] = borderline_count / len(X)
        
        # N3: 1-NN error rate
        errors = 0
        for i in range(len(X)):
            distances, indices = knn.kneighbors([X[i]], n_neighbors=2)
            nn_class = y[indices[0][1]]  # Nearest neighbor (exclude self)
            if nn_class != y[i]:
                errors += 1
        
        metrics['N3'] = errors / len(X)
        
        # Class distribution
        class_counts = Counter(y)
        metrics['imbalance_ratio'] = class_counts[0] / class_counts[1] if class_counts[1] > 0 else np.inf
        metrics['minority_ratio'] = class_counts[1] / len(y)
        
        return metrics
    
    def parse_keel_file(self, filepath):
        """Parse KEEL .dat file format with robust class label handling"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        # Find where @data section starts
        data_start_idx = None
        attributes = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 3:
                    attr_name = parts[1]
                    attributes.append(attr_name)
            elif line.startswith('@data'):
                data_start_idx = i + 1
                break
        
        if data_start_idx is None:
            raise ValueError(f"No @data section found in {filepath}")
        
        # Parse the data
        X = []
        y_raw = []
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('@') and not line.startswith('%'):
                values = line.split(',')
                if len(values) > 1:
                    features = []
                    for v in values[:-1]:
                        try:
                            features.append(float(v))
                        except ValueError:
                            features.append(np.nan)
                    
                    # Store raw class label
                    label = values[-1].strip()
                    y_raw.append(label)
                    X.append(features)
        
        X = np.array(X)
        
        # Handle missing values
        if np.any(np.isnan(X)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # Robust class label handling: minority class = 1, majority class = 0
        unique_labels = list(set(y_raw))
        if len(unique_labels) != 2:
            raise ValueError(f"Expected 2 classes, found {len(unique_labels)}: {unique_labels}")
        
        label_counts = Counter(y_raw)
        minority_label = min(label_counts.keys(), key=lambda k: label_counts[k])
        
        y = np.array([1 if label == minority_label else 0 for label in y_raw])
        
        return X, y
    
    def evaluate_mitigation(self, X_train, y_train, X_test, y_test, technique_name, classifier_name):
        """Evaluate a mitigation technique with a specific classifier"""
        technique = self.mitigation_techniques[technique_name]
        classifier = self.classifiers[classifier_name]
        
        # Apply mitigation
        if technique is not None:
            try:
                X_resampled, y_resampled = technique.fit_resample(X_train, y_train)
            except Exception as e:
                # Silently fall back to original data
                X_resampled, y_resampled = X_train, y_train
        else:
            X_resampled, y_resampled = X_train, y_train
        
        # Train classifier
        try:
            classifier_copy = classifier.__class__(**classifier.get_params())
            classifier_copy.fit(X_resampled, y_resampled)
        except Exception as e:
            return None
        
        # Evaluate
        y_pred = classifier_copy.predict(X_test)
        
        try:
            y_proba = classifier_copy.predict_proba(X_test)[:, 1] if hasattr(classifier_copy, 'predict_proba') else None
        except:
            y_proba = None
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if y_proba is not None and len(np.unique(y_test)) > 1:
            try:
                auc_score = roc_auc_score(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                auprc = auc(recall, precision)
            except:
                auc_score = np.nan
                auprc = np.nan
        else:
            auc_score = np.nan
            auprc = np.nan
        
        # Compute overlap metrics after mitigation
        overlap_after = self.compute_overlap_metrics(X_resampled, y_resampled)
        
        return {
            'f1': f1,
            'auc': auc_score,
            'auprc': auprc,
            'n_samples_after': len(y_resampled),
            'minority_ratio_after': overlap_after['minority_ratio'],
            'N1_after': overlap_after['N1'],
            'N3_after': overlap_after['N3']
        }
    
    def run_experiments(self, dataset_names=None):
        """Run mitigation experiments on ALL datasets"""
        # Load ALL datasets from the folder dynamically
        if dataset_names is None:
            dataset_names = [f.replace('.dat', '') for f in os.listdir(self.datasets_path) 
                           if f.endswith('.dat') and not f.startswith('._')]
        
        print(f"Testing {len(self.mitigation_techniques)} mitigation techniques")
        print(f"On {len(dataset_names)} datasets")
        print(f"With {len(self.classifiers)} classifiers")
        print(f"Total experiments: {len(self.mitigation_techniques) * len(dataset_names) * len(self.classifiers)}")
        
        # Progress tracking
        total_experiments = len(dataset_names) * len(self.mitigation_techniques) * len(self.classifiers)
        
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            for dataset_name in dataset_names:
                # Load dataset
                filepath = os.path.join(self.datasets_path, f"{dataset_name}.dat")
                if not os.path.exists(filepath):
                    pbar.update(len(self.mitigation_techniques) * len(self.classifiers))
                    continue
                
                try:
                    X, y = self.parse_keel_file(filepath)
                    
                    # Skip if not enough samples or only one class
                    if len(X) < 20 or len(np.unique(y)) < 2:
                        pbar.update(len(self.mitigation_techniques) * len(self.classifiers))
                        continue
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    
                    # Compute original overlap metrics
                    overlap_before = self.compute_overlap_metrics(X, y)
                    
                    # 5-fold cross-validation
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    
                    for technique_name in self.mitigation_techniques:
                        for classifier_name in self.classifiers:
                            fold_results = []
                            
                            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                                X_train, X_test = X[train_idx], X[test_idx]
                                y_train, y_test = y[train_idx], y[test_idx]
                                
                                # Skip if train set has only one class
                                if len(np.unique(y_train)) < 2:
                                    continue
                                
                                # Evaluate mitigation
                                fold_result = self.evaluate_mitigation(
                                    X_train, y_train, X_test, y_test,
                                    technique_name, classifier_name
                                )
                                if fold_result:
                                    fold_results.append(fold_result)
                            
                            if fold_results:
                                # Aggregate results
                                result = {
                                    'dataset': dataset_name,
                                    'technique': technique_name,
                                    'classifier': classifier_name,
                                    'f1_mean': np.mean([r['f1'] for r in fold_results]),
                                    'f1_std': np.std([r['f1'] for r in fold_results]),
                                    'auc_mean': np.nanmean([r['auc'] for r in fold_results]),
                                    'auc_std': np.nanstd([r['auc'] for r in fold_results]),
                                    'auprc_mean': np.nanmean([r['auprc'] for r in fold_results]),
                                    **overlap_before,  # Original overlap metrics
                                    'N1_after_mean': np.mean([r['N1_after'] for r in fold_results]),
                                    'N3_after_mean': np.mean([r['N3_after'] for r in fold_results]),
                                    'minority_ratio_after_mean': np.mean([r['minority_ratio_after'] for r in fold_results])
                                }
                                
                                self.results.append(result)
                            
                            pbar.update(1)
                            
                except Exception as e:
                    print(f"\nError processing {dataset_name}: {str(e)}")
                    pbar.update(len(self.mitigation_techniques) * len(self.classifiers))
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        # Calculate improvements
        if len(self.results_df) > 0:
            self.calculate_improvements()
        
        return self.results_df
    
    def calculate_improvements(self):
        """Calculate improvement metrics compared to baseline (no mitigation)"""
        # Get baseline results
        baseline_results = self.results_df[self.results_df['technique'] == 'None']
        
        # Calculate improvements for each technique
        improvements = []
        
        for _, row in self.results_df.iterrows():
            if row['technique'] != 'None':
                # Find corresponding baseline
                baseline = baseline_results[
                    (baseline_results['dataset'] == row['dataset']) &
                    (baseline_results['classifier'] == row['classifier'])
                ]
                
                if not baseline.empty:
                    baseline_f1 = baseline.iloc[0]['f1_mean']
                    baseline_auc = baseline.iloc[0]['auc_mean']
                    
                    if baseline_f1 > 0:
                        f1_improvement = (row['f1_mean'] - baseline_f1) / baseline_f1 * 100
                    else:
                        f1_improvement = 0
                    
                    if baseline_auc > 0 and not np.isnan(baseline_auc):
                        auc_improvement = (row['auc_mean'] - baseline_auc) / baseline_auc * 100
                    else:
                        auc_improvement = np.nan
                    
                    improvement = {
                        'dataset': row['dataset'],
                        'technique': row['technique'],
                        'classifier': row['classifier'],
                        'baseline_f1': baseline_f1,
                        'f1_mean': row['f1_mean'],
                        'f1_improvement': f1_improvement,
                        'f1_improvement_abs': row['f1_mean'] - baseline_f1,
                        'auc_improvement': auc_improvement,
                        'auc_improvement_abs': row['auc_mean'] - baseline_auc,
                        'N1_reduction': row['N1'] - row['N1_after_mean'] if not np.isnan(row['N1']) else np.nan,
                        'N3_reduction': row['N3'] - row['N3_after_mean'] if not np.isnan(row['N3']) else np.nan
                    }
                    improvements.append(improvement)
        
        self.improvements_df = pd.DataFrame(improvements)
    
    def analyze_results(self):
        """Analyze and visualize mitigation results"""
        print("\n" + "="*80)
        print("MITIGATION STRATEGY ANALYSIS")
        print("="*80)
        
        if len(self.results_df) == 0:
            print("No results to analyze!")
            return
        
        # 1. Overall technique performance
        print("\n1. AVERAGE PERFORMANCE BY TECHNIQUE")
        print("-"*40)
        technique_summary = self.results_df.groupby('technique').agg({
            'f1_mean': ['mean', 'std'],
            'auc_mean': ['mean', 'std']
        }).round(4)
        print(technique_summary)
        
        # 2. Best technique per overlap characteristic
        print("\n\n2. BEST TECHNIQUE BY OVERLAP LEVEL")
        print("-"*40)
        
        # Categorize datasets by N1 level
        valid_n1 = self.results_df.dropna(subset=['N1'])
        if len(valid_n1) > 0:
            self.results_df['N1_category'] = pd.cut(
                self.results_df['N1'], 
                bins=[0, 0.15, 0.20, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            n1_data = self.results_df.dropna(subset=['N1_category'])
            
            if len(n1_data) > 0:
                best_by_n1 = n1_data.groupby(['N1_category', 'technique'])['f1_mean'].mean().reset_index()
                
                for category in ['Low', 'Medium', 'High']:
                    cat_data = best_by_n1[best_by_n1['N1_category'] == category]
                    if len(cat_data) > 0:
                        best_row = cat_data.loc[cat_data['f1_mean'].idxmax()]
                        print(f"  {category} N1: {best_row['technique']} (F1 = {best_row['f1_mean']:.4f})")
        
        # 3. Improvement statistics
        if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
            print("\n\n3. IMPROVEMENT STATISTICS")
            print("-"*40)
            improvement_summary = self.improvements_df.groupby('technique').agg({
                'f1_improvement': ['mean', 'std', 'min', 'max'],
                'N1_reduction': 'mean',
                'N3_reduction': 'mean'
            }).round(2)
            print(improvement_summary)
        
        # 4. Technique-Classifier interactions
        print("\n\n4. TECHNIQUE-CLASSIFIER INTERACTIONS")
        print("-"*40)
        interaction_matrix = self.results_df.pivot_table(
            values='f1_mean',
            index='technique',
            columns='classifier',
            aggfunc='mean'
        ).round(4)
        print(interaction_matrix)
        
        # Save detailed results
        self.save_results()
        
        # Create visualizations
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Warning: Error creating visualizations: {str(e)}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations of mitigation results"""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # F1 scores by technique
        ax = axes[0, 0]
        technique_order = self.results_df.groupby('technique')['f1_mean'].mean().sort_values(ascending=False).index
        data_to_plot = [self.results_df[self.results_df['technique'] == t]['f1_mean'].dropna().values 
                       for t in technique_order]
        ax.boxplot(data_to_plot, labels=technique_order)
        ax.set_title('F1 Score Distribution by Mitigation Technique')
        ax.set_xlabel('Technique')
        ax.set_ylabel('F1 Score')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Improvement heatmap
        ax = axes[0, 1]
        if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
            improvement_matrix = self.improvements_df.pivot_table(
                values='f1_improvement',
                index='technique',
                columns='classifier',
                aggfunc='mean'
            )
            if not improvement_matrix.empty:
                sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                           center=0, ax=ax, cbar_kws={'label': 'F1 Improvement (%)'})
                ax.set_title('F1 Improvement by Technique and Classifier')
        
        # Overlap reduction scatter
        ax = axes[1, 0]
        if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
            valid_data = self.improvements_df.dropna(subset=['N1_reduction', 'f1_improvement'])
            for technique in valid_data['technique'].unique():
                tech_data = valid_data[valid_data['technique'] == technique]
                ax.scatter(tech_data['N1_reduction'], tech_data['f1_improvement'], 
                         label=technique, alpha=0.6, s=50)
            ax.set_xlabel('N1 Reduction')
            ax.set_ylabel('F1 Improvement (%)')
            ax.set_title('F1 Improvement vs N1 Reduction (Paradox Check)')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Classifier performance
        ax = axes[1, 1]
        classifier_perf = self.results_df.groupby('classifier')['f1_mean'].mean().sort_values(ascending=True)
        classifier_perf.plot(kind='barh', ax=ax)
        ax.set_title('Mean F1 Score by Classifier')
        ax.set_xlabel('F1 Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'mitigation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.results_path}/")
    
    def save_results(self):
        """Save all results to CSV files"""
        # Save main results
        self.results_df.to_csv(os.path.join(self.results_path, 'mitigation_results.csv'), index=False)
        
        # Save improvements
        if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
            self.improvements_df.to_csv(os.path.join(self.results_path, 'mitigation_improvements.csv'), index=False)
        
        # Save summary report
        self.save_summary_report()
        
        print(f"Results saved to {self.results_path}/")
    
    def save_summary_report(self):
        """Save comprehensive summary report"""
        report_path = os.path.join(self.results_path, 'mitigation_summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MITIGATION STRATEGY COMPARISON - SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Datasets: {self.results_df['dataset'].nunique()}\n")
            f.write(f"Techniques: {len(self.mitigation_techniques)}\n")
            f.write(f"Classifiers: {len(self.classifiers)}\n")
            f.write(f"Total experiments: {len(self.results_df)}\n\n")
            
            f.write("TECHNIQUE PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n")
            technique_summary = self.results_df.groupby('technique')['f1_mean'].agg(['mean', 'std']).round(4)
            technique_summary = technique_summary.sort_values('mean', ascending=False)
            f.write(technique_summary.to_string())
            f.write("\n\n")
            
            if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
                f.write("IMPROVEMENT SUMMARY\n")
                f.write("-"*40 + "\n")
                imp_summary = self.improvements_df.groupby('technique')['f1_improvement'].agg(['mean', 'std', 'min', 'max']).round(2)
                imp_summary = imp_summary.sort_values('mean', ascending=False)
                f.write(imp_summary.to_string())
                f.write("\n\n")
                
                # Paradox check
                f.write("OVERLAP REDUCTION PARADOX CHECK\n")
                f.write("-"*40 + "\n")
                valid = self.improvements_df.dropna(subset=['N1_reduction', 'f1_improvement'])
                if len(valid) > 10:
                    from scipy import stats
                    r, p = stats.pearsonr(valid['N1_reduction'], valid['f1_improvement'])
                    f.write(f"Correlation (N1 reduction vs F1 improvement): r = {r:.3f}, p = {p:.4f}\n")
                    if r < -0.3 and p < 0.05:
                        f.write("PARADOX CONFIRMED: Greater overlap reduction correlates with worse performance\n")
                    elif r > 0.3 and p < 0.05:
                        f.write("Overlap reduction correlates with better performance\n")
                    else:
                        f.write("No significant correlation between overlap reduction and performance\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")


def main():
    """Main execution function for mitigation comparison"""
    
    # Initialize the framework - UPDATE PATH FOR YOUR SYSTEM
    datasets_path = "Insert the path"
    mitigation = MitigationComparison(datasets_path)
    
    print("="*80)
    print("COMPREHENSIVE MITIGATION STRATEGY COMPARISON")
    print("="*80)
    
    # Run experiments on ALL datasets
    print("\nStep 1: Running mitigation experiments...")
    results = mitigation.run_experiments()
    
    # Analyze results
    print("\nStep 2: Analyzing results...")
    mitigation.analyze_results()
    
    print("\n" + "="*80)
    print("MITIGATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {mitigation.results_path}/")
    print("\nKey outputs:")
    print("- mitigation_results.csv: Full experimental results")
    print("- mitigation_improvements.csv: Improvement metrics")
    print("- mitigation_summary_report.txt: Comprehensive summary")
    print("- mitigation_comparison.png: Main visualization")


if __name__ == "__main__":
    main()