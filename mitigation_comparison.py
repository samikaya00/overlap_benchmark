def create_visualizations(self):
        """Create comprehensive visualizations of mitigation results"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        
        try:
            # 1. Technique comparison boxplot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # F1 scores by technique
            ax = axes[0, 0]
            technique_order = self.results_df.groupby('technique')['f1_mean'].mean().sort_values(ascending=False).index
            
            # Create boxplot manually to avoid pandas boxplot issues
            data_to_plot = []
            labels = []
            for tech in technique_order:
                tech_data = self.results_df[self.results_df['technique'] == tech]['f1_mean'].dropna()
                if len(tech_data) > 0:
                    data_to_plot.append(tech_data.values)
                    labels.append(tech)
            
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title('F1 Score Distribution by Mitigation Technique')
                ax.set_xlabel('Technique')
                ax.set_ylabel('F1 Score')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Improvement heatmap
            ax = axes[0, 1]
            if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
                improvement_matrix = self.improvements_df.pivot_table(
                    values='f1_improvement',
                    index='dataset',
                    columns='technique',
                    aggfunc='mean'
                )
                if not improvement_matrix.empty:
                    sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                               center=0, ax=ax, cbar_kws={'label': 'F1 Improvement (%)'})
                    ax.set_title('F1 Score Improvement by Dataset and Technique')
            
            # Overlap reduction scatter
            ax = axes[1, 0]
            if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
                for technique in self.improvements_df['technique'].unique()[:10]:  # Limit to 10 techniques
                    tech_data = self.improvements_df[self.improvements_df['technique'] == technique]
                    if len(tech_data) > 0:
                        ax.scatter(tech_data['N1_reduction'], tech_data['f1_improvement'], 
                                 label=technique, alpha=0.6, s=50)
                ax.set_xlabel('N1 Reduction')
                ax.set_ylabel('F1 Improvement (%)')
                ax.set_title('F1 Improvement vs N1 Reduction')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Technique ranking by dataset characteristics
            ax = axes[1, 1]
            if 'N1_category' in self.results_df.columns:
                # Handle NaN values in N1_category
                ranking_data = self.results_df.dropna(subset=['N1_category']).groupby(
                    ['N1_category', 'technique']
                )['f1_mean'].mean().unstack()
                
                if not ranking_data.empty:
                    ranking_data.plot(kind='bar', ax=ax)
                    ax.set_title('Average F1 Score by N1 Level and Technique')
                    ax.set_xlabel('N1 Level')
                    ax.set_ylabel('F1 Score')
                    ax.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_path, 'mitigation_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Create technique effectiveness radar chart
            self.create_radar_chart()
            
            # 3. Create improvement distribution plot
            self.create_improvement_distribution()
            
            print(f"Visualizations saved to {self.results_path}/")
            
        except Exception as e:
            print(f"Error creating main visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            
import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

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
        # Separate minority and majority classes
        X_min = X[y == 1]
        X_maj = X[y == 0]
        
        # Determine number of clusters (at least 2, at most k_clusters)
        n_clusters = min(self.k_clusters, max(2, len(X_min) // 10))
        
        # Cluster minority samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_min)
        
        # Calculate borderline ratio for each cluster
        cluster_borderline_ratios = {}
        knn = NearestNeighbors(n_neighbors=6)
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
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = X_min[cluster_mask]
            
            if len(cluster_samples) < 2:
                continue
                
            # Samples to generate for this cluster
            weight = cluster_borderline_ratios[cluster_id]
            n_synthetic = int(n_synthetic_total * weight / sum(cluster_borderline_ratios.values()))
            
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
        
        # Classifiers to test
        self.classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'NaiveBayes': GaussianNB()
        }
        
    def compute_overlap_metrics(self, X, y):
        """Compute overlap metrics for the dataset"""
        metrics = {}
        
        # N1: Fraction of borderline points
        knn = NearestNeighbors(n_neighbors=6)
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
        """Parse KEEL .dat file format"""
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
        y = []
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('@'):
                values = line.split(',')
                if len(values) > 1:
                    features = []
                    for v in values[:-1]:
                        try:
                            features.append(float(v))
                        except ValueError:
                            features.append(np.nan)
                    
                    # Handle class label
                    label = values[-1].strip()
                    if label in ['positive', 'Positive', '1', '1.0']:
                        y.append(1)
                    else:
                        y.append(0)
                    
                    X.append(features)
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle missing values
        if np.any(np.isnan(X)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
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
                print(f"Error with {technique_name}: {str(e)}")
                X_resampled, y_resampled = X_train, y_train
        else:
            X_resampled, y_resampled = X_train, y_train
        
        # Train classifier
        classifier_copy = classifier.__class__(**classifier.get_params())
        classifier_copy.fit(X_resampled, y_resampled)
        
        # Evaluate
        y_pred = classifier_copy.predict(X_test)
        y_proba = classifier_copy.predict_proba(X_test)[:, 1] if hasattr(classifier_copy, 'predict_proba') else None
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        
        if y_proba is not None:
            auc_score = roc_auc_score(y_test, y_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            auprc = auc(recall, precision)
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
        """Run mitigation experiments on specified datasets"""
        # Load previous results to identify worst performers
        if dataset_names is None:
            # Use the 10 worst performing datasets from previous analysis
            dataset_names = [
                '03subcl5-600-5-70-BI',
                '04clover5z-600-5-70-BI',
                '03subcl5-600-5-60-BI',
                '04clover5z-600-5-60-BI',
                '03subcl5-600-5-50-BI',
                '04clover5z-600-5-50-BI',
                'paw02a-600-5-70-BI',
                'paw02a-600-5-60-BI',
                '03subcl5-600-5-30-BI',
                '03subcl5-800-7-70-BI'
            ]
        
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
                    print(f"Dataset {dataset_name} not found, skipping...")
                    pbar.update(len(self.mitigation_techniques) * len(self.classifiers))
                    continue
                
                try:
                    X, y = self.parse_keel_file(filepath)
                    
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
                                
                                # Evaluate mitigation
                                fold_result = self.evaluate_mitigation(
                                    X_train, y_train, X_test, y_test,
                                    technique_name, classifier_name
                                )
                                fold_results.append(fold_result)
                            
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
                    print(f"Error processing {dataset_name}: {str(e)}")
                    pbar.update(len(self.mitigation_techniques) * len(self.classifiers))
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        # Calculate improvements
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
                    
                    improvement = {
                        'dataset': row['dataset'],
                        'technique': row['technique'],
                        'classifier': row['classifier'],
                        'f1_improvement': (row['f1_mean'] - baseline_f1) / baseline_f1 * 100,
                        'f1_improvement_abs': row['f1_mean'] - baseline_f1,
                        'auc_improvement': (row['auc_mean'] - baseline_auc) / baseline_auc * 100,
                        'auc_improvement_abs': row['auc_mean'] - baseline_auc,
                        'N1_reduction': row['N1'] - row['N1_after_mean'],
                        'N3_reduction': row['N3'] - row['N3_after_mean']
                    }
                    improvements.append(improvement)
        
        self.improvements_df = pd.DataFrame(improvements)
    
    def analyze_results(self):
        """Analyze and visualize mitigation results"""
        print("\n" + "="*80)
        print("MITIGATION STRATEGY ANALYSIS")
        print("="*80)
        
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
        # Handle potential NaN values in N1
        self.results_df['N1_category'] = pd.cut(
            self.results_df['N1'], 
            bins=[0, 0.15, 0.20, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Drop rows with NaN categories before grouping
        n1_data = self.results_df.dropna(subset=['N1_category'])
        
        if len(n1_data) > 0:
            best_by_n1 = n1_data.groupby(['N1_category', 'technique'])['f1_mean'].mean().reset_index()
            
            # Get best technique for each category
            best_techniques = []
            for category in ['Low', 'Medium', 'High']:  # Use explicit category list
                cat_data = best_by_n1[best_by_n1['N1_category'] == category]
                if len(cat_data) > 0 and not cat_data['f1_mean'].isna().all():
                    # Find the row with maximum f1_mean
                    max_f1 = cat_data['f1_mean'].max()
                    best_row = cat_data[cat_data['f1_mean'] == max_f1].iloc[0]
                    best_techniques.append({
                        'N1_category': category,
                        'technique': best_row['technique'],
                        'f1_mean': best_row['f1_mean']
                    })
            
            if best_techniques:
                print("\nBest technique by N1 level:")
                for item in best_techniques:
                    print(f"  {item['N1_category']}: {item['technique']} (F1 = {item['f1_mean']:.4f})")
            else:
                print("No valid N1 categories found for analysis")
        else:
            print("No data available for N1 category analysis")
        
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
        
        # Create visualizations with error handling
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Warning: Error creating visualizations: {str(e)}")
            print("Continuing with analysis...")
    
    def create_visualizations(self):
        """Create comprehensive visualizations of mitigation results"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Technique comparison boxplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # F1 scores by technique
        ax = axes[0, 0]
        technique_order = self.results_df.groupby('technique')['f1_mean'].mean().sort_values(ascending=False).index
        self.results_df.boxplot(column='f1_mean', by='technique', ax=ax, rot=45)
        ax.set_title('F1 Score Distribution by Mitigation Technique')
        ax.set_xlabel('Technique')
        ax.set_ylabel('F1 Score')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Improvement heatmap
        ax = axes[0, 1]
        if hasattr(self, 'improvements_df'):
            improvement_matrix = self.improvements_df.pivot_table(
                values='f1_improvement',
                index='dataset',
                columns='technique',
                aggfunc='mean'
            )
            sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=ax, cbar_kws={'label': 'F1 Improvement (%)'})
            ax.set_title('F1 Score Improvement by Dataset and Technique')
        
        # Overlap reduction scatter
        ax = axes[1, 0]
        if hasattr(self, 'improvements_df'):
            for technique in self.improvements_df['technique'].unique():
                tech_data = self.improvements_df[self.improvements_df['technique'] == technique]
                ax.scatter(tech_data['N1_reduction'], tech_data['f1_improvement'], 
                         label=technique, alpha=0.6, s=50)
            ax.set_xlabel('N1 Reduction')
            ax.set_ylabel('F1 Improvement (%)')
            ax.set_title('F1 Improvement vs N1 Reduction')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Technique ranking by dataset characteristics
        ax = axes[1, 1]
        if 'N1_category' in self.results_df.columns:
            ranking_data = self.results_df.groupby(['N1_category', 'technique'])['f1_mean'].mean().unstack()
            ranking_data.plot(kind='bar', ax=ax)
            ax.set_title('Average F1 Score by N1 Level and Technique')
            ax.set_xlabel('N1 Level')
            ax.set_ylabel('F1 Score')
            ax.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'mitigation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create technique effectiveness radar chart
        self.create_radar_chart()
        
        # 3. Create improvement distribution plot
        self.create_improvement_distribution()
        
        print(f"Visualizations saved to {self.results_path}/")
    
    def create_radar_chart(self):
        """Create radar chart showing technique effectiveness across different metrics"""
        if not hasattr(self, 'improvements_df'):
            return
            
        # Calculate average metrics for each technique
        metrics = ['f1_improvement', 'N1_reduction', 'N3_reduction']
        technique_metrics = self.improvements_df.groupby('technique')[metrics].mean()
        
        # Normalize metrics to 0-1 scale for radar chart
        for col in metrics:
            max_val = technique_metrics[col].max()
            min_val = technique_metrics[col].min()
            if max_val > min_val:
                technique_metrics[col] = (technique_metrics[col] - min_val) / (max_val - min_val)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for technique in technique_metrics.index[:6]:  # Top 6 techniques
            values = technique_metrics.loc[technique].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=technique)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['F1 Improvement', 'N1 Reduction', 'N3 Reduction'])
        ax.set_ylim(0, 1)
        ax.set_title('Mitigation Technique Effectiveness\n(Normalized Metrics)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'technique_effectiveness_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_improvement_distribution(self):
        """Create distribution plot of improvements"""
        if not hasattr(self, 'improvements_df'):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # F1 improvement distribution
        ax = axes[0]
        for technique in self.improvements_df['technique'].unique():
            tech_data = self.improvements_df[self.improvements_df['technique'] == technique]
            ax.hist(tech_data['f1_improvement'], bins=20, alpha=0.5, label=technique, density=True)
        
        ax.set_xlabel('F1 Improvement (%)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of F1 Score Improvements')
        ax.legend(fontsize=8)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Cumulative improvement
        ax = axes[1]
        techniques = self.improvements_df.groupby('technique')['f1_improvement'].mean().sort_values(ascending=False).index
        cumulative_data = []
        
        for tech in techniques:
            tech_improvements = self.improvements_df[self.improvements_df['technique'] == tech]['f1_improvement'].values
            cumulative_data.append({
                'technique': tech,
                'mean': np.mean(tech_improvements),
                'positive_rate': np.sum(tech_improvements > 0) / len(tech_improvements) * 100
            })
        
        cumulative_df = pd.DataFrame(cumulative_data)
        x = range(len(cumulative_df))
        
        ax.bar(x, cumulative_df['positive_rate'], alpha=0.6, label='% Positive Improvement')
        ax.set_xticks(x)
        ax.set_xticklabels(cumulative_df['technique'], rotation=45, ha='right')
        ax.set_ylabel('Percentage')
        ax.set_title('Success Rate of Mitigation Techniques')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'improvement_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results to CSV files"""
        # Save main results
        self.results_df.to_csv(os.path.join(self.results_path, 'mitigation_results.csv'), index=False)
        
        # Save improvements
        if hasattr(self, 'improvements_df'):
            self.improvements_df.to_csv(os.path.join(self.results_path, 'mitigation_improvements.csv'), index=False)
        
        # Save summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        with open(os.path.join(self.results_path, 'mitigation_summary_report.txt'), 'w') as f:
            f.write("="*80 + "\n")
            f.write("MITIGATION STRATEGY COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Dataset summary
            f.write("EXPERIMENT SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Datasets tested: {len(self.results_df['dataset'].unique())}\n")
            f.write(f"Mitigation techniques: {len(self.results_df['technique'].unique())}\n")
            f.write(f"Classifiers tested: {len(self.results_df['classifier'].unique())}\n")
            f.write(f"Total experiments: {len(self.results_df)}\n\n")
            
            # Best overall techniques
            f.write("TOP 5 MITIGATION TECHNIQUES (by F1 improvement)\n")
            f.write("-"*40 + "\n")
            if hasattr(self, 'improvements_df'):
                top_techniques = self.improvements_df.groupby('technique').agg({
                    'f1_improvement': ['mean', 'std'],
                    'f1_improvement_abs': 'mean'
                }).round(2)
                top_techniques.columns = ['Mean %', 'Std %', 'Absolute']
                top_techniques = top_techniques.sort_values('Mean %', ascending=False).head(5)
                f.write(top_techniques.to_string())
                f.write("\n\n")
            
            # Technique recommendations by overlap type
            f.write("RECOMMENDATIONS BY OVERLAP CHARACTERISTICS\n")
            f.write("-"*40 + "\n")
            
            # High N1 datasets
            high_n1_data = self.results_df[self.results_df['N1'] > 0.15]
            if len(high_n1_data) > 0:
                best_for_high_n1 = high_n1_data.groupby('technique')['f1_mean'].mean().sort_values(ascending=False).head(3)
                f.write("\nFor High N1 (> 0.15):\n")
                for tech, score in best_for_high_n1.items():
                    f.write(f"  {tech}: F1 = {score:.4f}\n")
            
            # High N3 datasets
            high_n3_data = self.results_df[self.results_df['N3'] > 0.20]
            if len(high_n3_data) > 0:
                best_for_high_n3 = high_n3_data.groupby('technique')['f1_mean'].mean().sort_values(ascending=False).head(3)
                f.write("\nFor High N3 (> 0.20):\n")
                for tech, score in best_for_high_n3.items():
                    f.write(f"  {tech}: F1 = {score:.4f}\n")
            
            # Classifier-specific recommendations
            f.write("\n\nCLASSIFIER-SPECIFIC BEST TECHNIQUES\n")
            f.write("-"*40 + "\n")
            for classifier in self.results_df['classifier'].unique():
                clf_data = self.results_df[self.results_df['classifier'] == classifier]
                best_tech = clf_data.groupby('technique')['f1_mean'].mean().idxmax()
                best_score = clf_data.groupby('technique')['f1_mean'].mean().max()
                f.write(f"\n{classifier}:\n")
                f.write(f"  Best: {best_tech} (F1 = {best_score:.4f})\n")
            
            # Statistical significance
            f.write("\n\nSTATISTICAL SIGNIFICANCE\n")
            f.write("-"*40 + "\n")
            if hasattr(self, 'improvements_df'):
                significant_improvements = self.improvements_df[self.improvements_df['f1_improvement'] > 5]
                sig_rate = len(significant_improvements) / len(self.improvements_df) * 100
                f.write(f"Experiments with >5% F1 improvement: {sig_rate:.1f}%\n")
                
                # Best improvement cases
                best_improvements = self.improvements_df.nlargest(5, 'f1_improvement')
                f.write("\nTop 5 improvement cases:\n")
                for _, row in best_improvements.iterrows():
                    f.write(f"  {row['dataset']} + {row['technique']} + {row['classifier']}: "
                           f"{row['f1_improvement']:.1f}% improvement\n")
            
            # Overlap reduction effectiveness
            f.write("\n\nOVERLAP REDUCTION EFFECTIVENESS\n")
            f.write("-"*40 + "\n")
            overlap_reduction = self.results_df.groupby('technique').agg({
                'N1': 'mean',
                'N1_after_mean': 'mean',
                'N3': 'mean',
                'N3_after_mean': 'mean'
            })
            overlap_reduction['N1_reduction_%'] = (overlap_reduction['N1'] - overlap_reduction['N1_after_mean']) / overlap_reduction['N1'] * 100
            overlap_reduction['N3_reduction_%'] = (overlap_reduction['N3'] - overlap_reduction['N3_after_mean']) / overlap_reduction['N3'] * 100
            
            f.write("Average overlap reduction by technique:\n")
            for tech in overlap_reduction.index:
                if tech != 'None':
                    f.write(f"\n{tech}:\n")
                    f.write(f"  N1 reduction: {overlap_reduction.loc[tech, 'N1_reduction_%']:.1f}%\n")
                    f.write(f"  N3 reduction: {overlap_reduction.loc[tech, 'N3_reduction_%']:.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")


def main():
    """Main execution function for mitigation comparison"""
    
    # Initialize the framework
    datasets_path = r"C:\Users\SAMI\.vscode\overlap_datasets"
    mitigation = MitigationComparison(datasets_path)
    
    print("="*80)
    print("COMPREHENSIVE MITIGATION STRATEGY COMPARISON")
    print("="*80)
    
    # Run experiments on worst performing datasets
    print("\nStep 1: Running mitigation experiments...")
    results = mitigation.run_experiments()
    
    # Analyze results
    print("\nStep 2: Analyzing results...")
    mitigation.analyze_results()
    
    # Generate recommendations
    print("\nStep 3: Generating recommendations...")
    generate_recommendations(mitigation)
    
    print("\n" + "="*80)
    print("MITIGATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {mitigation.results_path}/")
    print("\nKey outputs:")
    print("- mitigation_results.csv: Full experimental results")
    print("- mitigation_improvements.csv: Improvement metrics")
    print("- mitigation_summary_report.txt: Comprehensive summary")
    print("- mitigation_comparison.png: Main visualization")
    print("- technique_effectiveness_radar.png: Radar chart")
    print("- improvement_distributions.png: Distribution analysis")
    

def generate_recommendations(mitigation):
    """Generate specific recommendations based on results"""
    print("\n" + "="*60)
    print("MITIGATION STRATEGY RECOMMENDATIONS")
    print("="*60)
    
    if not hasattr(mitigation, 'improvements_df'):
        print("No improvement data available")
        return
    
    # Find best techniques overall
    best_overall = mitigation.improvements_df.groupby('technique')['f1_improvement'].mean().sort_values(ascending=False)
    
    print("\n1. OVERALL BEST TECHNIQUES")
    print("-"*30)
    for i, (tech, improvement) in enumerate(best_overall.head(3).items()):
        print(f"{i+1}. {tech}: Average {improvement:.1f}% F1 improvement")
    
    # Recommendations by overlap level
    print("\n2. OVERLAP-SPECIFIC RECOMMENDATIONS")
    print("-"*30)
    
    # High N1 (borderline samples)
    high_n1_results = mitigation.results_df[mitigation.results_df['N1'] > 0.15]
    if len(high_n1_results) > 0:
        best_n1_tech = high_n1_results.groupby('technique')['f1_mean'].mean().idxmax()
        print(f"High N1 (>0.15): Use {best_n1_tech}")
    
    # High N3 (local overlap)
    high_n3_results = mitigation.results_df[mitigation.results_df['N3'] > 0.20]
    if len(high_n3_results) > 0:
        best_n3_tech = high_n3_results.groupby('technique')['f1_mean'].mean().idxmax()
        print(f"High N3 (>0.20): Use {best_n3_tech}")
    
    # Combined high overlap
    high_both = mitigation.results_df[(mitigation.results_df['N1'] > 0.15) & (mitigation.results_df['N3'] > 0.20)]
    if len(high_both) > 0:
        best_both_tech = high_both.groupby('technique')['f1_mean'].mean().idxmax()
        print(f"High N1 & N3: Use {best_both_tech}")
    
    print("\n3. CLASSIFIER-SPECIFIC PAIRINGS")
    print("-"*30)
    
    # Best technique for each classifier
    for classifier in ['RandomForest', 'XGBoost', 'SVM', 'NaiveBayes']:
        clf_data = mitigation.improvements_df[mitigation.improvements_df['classifier'] == classifier]
        if len(clf_data) > 0:
            best_tech = clf_data.groupby('technique')['f1_improvement'].mean().idxmax()
            best_imp = clf_data.groupby('technique')['f1_improvement'].mean().max()
            print(f"{classifier}: {best_tech} (+{best_imp:.1f}%)")
    
    print("\n4. PRACTICAL GUIDELINES")
    print("-"*30)
    print("• For minimal complexity: Start with SMOTE or BorderlineSMOTE")
    print("• For best results: Use hybrid methods (SMOTEENN, SMOTETomek)")
    print("• For high overlap: Consider CBO or cleaning methods first")
    print("• Always validate: Test multiple techniques on your specific data")
    
    # Save recommendations to file
    with open(os.path.join(mitigation.results_path, 'recommendations.txt'), 'w') as f:
        f.write("MITIGATION STRATEGY RECOMMENDATIONS\n")
        f.write("="*60 + "\n\n")
        
        f.write("QUICK REFERENCE GUIDE\n")
        f.write("-"*30 + "\n")
        f.write("If N1 > 0.3: Use BorderlineSMOTE or CBO\n")
        f.write("If N3 > 0.4: Use Tomek Links or ENN first\n")
        f.write("If both high: Use SMOTEENN or SMOTETomek\n")
        f.write("If unsure: Try BorderlineSMOTE (good general performance)\n\n")
        
        f.write("EXPECTED IMPROVEMENTS\n")
        f.write("-"*30 + "\n")
        for tech, imp in best_overall.head(5).items():
            f.write(f"{tech}: {imp:.1f}% average F1 improvement\n")


if __name__ == "__main__":
    main()