import pandas as pd
import numpy as np
from datetime import datetime
import sys

class QuickAnalysis:
    def __init__(self, output_file='analysis_output.txt'):
        self.output_file = output_file
        self.output_lines = []
        
    def log(self, text=""):
        """Print to console and save to list for file output"""
        print(text)
        self.output_lines.append(text)
    
    def save_output(self):
        """Save all output to text file"""
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(self.output_lines))
        print(f"\nOutput saved to {self.output_file}")
    
    def run_analysis(self):
        """Run quick analysis of existing results"""
        self.log("="*80)
        self.log("CLASSIFICATION RESULTS ANALYSIS")
        self.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*80)
        
        try:
            # Load data
            self.log("\nLoading results...")
            combined_df = pd.read_csv('classification_results/combined_overlap_performance.csv')
            regression_df = pd.read_csv('classification_results/regression_model_performance.csv')
            
            # 1. Best classifiers
            self.log("\n1. BEST PERFORMING CLASSIFIERS")
            self.log("-"*40)
            metrics = ['accuracy', 'f1', 'auc']
            for metric in metrics:
                top_clf = combined_df.groupby('classifier')[metric].mean().sort_values(ascending=False).head(3)
                self.log(f"\nTop 3 by {metric.upper()}:")
                for clf, score in top_clf.items():
                    self.log(f"  {clf}: {score:.4f}")
            
            # 2. Worst performing datasets
            self.log("\n\n2. DATASETS NEEDING MITIGATION")
            self.log("-"*40)
            worst_datasets = combined_df.groupby('dataset').agg({
                'f1': 'mean',
                'N1': 'first',
                'N3': 'first',
                'imbalance_ratio': 'first'
            }).sort_values('f1').head(10)
            
            self.log("\nWorst 10 datasets by F1-score:")
            for dataset, row in worst_datasets.iterrows():
                self.log(f"\n{dataset}:")
                self.log(f"  F1: {row['f1']:.3f}, N1: {row['N1']:.3f}, N3: {row['N3']:.3f}, Imbalance: {row['imbalance_ratio']:.1f}")
                
                # Recommend mitigation
                if row['N1'] > 0.3:
                    self.log("  → Recommend: CBO (high borderline ratio)")
                if row['N3'] > 0.4:
                    self.log("  → Recommend: Noise filtering (high 1-NN error)")
                if row['imbalance_ratio'] > 10:
                    self.log("  → Recommend: Advanced resampling")
            
            # 3. Most predictive overlap metrics
            self.log("\n\n3. MOST PREDICTIVE OVERLAP METRICS")
            self.log("-"*40)
            best_models = regression_df[regression_df['regression_model'] == 'Random Forest'].groupby('target_metric')['r2'].mean()
            self.log("\nRandom Forest R² scores for predicting:")
            for metric, r2 in best_models.items():
                self.log(f"  {metric}: {r2:.3f}")
            
            # 4. Critical thresholds
            self.log("\n\n4. CRITICAL OVERLAP THRESHOLDS")
            self.log("-"*40)
            
            # Find N1 threshold where average F1 < 0.7
            n1_groups = pd.qcut(combined_df['N1'], q=10, duplicates='drop')
            group_perf = combined_df.groupby(n1_groups)['f1'].mean()
            critical_groups = group_perf[group_perf < 0.7]
            if not critical_groups.empty:
                critical_n1 = critical_groups.index[0].left
                self.log(f"\nN1 > {critical_n1:.3f}: Average F1 drops below 0.7")
            
            # Same for N3
            n3_groups = pd.qcut(combined_df['N3'], q=10, duplicates='drop')
            group_perf = combined_df.groupby(n3_groups)['f1'].mean()
            critical_groups = group_perf[group_perf < 0.7]
            if not critical_groups.empty:
                critical_n3 = critical_groups.index[0].left
                self.log(f"N3 > {critical_n3:.3f}: Average F1 drops below 0.7")
            
            # 5. Summary recommendations
            self.log("\n\n5. RECOMMENDATIONS FOR NEXT PHASE")
            self.log("-"*40)
            self.log("\nPriority order for mitigation strategies:")
            self.log("1. Implement CBO for datasets with N1 > 0.3")
            self.log("2. Apply noise filtering for datasets with N3 > 0.4")
            self.log("3. Use hybrid approaches for datasets with both high N1 and N3")
            self.log("4. Consider advanced resampling for extreme imbalance (ratio > 10)")
            
            self.log("\nExpected improvements:")
            self.log("- CBO: 5-15% improvement in F1-score")
            self.log("- Noise filtering: 3-10% improvement")
            self.log("- Hybrid approach: 10-20% improvement")
            
        except Exception as e:
            self.log(f"\nError during analysis: {str(e)}")
            self.log("Make sure you have run the classification experiments first.")
        
        # Save output
        self.save_output()


if __name__ == "__main__":
    # Run analysis
    analyzer = QuickAnalysis('classification_analysis_output.txt')
    analyzer.run_analysis()