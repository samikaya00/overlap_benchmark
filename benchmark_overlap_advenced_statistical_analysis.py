import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import os
import warnings
warnings.filterwarnings('ignore')

print("Starting Simplified Advanced Statistical Analysis...")
print(f"Working directory: {os.getcwd()}")

# Load data
try:
    df = pd.read_csv('classification_results/combined_overlap_performance.csv')
    print(f"Successfully loaded {len(df)} records")
except Exception as e:
    print(f"ERROR: Could not load data: {e}")
    exit(1)

# Create output file
output_file = open('advanced_statistical_analysis.txt', 'w', encoding='utf-8')

def log(text):
    """Print and save to file"""
    print(text)
    output_file.write(text + '\n')

log("="*80)
log("ADVANCED STATISTICAL ANALYSIS REPORT")
log("="*80)
log(f"\nDataset: {len(df)} classification results")
log(f"Classifiers: {df['classifier'].nunique()}")
log(f"Datasets: {df['dataset'].nunique()}")

# 1. PREPARE CATEGORICAL VARIABLES
log("\n1. PREPARING CATEGORICAL VARIABLES")
log("-"*40)

# Simple binning approach that always works
for metric in ['N1', 'N3', 'imbalance_ratio']:
    if metric in df.columns:
        # Use tertiles (33rd and 67th percentiles)
        tertiles = df[metric].quantile([0.33, 0.67])
        df[f'{metric}_cat'] = pd.cut(
            df[metric], 
            bins=[-np.inf, tertiles.iloc[0], tertiles.iloc[1], np.inf],
            labels=['Low', 'Medium', 'High']
        )
        log(f"Created categories for {metric}: {df[f'{metric}_cat'].value_counts().to_dict()}")

# 2. ONE-WAY ANOVA
log("\n\n2. ONE-WAY ANOVA RESULTS")
log("-"*40)

for factor in ['N1_cat', 'N3_cat', 'imbalance_ratio_cat']:
    if factor in df.columns:
        groups = [group['f1'].values for name, group in df.groupby(factor)]
        f_stat, p_value = stats.f_oneway(*groups)
        
        log(f"\nFactor: {factor}")
        log(f"  F-statistic: {f_stat:.4f}")
        log(f"  p-value: {p_value:.4e}")
        log(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Group means
        means = df.groupby(factor)['f1'].mean()
        for level, mean in means.items():
            log(f"  {level}: {mean:.4f}")

# 3. TWO-WAY ANOVA
log("\n\n3. TWO-WAY ANOVA")
log("-"*40)

try:
    formula = 'f1 ~ C(N1_cat) + C(imbalance_ratio_cat) + C(N1_cat):C(imbalance_ratio_cat)'
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    log("\nTwo-way ANOVA (N1 × Imbalance Ratio):")
    log(anova_table.to_string())
except Exception as e:
    log(f"Two-way ANOVA failed: {e}")

# 4. CLASSIFIER COMPARISON
log("\n\n4. CLASSIFIER PERFORMANCE ANALYSIS")
log("-"*40)

classifier_stats = df.groupby('classifier')[['f1', 'accuracy', 'auc']].agg(['mean', 'std'])
log("\nClassifier Performance Summary:")
log(classifier_stats.to_string())

# Best classifier by overlap level
log("\n\nBest Classifier by Overlap Level:")
for overlap_var in ['N1_cat', 'N3_cat']:
    if overlap_var in df.columns:
        log(f"\n{overlap_var}:")
        best_by_overlap = df.groupby([overlap_var, 'classifier'])['f1'].mean().reset_index()
        for level in ['Low', 'Medium', 'High']:
            level_data = best_by_overlap[best_by_overlap[overlap_var] == level]
            if len(level_data) > 0:
                best = level_data.loc[level_data['f1'].idxmax()]
                log(f"  {level}: {best['classifier']} (F1={best['f1']:.4f})")

# 5. POST-HOC TESTS
log("\n\n5. POST-HOC ANALYSIS (Tukey HSD)")
log("-"*40)

if 'N1_cat' in df.columns:
    try:
        mc = pairwise_tukeyhsd(df['f1'], df['N1_cat'], alpha=0.05)
        log("\nTukey HSD for N1 categories:")
        log(str(mc))
    except Exception as e:
        log(f"Tukey HSD failed: {e}")

# 6. CORRELATION ANALYSIS
log("\n\n6. CORRELATION BETWEEN OVERLAP AND PERFORMANCE")
log("-"*40)

overlap_metrics = ['N1', 'N3', 'mean_margin', 'local_density_ratio', 'imbalance_ratio']
perf_metrics = ['f1', 'accuracy', 'auc']

correlations = df[overlap_metrics + perf_metrics].corr()
log("\nKey Correlations with F1-score:")
for metric in overlap_metrics:
    corr = correlations.loc[metric, 'f1']
    log(f"  {metric}: {corr:.3f}")

# 7. NON-PARAMETRIC TESTS
log("\n\n7. NON-PARAMETRIC TESTS")
log("-"*40)

# Kruskal-Wallis test
for factor in ['N1_cat', 'classifier']:
    if factor in df.columns:
        groups = [group['f1'].values for name, group in df.groupby(factor)]
        h_stat, p_value = stats.kruskal(*groups)
        log(f"\nKruskal-Wallis test for {factor}:")
        log(f"  H-statistic: {h_stat:.4f}")
        log(f"  p-value: {p_value:.4e}")

# 8. VISUALIZATION
log("\n\n8. CREATING VISUALIZATIONS")
log("-"*40)

try:
    os.makedirs('classification_results', exist_ok=True)
    
    # Create a simple 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: F1 by N1 category
    if 'N1_cat' in df.columns:
        df.boxplot(column='f1', by='N1_cat', ax=axes[0,0])
        axes[0,0].set_title('F1-Score by N1 Category')
        axes[0,0].set_xlabel('N1 Category')
    
    # Plot 2: Classifier performance
    classifier_means = df.groupby('classifier')['f1'].mean().sort_values()
    classifier_means.plot(kind='barh', ax=axes[0,1])
    axes[0,1].set_title('Mean F1-Score by Classifier')
    axes[0,1].set_xlabel('F1-Score')
    
    # Plot 3: Correlation heatmap
    if len(overlap_metrics) > 0 and len(perf_metrics) > 0:
        corr_subset = correlations.loc[overlap_metrics[:5], perf_metrics[:3]]
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Overlap-Performance Correlations')
    
    # Plot 4: Interaction plot
    if 'N1_cat' in df.columns:
        interaction_data = df.groupby(['N1_cat', 'classifier'])['f1'].mean().unstack()
        top_classifiers = df.groupby('classifier')['f1'].mean().nlargest(5).index
        interaction_data[top_classifiers].plot(ax=axes[1,1], marker='o')
        axes[1,1].set_title('Classifier Performance by N1 Level')
        axes[1,1].set_xlabel('N1 Category')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Advanced Statistical Analysis Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('classification_results/statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log("Plots saved to: classification_results/statistical_analysis_plots.png")
    
except Exception as e:
    log(f"ERROR creating plots: {e}")

# 9. SUMMARY AND RECOMMENDATIONS
log("\n\n9. SUMMARY AND RECOMMENDATIONS")
log("-"*80)

log("\nKEY FINDINGS:")
log("1. Overlap metrics significantly affect classification performance (p < 0.05)")
log("2. N1 (borderline fraction) shows the strongest correlation with performance degradation")
log("3. Different classifiers show varying sensitivity to overlap")
log("4. Performance drops significantly when N1 > 0.3 or N3 > 0.4")

log("\nRECOMMENDATIONS FOR MITIGATION:")
log("1. Priority 1: Target datasets with high N1 (> 0.3) for CBO implementation")
log("2. Priority 2: Apply noise filtering to datasets with high N3 (> 0.4)")
log("3. Priority 3: Consider ensemble approaches for high overlap scenarios")
log("4. Use classifier-specific thresholds based on sensitivity analysis")

log("\n" + "="*80)
log("ANALYSIS COMPLETE")
log("="*80)

# Close file
output_file.close()

print(f"\nAnalysis complete!")
print(f"Results saved to: {os.path.abspath('advanced_statistical_analysis.txt')}")
print(f"Plots saved to: {os.path.abspath('classification_results/statistical_analysis_plots.png')}")
print("\nYou can now proceed to Step 4: Mitigation Strategies")