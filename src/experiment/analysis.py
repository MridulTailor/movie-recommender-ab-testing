from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import pandas as pd
import numpy as np

def analyze_ab_test(df_results: pd.DataFrame):
    """
    Performs statistical analysis on A/B test results.
    We are comparing Conversion Rate (Proportion).
    """
    summary = df_results.groupby("group")['converted'].agg(['count', 'sum', 'mean'])
    summary.columns = ['visitors', 'conversions', 'conversion_rate']
    
    control_stats = summary.loc['control']
    treatment_stats = summary.loc['treatment']
    
    print("\n=== Statistical Analysis ===")
    print(summary)
    
    # 1. Z-Test for Proportions
    # H0: p_control = p_treatment
    # H1: p_treatment != p_control (Two-tailed)
    
    nobs = np.array([control_stats['visitors'], treatment_stats['visitors']])
    count = np.array([control_stats['conversions'], treatment_stats['conversions']])
    
    stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
    
    print(f"\nZ-Score: {stat:.4f}")
    print(f"P-Value: {pval:.4f}")
    
    alpha = 0.05
    if pval < alpha:
        print(f"Result is Statistically Significant (p < {alpha})!")
        if treatment_stats['conversion_rate'] > control_stats['conversion_rate']:
            print("Treatment Performed BETTER.")
        else:
            print("Treatment Performed WORSE.")
    else:
        print(f"Result is NOT Statistically Significant (p >= {alpha}). Cannot reject Null Hypothesis.")

    return summary, pval

if __name__ == "__main__":
    # Mock Data for testing
    data = {
        'group': ['control']*1000 + ['treatment']*1000,
        'converted': [0]*900 + [1]*100 + [0]*850 + [1]*150 # 10% vs 15%
    }
    df = pd.DataFrame(data)
    analyze_ab_test(df)
