import numpy as np
import pandas as pd
from scipy import stats

def calculate_optimal_hedge_ratio(hedged_returns, unhedged_returns):
    """
    Calculate the optimal hedge ratio between hedged and unhedged positions
    using minimum variance approach.
    
    Parameters:
    -----------
    hedged_returns : array-like
        Returns of the hedged position
    unhedged_returns : array-like
        Returns of the unhedged position
        
    Returns:
    --------
    dict
        Dictionary containing optimal hedge ratio and related statistics
    """
    # Convert inputs to numpy arrays
    hedged = np.array(hedged_returns)
    unhedged = np.array(unhedged_returns)
    
    # Calculate variance and covariance
    var_hedged = np.var(hedged)
    var_unhedged = np.var(unhedged)
    cov_matrix = np.cov(hedged, unhedged)
    covariance = cov_matrix[0, 1]
    
    # Calculate optimal hedge ratio
    optimal_hedge_ratio = covariance / var_hedged
    
    # Calculate portfolio variance with optimal hedge
    portfolio_variance = (
        var_unhedged + 
        (optimal_hedge_ratio ** 2) * var_hedged - 
        2 * optimal_hedge_ratio * covariance
    )
    
    # Calculate correlation coefficient
    correlation = covariance / (np.std(hedged) * np.std(unhedged))
    
    # Calculate different hedge effectiveness measures
    # 1. R-squared measure
    hedge_effectiveness_rsquared = correlation ** 2
    
    # 2. Variance reduction measure
    variance_reduction = (var_unhedged - portfolio_variance) / var_unhedged
    
    # 3. VaR reduction measure (95% confidence)
    confidence_level = 0.95
    z_score = stats.norm.ppf(confidence_level)
    
    var_unhedged_position = -z_score * np.sqrt(var_unhedged)
    var_hedged_portfolio = -z_score * np.sqrt(portfolio_variance)
    var_reduction = (var_unhedged_position - var_hedged_portfolio) / var_unhedged_position
    
    return {
        'optimal_hedge_ratio': optimal_hedge_ratio,
        'portfolio_variance': portfolio_variance,
        'correlation': correlation,
        'hedge_effectiveness': {
            'rsquared': hedge_effectiveness_rsquared,
            'variance_reduction': variance_reduction,
            'var_reduction': var_reduction
        },
        'hedged_variance': var_hedged,
        'unhedged_variance': var_unhedged
    }

def example_usage():
    """
    Example of how to use the optimal hedge ratio calculator
    """
    # Sample data: Daily returns for hedged and unhedged positions
    np.random.seed(42)
    
    # Generate sample returns (this is just for demonstration)
    n_days = 252  # One trading year
    hedged_returns = np.random.normal(0.0001, 0.01, n_days)
    unhedged_returns = hedged_returns + np.random.normal(0, 0.005, n_days)
    
    # Calculate optimal hedge ratio
    results = calculate_optimal_hedge_ratio(hedged_returns, unhedged_returns)
    
    # Print results
    print(f"Optimal Hedge Ratio: {results['optimal_hedge_ratio']:.4f}")
    print(f"Portfolio Variance: {results['portfolio_variance']:.6f}")
    print(f"Correlation: {results['correlation']:.4f}")
    print("\nHedge Effectiveness Measures:")
    print(f"R-squared: {results['hedge_effectiveness']['rsquared']:.4f}")
    print(f"Variance Reduction: {results['hedge_effectiveness']['variance_reduction']:.4f}")
    print(f"VaR Reduction: {results['hedge_effectiveness']['var_reduction']:.4f}")
    
    return results

if __name__ == "__main__":
    example_usage()