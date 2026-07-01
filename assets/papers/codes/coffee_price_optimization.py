"""
Coffee Shop Price Optimization via Linear Regression
=====================================================

This script implements a linear regression model to find the optimal
coffee price that maximizes revenue, based on observed price-demand data.

The model fits a linear demand function Q(p) = ap + b, then derives the
quadratic revenue function R(p) = p·Q(p) = ap² + bp, which is optimized
analytically to find the price that maximizes revenue.

Author: Caique Oliveira
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DATA
# ============================================================================

# Observed data: price (USD) and units sold
price_data = np.array([2.00, 3.00, 4.00, 5.00, 6.00, 
                       2.50, 3.50, 4.50, 5.50, 6.50])
units_data = np.array([100, 85, 60, 45, 30, 
                       95, 70, 50, 35, 20])

# ============================================================================
# LINEAR REGRESSION: Least Squares
# ============================================================================

def fit_linear_regression(X, y):
    """
    Fit a linear regression model using ordinary least squares.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    
    Returns:
    --------
    coefficients : np.ndarray
        Fitted coefficients
    """
    # Add intercept column
    A = np.column_stack([X, np.ones_like(X)])
    
    # Solve normal equations: A^T A β = A^T y
    coefficients = np.linalg.solve(A.T @ A, A.T @ y)
    
    return coefficients

# Fit the demand model: units = a·price + b
coefficients = fit_linear_regression(price_data, units_data)
a, b = coefficients[0], coefficients[1]

print("=" * 60)
print("DEMAND MODEL: LINEAR REGRESSION")
print("=" * 60)
print(f"Coefficients: a = {a:.2f}, b = {b:.2f}")
print(f"Demand function: Q(p) = {a:.2f}·p + {b:.2f}")

# ============================================================================
# MODEL EVALUATION: RMSE
# ============================================================================

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    rmse : float
        Root Mean Squared Error
    """
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    return rmse

# Predicted demand
A_full = np.column_stack([price_data, np.ones_like(price_data)])
units_pred = A_full @ coefficients

# Calculate RMSE
rmse = calculate_rmse(units_data, units_pred)

print(f"\nModel Evaluation:")
print(f"RMSE = {rmse:.2f} units")
print(f"Interpretation: Predictions deviate by ~{rmse:.0f} units on average")

# ============================================================================
# REVENUE OPTIMIZATION
# ============================================================================

def revenue(price, a, b):
    """
    Calculate revenue given price and demand function coefficients.
    
    R(p) = p · Q(p) = p · (a·p + b) = a·p² + b·p
    
    Parameters:
    -----------
    price : float or np.ndarray
        Price(s) to evaluate
    a : float
        Slope coefficient of demand function
    b : float
        Intercept of demand function
    
    Returns:
    --------
    revenue : float or np.ndarray
        Revenue at given price(s)
    """
    return price * (a * price + b)

def optimal_price(a, b):
    """
    Calculate the price that maximizes revenue.
    
    R(p) = a·p² + b·p
    R'(p) = 2a·p + b = 0
    p* = -b / (2a)
    
    Parameters:
    -----------
    a : float
        Slope coefficient (must be negative for concave revenue)
    b : float
        Intercept coefficient
    
    Returns:
    --------
    p_opt : float
        Optimal price
    """
    return -b / (2 * a)

# Calculate optimal price
p_opt = optimal_price(a, b)
r_opt = revenue(p_opt, a, b)

print("\n" + "=" * 60)
print("REVENUE OPTIMIZATION")
print("=" * 60)
print(f"Optimal price: ${p_opt:.2f}")
print(f"Maximum revenue: ${r_opt:.2f}")

# ============================================================================
# INDIFFERENCE ZONE ANALYSIS
# ============================================================================

def analyze_indifference_zone(p_opt, a, b, tolerance=0.01):
    """
    Analyze the price range around the optimum where revenue is near maximum.
    
    Parameters:
    -----------
    p_opt : float
        Optimal price
    a : float
        Slope coefficient
    b : float
        Intercept coefficient
    tolerance : float
        Fraction of revenue reduction considered acceptable (default 1%)
    
    Returns:
    --------
    p_lower : float
        Lower bound of indifference zone
    p_upper : float
        Upper bound of indifference zone
    """
    # Test prices around the optimum
    test_prices = np.linspace(p_opt - 0.5, p_opt + 0.5, 21)
    revenues = revenue(test_prices, a, b)
    
    # Find prices where revenue is within tolerance of maximum
    threshold = r_opt * (1 - tolerance)
    valid_prices = test_prices[revenues >= threshold]
    
    return valid_prices[0], valid_prices[-1]

p_lower, p_upper = analyze_indifference_zone(p_opt, a, b, tolerance=0.01)

print("\n" + "=" * 60)
print("INDIFFERENCE ZONE")
print("=" * 60)
print(f"Price range: ${p_lower:.2f} - ${p_upper:.2f}")
print(f"Revenue reduction in this zone: < 1%")

# Test specific prices
test_prices = [3.20, 3.40, 3.60, 3.74, 3.80, 4.00, 4.20]
print("\nRevenue at specific prices:")
for p in test_prices:
    r = revenue(p, a, b)
    diff = r_opt - r
    pct = (diff / r_opt) * 100
    print(f"  p = ${p:.2f} → R = ${r:.2f} (Δ = ${diff:.2f}, {pct:.2f}% below optimum)")

# ============================================================================
# VISUALIZATION: Demand Function
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Price range for plotting
p_range = np.linspace(2.0, 7.0, 100)

# Demand function plot
Q_pred = a * p_range + b
ax1.scatter(price_data, units_data, color='#E74C3C', s=80, 
            label='Observed data', zorder=5, edgecolors='black', linewidth=1.5)
ax1.plot(p_range, Q_pred, color='#3498DB', linewidth=2.5, 
         label=f'Fitted: Q(p) = {a:.2f}p + {b:.2f}')
ax1.set_xlabel('Price ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Units Sold', fontsize=12, fontweight='bold')
ax1.set_title('Demand Function', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Revenue function plot
R_curve = revenue(p_range, a, b)
ax2.plot(p_range, R_curve, color='#27AE60', linewidth=2.5, 
         label='Revenue curve R(p)')
ax2.scatter(p_opt, r_opt, color='#E74C3C', s=150, 
            label=f'Optimum: (${p_opt:.2f}, ${r_opt:.2f})', 
            zorder=5, edgecolors='black', linewidth=1.5)

# Highlight indifference zone
ax2.axvspan(p_lower, p_upper, alpha=0.2, color='#F39C12', 
            label=f'Indifference zone: ${p_lower:.2f}-${p_upper:.2f}')

ax2.set_xlabel('Price ($)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
ax2.set_title('Revenue Optimization', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coffee_price_optimization.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Visualization saved as 'coffee_price_optimization.png'")
print("=" * 60)

# ============================================================================
# UNCERTAINTY ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("UNCERTAINTY ANALYSIS")
print("=" * 60)
print(f"RMSE = {rmse:.2f} units")
print(f"At optimal price p* = ${p_opt:.2f}:")
print(f"  Predicted demand: {a * p_opt + b:.1f} units")
print(f"  Maximum revenue: ${r_opt:.2f}")
print(f"  Uncertainty interval: ${r_opt - p_opt * rmse:.2f} - ${r_opt + p_opt * rmse:.2f}")
print(f"  (assuming demand error ~ RMSE)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)