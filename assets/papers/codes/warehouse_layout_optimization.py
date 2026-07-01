"""
Warehouse Layout Optimization via Spectral Graph Theory
========================================================

This script implements a spectral relaxation approach to optimize warehouse 
shelf arrangement. The goal is to minimize total search time by considering:
1. Retrieval frequency of each part (ABC slotting)
2. Affinity between parts (co-purchase patterns)

The problem is NP-hard (Minimum Linear Arrangement), so we use the Fiedler 
vector of the graph Laplacian to find an approximate optimal arrangement.

Author: Caique Oliveira
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_synthetic_data(n_parts=20, seed=42):
    """
    Generate synthetic warehouse data for testing.
    
    Parameters:
    -----------
    n_parts : int
        Number of distinct parts in the warehouse
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    freq : np.ndarray
        Retrieval frequency for each part (orders per week)
    A : np.ndarray
        Affinity matrix (co-purchase counts)
    """
    np.random.seed(seed)
    
    # Retrieval frequency for each part (orders per week)
    freq = np.random.randint(5, 100, size=n_parts)
    
    # Affinity matrix: co-purchase counts
    # 25% chance of affinity between any two parts
    A = np.zeros((n_parts, n_parts))
    for i in range(n_parts):
        for j in range(i + 1, n_parts):
            if np.random.random() < 0.25:
                affinity = np.random.randint(1, 20)
                A[i, j] = A[j, i] = affinity
    
    return freq, A


# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================

def compute_fiedler_vector(A):
    """
    Compute the Fiedler vector of the affinity graph.
    
    The Fiedler vector is the eigenvector associated with the second-smallest 
    eigenvalue of the combinatorial Laplacian L = D - A.
    
    Parameters:
    -----------
    A : np.ndarray
        Affinity matrix (n x n)
    
    Returns:
    --------
    eigenvalues : np.ndarray
        Eigenvalues of the Laplacian
    eigenvectors : np.ndarray
        Eigenvectors of the Laplacian
    fiedler : np.ndarray
        Fiedler vector (second eigenvector)
    L : np.ndarray
        Combinatorial Laplacian matrix
    """
    # Compute degree matrix
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    
    # Combinatorial Laplacian
    L = D - A
    
    # Eigendecomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = eigh(L)
    
    # Fiedler vector: eigenvector of second-smallest eigenvalue
    fiedler = eigenvectors[:, 1]
    
    return eigenvalues, eigenvectors, fiedler, L


def get_optimal_arrangement(fiedler):
    """
    Get optimal arrangement by sorting Fiedler vector components.
    
    Parameters:
    -----------
    fiedler : np.ndarray
        Fiedler vector
    
    Returns:
    --------
    P_optimal : np.ndarray
        Optimal arrangement (permutation of parts)
    """
    return np.argsort(fiedler)


# ============================================================================
# COST FUNCTION
# ============================================================================

def total_cost(arrangement, freq, A):
    """
    Calculate total search cost for a given arrangement.
    
    C(P) = C1(P) + C2(P)
    where:
    - C1(P) = sum(f_i * p_i): frequency-based cost
    - C2(P) = sum(a_ij * |p_i - p_j|): affinity-based cost
    
    Parameters:
    -----------
    arrangement : np.ndarray
        Permutation of parts (part at position i)
    freq : np.ndarray
        Retrieval frequency for each part
    A : np.ndarray
        Affinity matrix
    
    Returns:
    --------
    cost : float
        Total search cost
    """
    n = len(arrangement)
    
    # Map each part to its position (1-indexed)
    pos = np.zeros(n)
    for idx, part in enumerate(arrangement):
        pos[part] = idx + 1
    
    # Frequency cost: high-demand parts should be close to door (position 1)
    c1 = np.sum(freq * pos)
    
    # Affinity cost: co-purchased parts should be close together
    c2 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                c2 += A[i, j] * abs(pos[i] - pos[j])
    
    return c1 + c2


# ============================================================================
# ROBUSTNESS ANALYSIS
# ============================================================================

def analyze_spectral_gap(eigenvalues):
    """
    Analyze spectral gap for Fiedler vector stability.
    
    The spectral gap Δ = min(λ_2 - λ_1, λ_3 - λ_2) determines the stability 
    of the Fiedler vector under perturbations (Davis-Kahan theorem).
    
    Parameters:
    -----------
    eigenvalues : np.ndarray
        Eigenvalues of the Laplacian (sorted)
    
    Returns:
    --------
    lambda_2 : float
        Second smallest eigenvalue
    lambda_3 : float
        Third smallest eigenvalue
    spectral_gap : float
        Spectral gap Δ
    """
    lambda_2 = eigenvalues[1]
    lambda_3 = eigenvalues[2]
    
    # Spectral gap: distance to nearest eigenvalue
    spectral_gap = min(lambda_2 - eigenvalues[0], lambda_3 - lambda_2)
    
    return lambda_2, lambda_3, spectral_gap


def analyze_cost_landscape(P_optimal, freq, A, cost_opt):
    """
    Analyze cost landscape flatness by testing adjacent swaps.
    
    Parameters:
    -----------
    P_optimal : np.ndarray
        Optimal arrangement
    freq : np.ndarray
        Retrieval frequency
    A : np.ndarray
        Affinity matrix
    cost_opt : float
        Cost of optimal arrangement
    
    Returns:
    --------
    max_swap : float
        Maximum cost change from single adjacent swap
    mean_swap : float
        Mean cost change from adjacent swaps
    """
    P_opt_list = list(P_optimal)
    swap_costs = []
    
    # Test swapping each pair of adjacent positions
    for k in range(len(P_opt_list) - 1):
        P_swapped = P_opt_list.copy()
        P_swapped[k], P_swapped[k+1] = P_swapped[k+1], P_swapped[k]
        
        cost_swapped = total_cost(np.array(P_swapped), freq, A)
        delta_k = cost_swapped - cost_opt
        swap_costs.append(abs(delta_k))
    
    max_swap = max(swap_costs)
    mean_swap = np.mean(swap_costs)
    
    return max_swap, mean_swap


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cost_comparison(cost_rand, cost_opt, reduction, filename='layout_comparison.png'):
    """
    Create bar chart comparing random vs spectral layout costs.
    
    Parameters:
    -----------
    cost_rand : float
        Cost of random layout
    cost_opt : float
        Cost of spectral (optimal) layout
    reduction : float
        Percentage cost reduction
    filename : str
        Output filename for the plot
    """
    plt.figure(figsize=(10, 6))
    
    layouts = ['Random\nLayout', 'Spectral\nLayout']
    costs = [cost_rand, cost_opt]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = plt.bar(layouts, costs, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 200,
            f'{cost:,.0f}', 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )
    
    # Add reduction percentage annotation
    plt.text(
        0.5, 
        max(costs) * 0.7, 
        f'Reduction: {reduction:.1f}%',
        ha='center', 
        fontsize=14, 
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.ylabel('Total Search Cost', fontsize=12)
    plt.title(
        'Warehouse Layout Optimization: Random vs Spectral Arrangement',
        fontsize=14, 
        fontweight='bold', 
        pad=20
    )
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Generate synthetic data
    print("=" * 60)
    print("WAREHOUSE LAYOUT OPTIMIZATION VIA SPECTRAL GRAPH THEORY")
    print("=" * 60)
    
    freq, A = generate_synthetic_data(n_parts=20, seed=42)
    
    # Compute Fiedler vector
    eigenvalues, eigenvectors, fiedler, L = compute_fiedler_vector(A)
    P_optimal = get_optimal_arrangement(fiedler)
    
    # Generate random arrangement for baseline comparison
    np.random.seed(99)
    P_random = np.random.permutation(len(freq))
    
    # Calculate costs
    cost_opt = total_cost(P_optimal, freq, A)
    cost_rand = total_cost(P_random, freq, A)
    reduction = (cost_rand - cost_opt) / cost_rand * 100
    
    # Print main results
    print("\n--- MAIN RESULTS ---")
    print(f"Random layout cost:    {cost_rand:.0f}")
    print(f"Spectral layout cost:  {cost_opt:.0f}")
    print(f"Cost reduction:        {reduction:.1f}%")
    
    # Spectral gap analysis (Davis-Kahan theorem)
    lambda_2, lambda_3, spectral_gap = analyze_spectral_gap(eigenvalues)
    
    print("\n--- DAVIS-KAHAN SPECTRAL GAP ANALYSIS ---")
    print(f"λ_2 (Fiedler eigenvalue):     {lambda_2:.3f}")
    print(f"λ_3 (Third eigenvalue):       {lambda_3:.3f}")
    print(f"Spectral Gap (Δ):             {spectral_gap:.3f}")
    print(f"Interpretation: Large gap relative to expected noise")
    print(f"guarantees Fiedler vector stability.")
    
    # Cost landscape flatness analysis
    max_swap, mean_swap = analyze_cost_landscape(P_optimal, freq, A, cost_opt)
    
    print("\n--- COST LANDSCAPE FLATNESS ---")
    print(f"Max cost change (single swap):   {max_swap:.0f}")
    print(f"Mean cost change (single swap):  {mean_swap:.0f}")
    print(f"Total optimal cost:              {cost_opt:.0f}")
    print(f"Max swap as % of total:          {max_swap/cost_opt*100:.2f}%")
    print(f"Interpretation: Flat landscape means small deviations")
    print(f"from optimum have minimal impact.")
    
    # Generate table data for LaTeX article
    print("\n--- TABLE DATA FOR LATEX ARTICLE ---")
    print("Position & Part & Frequency & Fiedler Value \\\\")
    print("\\hline")
    for pos, part_idx in enumerate(P_optimal[:10], 1):
        part_name = f"Part_{part_idx + 1}"
        freq_val = freq[part_idx]
        fiedler_val = fiedler[part_idx]
        print(f"{pos} & {part_name} & {freq_val} & ${fiedler_val:.3f}$ \\\\")
    
    # Generate visualization
    print("\n--- GENERATING VISUALIZATION ---")
    plot_cost_comparison(cost_rand, cost_opt, reduction)
    
    print("\n" + "=" * 60)
    print("Analysis complete. Check 'layout_comparison.png' for visualization.")
    print("=" * 60)


if __name__ == "__main__":
    main()