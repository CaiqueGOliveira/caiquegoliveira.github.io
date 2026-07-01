"""
Car Repair Shop Location Analysis in Spain
==========================================

Calculates the S index for 50 continental Spanish provinces to determine
the best locations for opening a car repair shop.

The index balances market potential (fleet, income, density) against
operating costs (rent, CPI, unemployment).

Author: Caique Oliveira
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DATA: 50 Continental Spanish Provinces
# ============================================================================

provinces = [
    "Araba/Álava", "Albacete", "Alicante/Alacant", "Almeria", "Avila", "Badajoz",
    "Balears (Illes)", "Barcelona", "Burgos", "Caceres", "Cadiz", "Castellon/Castello",
    "Ciudad Real", "Cordoba", "Coruña (A)", "Cuenca", "Girona", "Granada",
    "Guadalajara", "Gipuzkoa", "Huelva", "Huesca", "Jaen", "Leon", "Lleida",
    "Rioja (La)", "Lugo", "Madrid", "Malaga", "Murcia", "Navarra", "Ourense",
    "Asturias", "Palencia", "Palmas (Las)", "Pontevedra", "Salamanca",
    "Santa Cruz de Tenerife", "Cantabria", "Segovia", "Sevilla", "Soria",
    "Tarragona", "Teruel", "Toledo", "Valencia/València", "Valladolid", "Bizkaia",
    "Zamora", "Zaragoza"
]

# Vehicle fleet (passenger cars)
fleet = np.array([
    170126, 214826, 1129748, 422961, 99630, 418171, 768520, 2452618, 209409, 242376,
    631949, 340459, 277535, 417331, 667450, 126047, 447302, 518212, 170792, 344787,
    288360, 134476, 329191, 283136, 251717, 165966, 220871, 4171515, 957415, 865790,
    362771, 198628, 550797, 98180, 685799, 572787, 188894, 636391, 336037, 101208,
    1022948, 55531, 457395, 84631, 465698, 1352901, 283734, 541360, 109817, 456019
])

# Average net income per person (€/year)
income = np.array([
    17806, 13058, 12313, 11543, 13602, 12068, 16118, 17062, 15988, 12771,
    12335, 14029, 12529, 12386, 15319, 12725, 15130, 12709, 14689, 19616,
    12008, 15140, 11847, 15110, 14713, 14827, 14316, 18142, 12950, 12446,
    16423, 14049, 15784, 15147, 13806, 14137, 14583, 13366, 15043, 14949,
    12964, 15626, 14559, 14573, 12422, 14435, 15590, 18738, 13897, 15650
])

# Population density (inhabitants/km²)
density = np.array([
    112.6, 26.2, 349.5, 87.8, 20.0, 30.6, 250.4, 770.6, 25.4, 19.5,
    169.5, 94.6, 25.0, 56.1, 142.8, 11.7, 140.6, 74.8, 23.4, 370.1,
    53.2, 14.7, 45.8, 28.8, 37.7, 64.8, 33.1, 886.2, 245.1, 140.2,
    65.8, 42.0, 95.7, 19.7, 287.9, 210.8, 26.6, 322.1, 111.4, 22.9,
    140.9, 8.7, 138.8, 9.2, 49.1, 255.6, 65.2, 526.8, 15.7, 57.8
])

# Average residential rent price (€/m²/month)
rent = np.array([
    12.1, 8.0, 12.0, 8.5, 7.8, 7.1, 19.1, 20.4, 9.4, 7.4,
    10.4, 8.8, 7.1, 8.6, 9.5, 7.5, 12.6, 9.9, 10.0, 16.4,
    9.0, 10.1, 6.3, 7.8, 11.5, 9.1, 7.5, 20.8, 16.6, 9.0,
    11.1, 7.5, 10.2, 7.8, 15.4, 10.4, 9.5, 15.1, 10.9, 12.4,
    11.7, 8.5, 10.1, 7.4, 8.7, 13.6, 9.2, 14.7, 7.1, 10.7
])

# Consumer Price Index (CPI)
cpi = np.array([
    120.220, 121.476, 120.291, 120.524, 120.713, 120.159, 120.332, 118.829, 120.713, 120.159,
    120.524, 120.291, 121.476, 120.524, 120.469, 121.476, 118.829, 120.524, 121.476, 120.220,
    120.524, 120.016, 120.524, 120.713, 118.829, 120.117, 120.469, 119.383, 120.524, 119.672,
    120.082, 120.469, 119.791, 120.713, 119.798, 120.469, 120.713, 119.798, 119.991, 120.713,
    120.524, 120.713, 118.829, 120.016, 121.476, 120.291, 120.713, 120.220, 120.713, 120.016
])

# Unemployment rate (%)
unemployment = np.array([
    5.03, 11.96, 13.28, 14.46, 11.30, 17.59, 11.38, 8.44, 8.41, 13.96,
    20.82, 13.97, 16.27, 15.43, 8.44, 11.30, 9.28, 21.34, 9.31, 6.35,
    20.57, 10.85, 18.95, 10.34, 5.42, 9.52, 7.35, 9.72, 16.49, 11.32,
    9.32, 9.77, 11.81, 6.92, 15.48, 10.41, 10.56, 16.94, 7.48, 8.01,
    15.69, 5.36, 14.45, 9.46, 11.84, 11.93, 8.62, 6.69, 15.41, 6.96
])

# ============================================================================
# CALCULATION: S Index
# ============================================================================

# Convert unemployment to decimal rate
U = unemployment / 100.0

# Calculate the S index
# S = sqrt(Fleet) * Income * Density * (1 - U) / (Rent * CPI)
numerator = np.sqrt(fleet) * income * density * (1 - U)
denominator = rent * cpi
S = numerator / denominator

# Normalize by maximum (Madrid = 1.000)
S_norm = S / np.max(S)

# ============================================================================
# RESULTS: Top 15 Ranking
# ============================================================================

# Get ranking (descending order)
ranking = np.argsort(S_norm)[::-1]

print("=" * 70)
print("TOP 15 PROVINCES FOR CAR REPAIR SHOP (S Index)")
print("=" * 70)
for i, idx in enumerate(ranking[:15], 1):
    print(f"{i:2d}. {provinces[idx]:25s} S_norm={S_norm[idx]:.3f}")

# ============================================================================
# VISUALIZATION: Top 15 Bar Chart
# ============================================================================

# Extract top 15
top_n = 15
idx_top15 = ranking[:top_n]
provinces_top15 = [provinces[i] for i in idx_top15]
S_top15 = S_norm[idx_top15]

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(range(len(provinces_top15)), S_top15[::-1], color='#2196F3', edgecolor='white')

# Configure axes
ax.set_yticks(range(len(provinces_top15)))
ax.set_yticklabels(provinces_top15[::-1], fontsize=9)
ax.set_xlabel('S normalized (max = 1.00)', fontsize=10)
ax.set_title('Top 15 Provinces by the S Index', fontsize=12, fontweight='bold')

# Add value labels
for i, val in enumerate(S_top15[::-1]):
    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

# Set limits
ax.set_xlim(0, 1.1)

# Save and show
plt.tight_layout()
plt.savefig('top15_provinces.png', dpi=300, bbox_inches='tight')
print("\nChart saved as 'top15_provinces.png'")
plt.show()