# Example: aluminum FCC, ASE EMT calculator
# Plots: thermal conductivity comparison between serial and parallel IFC methods
# Prerequisite: run 1_force_constants.py and 2_conductivity.py first

import numpy as np
import matplotlib.pyplot as plt

params = {
    'figure.figsize': (8, 5),
    'legend.fontsize': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
plt.rcParams.update(params)

methods = ['RTA', 'Inverse (Exact)']
kappa_serial   = np.load('kappa_serial.npy')
kappa_parallel = np.load('kappa_parallel.npy')

# --- Bar chart: serial vs parallel per BTE method ---
x = np.arange(len(methods))
width = 0.35
fig, ax = plt.subplots()
bars_s = ax.bar(x - width / 2, kappa_serial,   width, label='Serial',   color='steelblue',  alpha=0.85)
bars_p = ax.bar(x + width / 2, kappa_parallel, width, label='Parallel', color='darkorange', alpha=0.85)

# Annotate with exact values
for bar in list(bars_s) + list(bars_p):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# Reference line for experimental value
ax.axhline(5.8, color='crimson', linestyle='--', linewidth=1.5, label='Ab Initio (LDA)')
ax.axhline(5.6, color='crimson', linestyle='--', linewidth=1.5, label='Ab Initio (GGA)')

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Thermal Conductivity (W/m/K)')
ax.set_title('Al FCC — EMT: Serial vs Parallel Third-Order IFCs\n'
             '300 K, 7×7×7 k-mesh, 3×3×3 supercell')
ax.legend()
ax.set_ylim(0, max(kappa_serial.max(), kappa_parallel.max()) * 1.25)

plt.tight_layout()
plt.savefig('conductivity_comparison.png', dpi=150)
plt.close()
print("Saved conductivity_comparison.png")

# --- Difference plot: |serial - parallel| ---
diffs = np.abs(kappa_serial - kappa_parallel)

fig2, ax2 = plt.subplots()
ax2.bar(methods, diffs, color='gray', alpha=0.8)
ax2.set_ylabel('|kappa_serial - kappa_parallel| (W/m/K)')
ax2.set_title('Difference between Serial and Parallel Results')
for i, d in enumerate(diffs):
    ax2.text(i, d + max(diffs) * 0.02, f'{d:.2e}', ha='center', va='bottom', fontsize=10)

ax2.set_ylim([0, max(diffs)*1.1])
plt.tight_layout()
plt.savefig('conductivity_difference.png', dpi=150)
plt.close()
print("Saved conductivity_difference.png")
