# kALDo's internal plotting routines don't currently support plotting more than one system
# on a single plot. Until this feature is available, you may use this script to compare the
# corrected and uncorrected MgO calculations.
from matplotlib import pyplot as plt
import numpy as np

def calculate_kappa_cumulative_freq(frequencies, kappa_folder):
    """
    Given an unsorted frequency array, return the sorted frequencies and cumulative kappa value by frequency
    Args
    ----
    frequencies = 1D array of frequencies
    """
    sort_args = np.argsort(frequencies.flatten())
    sorted_freq = frequencies.flatten()[sort_args]

    kappa_x = np.loadtxt(kappa_folder+"conductivity_0_0.dat")
    kappa_y = np.loadtxt(kappa_folder+"conductivity_1_1.dat")
    kappa_z = np.loadtxt(kappa_folder+"conductivity_2_2.dat")
    kappa = np.vstack([kappa_x, kappa_y, kappa_z]).mean(axis=0)
    kappa_cum = np.cumsum(kappa[sort_args])
    return sorted_freq, kappa_cum

def prettify_plot(ax):
    # Adjust plot ticks and outlines to make plot pretty
    panel_color_str = "k"
    line_width=2
    tl=4
    tw=2
    tlm=2
    for t in ax.xaxis.get_ticklines(): t.set_color(panel_color_str)
    for t in ax.yaxis.get_ticklines(): t.set_color(panel_color_str)
    for t in ax.xaxis.get_ticklines(): t.set_linewidth(line_width)
    for t in ax.yaxis.get_ticklines(): t.set_linewidth(line_width)
    ax.tick_params(which="major", length=tl, width=tw, labelsize=16)
    ax.tick_params(which="minor", length=tlm, width=tw, labelsize=16)
    ax.tick_params(which="both", axis="both", direction="in",
                   right=True, top=True, labelsize=14)
    ax.spines["bottom"].set_color(panel_color_str)
    ax.spines["top"].set_color(panel_color_str)
    ax.spines["left"].set_color(panel_color_str)
    ax.spines["right"].set_color(panel_color_str)
    ax.spines["bottom"].set_linewidth(line_width)
    ax.spines["top"].set_linewidth(line_width)
    ax.spines["left"].set_linewidth(line_width)
    ax.spines["right"].set_linewidth(line_width)
    ax.grid(False)


fig, ax0 = plt.subplots(ncols=1, figsize=(10, 6), gridspec_kw={"width_ratios": [1,]})

# Plot dispersion w/ NAC
frequencies_nac = np.loadtxt("forces/dispersion")
q_pts = np.loadtxt("forces/q")
ax0.plot(q_pts, frequencies_nac, color="r", lw=1.5, ms=1, label="NAC")

# Plot dispersion w/o charges
frequencies = np.loadtxt("forces_no_charges/dispersion")
ax0.plot(q_pts, frequencies, color="b", lw=1.5, ms=1, label="No Charges")

# Mark high symmetry points
symmetry_points = np.loadtxt("forces/Q_val")
for i in range(symmetry_points.size):
    ax0.axvline(symmetry_points[i], ymin=0, ymax=2, ls="--",  lw=2, c="k")

# Add labels
symmetry_names = np.loadtxt("forces/point_names", dtype="<U12")
symmetry_names[symmetry_names == "G"] = r"$\Gamma$"
ax0.set_xticks(symmetry_points, symmetry_names)

print(q_pts.max(), q_pts.min())
print(symmetry_points)

# Save Dispersion
prettify_plot(ax0)
ax0.set_ylabel("Frequency (THz)", fontsize=16)
ax0.set_xlabel(r"Wave vector ($\frac{2\pi}{a}$)", fontsize=16)
ax0.set_ylim([0,22])
ax0.set_yticks([0,5,10,15,20])
ax0.set_xlim([0,1])
plt.tight_layout(pad=1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("plots/dispersion_comparison.png")


# New figure for phonon properties
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

# Set loading path
path = "forces_no_charges/9_9_9/"
path_nac = "forces/9_9_9/"
temp_stats = "300/quantum/"

# Load phonon properties
frequencies = np.load(path+"frequency.npy")
frequencies_nac = np.load(path_nac+"frequency.npy")

velocities = np.linalg.norm(np.load(path+"velocity.npy"), axis=-1)
velocities_nac = np.linalg.norm(np.load(path_nac+"velocity.npy"), axis=-1)

bandwidth = np.load(path+temp_stats+"bandwidth.npy")
bandwidth_nac = np.load(path_nac+temp_stats+"bandwidth.npy")

lifetimes = np.reciprocal(bandwidth)
lifetimes_nac = np.reciprocal(bandwidth_nac)
lifetimes[bandwidth == 0.] = 0 # Remove "infinite" lifetimes
lifetimes_nac[bandwidth_nac == 0.] = 0

# use helper function to generate cumulative kappa by frequency
sort_freqs, kappa_c = calculate_kappa_cumulative_freq(frequencies, path+"inverse/300/quantum/")
sort_freqs_nac, kappa_c_nac = calculate_kappa_cumulative_freq(frequencies_nac, path_nac+"inverse/300/quantum/")

# plot velocities
ax0.set_ylabel(r"$v_{g}$ (km/s)",  color="k", fontsize=17)
ax0.scatter(frequencies, velocities/10,  edgecolor="b", facecolor="w", s=10)
ax0.scatter(frequencies_nac, velocities_nac/10, edgecolor="r", facecolor="w", s=10)
ax0.set_xlim([0, 22])

# plot lifetimes
ax1.set_ylabel(r"$\tau \ (ps)$",  color="k", fontsize=17)
ax1.set_yscale("log")
ax1.scatter(frequencies, lifetimes,  edgecolor="b", facecolor="w", s=10)
ax1.scatter(frequencies_nac, lifetimes_nac, edgecolor="r", facecolor="w", s=10)
ax1.set_xlim([0, 22])

#plot cumulative kappa
ax2.set_ylabel(r"$\kappa_{cum}(\omega)$ \ (W/m/K)",  color="k", fontsize=17)
ax2.plot(sort_freqs, kappa_c, color="b", label="No Charge")
ax2.plot(sort_freqs_nac, kappa_c_nac, color="r", label="w/ NAC")
ax2.set_xlim([0, 22])

# Save phonon property figure
prettify_plot(ax0)
prettify_plot(ax1)
prettify_plot(ax2)
legend = ax2.legend(loc="lower right")
fig.align_xlabels()
fig.align_ylabels()
plt.subplots_adjust(wspace=0.33)
plt.tight_layout(pad=1)
plt.savefig("plots/phonon_properties.png")
