import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
plt.style.use ("seaborn-darkgrid")

def resample(frequency, gamma, n_samples=18, alpha=1):
    # Prefer a non linear binning to weight low freqs more
    rescaled_freq = np.linspace (0, 1, n_samples)
    rescaled_freq = (frequency.max () - frequency.min ()) * rescaled_freq ** alpha + frequency.min ()
    resampled_freq = np.zeros (rescaled_freq.shape[0] - 1)
    resampled_gamma = np.zeros (rescaled_freq.shape[0] - 1)
    prev_freq = rescaled_freq[0]
    mean = 0
    for j in range (0, rescaled_freq.shape[0] - 1):
        next_freq = rescaled_freq[j + 1]
        resampled_freq[j] = (next_freq + prev_freq) / 2
        mean = np.mean (gamma[(frequency > prev_freq) & (frequency < next_freq)])
        resampled_gamma[j] = mean
        prev_freq = next_freq
    # Add (0, 0)
    new_freq = np.append(0, resampled_freq)
    new_gamma = np.append(0, resampled_gamma)
    return new_freq.reshape (-1, 1), new_gamma.reshape (-1, 1)


def spline_with_zero(freqs, gamma):
    x = np.array(freqs)
    y = np.array(gamma)
    x[0] = y[0] = 0
    spl = scipy.interpolate.splrep(x, y)
    return spl

train_folder = 'train'
interpolate_folder = 'interpolate'

plt.title('Bandwidth')
c = '0'
for i in range(len(samples)):
    # Training Frequencies/Gammas
    tr_freqs = np.load('./train/freqs/aSiGe_C'+c+'.npy')
    tr_gams = np.load('./train/gammas/aSiGe_C'+c+'_300.npy')
    inp_freqs = np.load('./interpolate/freqs/aSiGe_C'+c+'.npy')
    inp_gams = np.load('./interpolate/gammas/aSiGe_C'+c+'_300.npy')
    re_freqs, re_gams = resample(tr_freqs, tr_gams)
    spline = spline_with_zero(re_freqs, re_gams)
    plt.scatter(tr_freqs, tr_gams, label='Training Set')
    plt.scatter(inp_freqs, inp_gams, label='Input Set')
    plt.scatter(inp_freqs, spline, label='Interpolated')


plt.legend()
plt.grid()
plt.xlabel('freq(THz)', fontsize=16)
plt.ylabel('$\Gamma$(meV)', fontsize=16)
plt.savefig('testing.png')
