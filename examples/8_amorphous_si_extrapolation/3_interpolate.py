from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
plt.style.use ("seaborn-darkgrid")
import pandas as pd
import numpy as np
import scipy

desired_concentrations = [0.1] # concentration as decimal
train_size = '1728' # atoms
interpolate_size = '13284' # atoms

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

def interpolater(training, interpolating, plot=True)
    training_frequencies = np.load(training+'/frequency.npy')
    training_gammas = np.load(training+'/bandwidth.npy')

    # Note: resampling is not necessary, but because we care most about the behavior at low frequency,
    # increasing weights in this region can improve results.
    resample_frequencies, resample_gammas = resample(training_frequencies, training_gammas)

    interpolation_frequencies = np.load(interpolating+'/frequency.npy')
    spline = spline_with_zero(resample_frequencies, resample_gammas)
    interpolation_gammas = scipy.interpolate.splev(interpolation_frequencies, spline)

    if plot:
        plt.scatter(training_frequencies, training_gammas, label='Training Set')
        plt.scatter(interpolation_frequencies, interpolation_gammas, label='Interpolated')
        plt.xlabel('freq(THz)', fontsize=16); plt.ylabel('$\Gamma$(meV)', fontsize=16)
        plt.legend(); plt.grid(); plt.title('Bandwidth')
        plt.savefig('testing.png')

for c in desired_concentrations:
    training_folder = 'structures/'+train_size+'_atom/aSiGe_C'+str(int(c*100))
    interpolating_folder = 'structures/'+interpolate_size+'_atom/aSiGe_C'+str(int(c*100))
    interpolater(c, training_folder, interpolating_folder)
    print(str(int(c*100))+' concentration sample interpolated')
