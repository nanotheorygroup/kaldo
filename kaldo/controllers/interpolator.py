"""
Frequency-dependent interpolation utilities for phonon linewidths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from kaldo.helpers.logger import get_logger

logging = get_logger()


class BandwidthInterpolator:
    """Gaussian-smoothed frequency interpolator for anharmonic linewidths.

    Parameters
    ----------
    frequency : array_like
        Source phonon frequencies in THz. The array may have any shape; it is
        flattened for fitting and prediction results preserve target shape.
    bandwidth : array_like
        Source anharmonic linewidths in the same THz convention used by
        :attr:`kaldo.phonons.Phonons.bandwidth`.
    sigma : float
        Gaussian smoothing width in THz.
    low_frequency_cutoff : float, optional
        Frequency in THz below which predictions use the explicit quadratic
        law ``Gamma(omega) = A * omega**2``. ``A`` is chosen from the fitted
        curve at the crossover, so the join is C0 continuous.
    extrapolation : {'error', 'constant', 'spline'}, optional
        Behavior for target frequencies above the source frequency range.
        Frequencies below ``low_frequency_cutoff`` always use the quadratic
        extension.
    max_bandwidth : float, optional
        Maximum source bandwidth in THz retained for fitting. Values above
        this cutoff are treated as unphysical outliers, excluded, logged, and
        recorded in the interpolation metadata.
    folder : str or path-like, optional
        If provided, fit inputs, smoothed data, parameters, and labeled
        predictions are written below this folder.
    n_grid_points : int, optional
        Number of frequency grid points used for the smoothed fit.
    chunk_size : int, optional
        Maximum number of evaluation frequencies handled in one Gaussian
        smoothing chunk. Chunking avoids large dense ``(N_eval, N_modes)``
        arrays for large source models.
    """

    _VALID_EXTRAPOLATION = ("error", "constant", "spline")

    def __init__(
        self,
        frequency,
        bandwidth,
        sigma,
        *,
        low_frequency_cutoff=2.0,
        extrapolation="error",
        max_bandwidth=None,
        folder=None,
        n_grid_points=400,
        chunk_size=512,
    ):
        self.source_frequency = np.asarray(frequency, dtype=float)
        self.source_bandwidth = np.asarray(bandwidth, dtype=float)
        self.source_shape = self.source_frequency.shape
        self.bandwidth_shape = self.source_bandwidth.shape
        self.sigma = float(sigma)
        self.low_frequency_cutoff = float(low_frequency_cutoff)
        self.extrapolation = extrapolation
        self.max_bandwidth = None if max_bandwidth is None else float(max_bandwidth)
        self.folder = None if folder is None else Path(folder)
        self.n_grid_points = int(n_grid_points)
        self.chunk_size = int(chunk_size)
        self.frequency_units = "THz"
        self.bandwidth_units = "THz"
        self.frequency = None
        self.bandwidth = None
        self._last_prediction = None
        self.excluded_source_counts = {}

        self._validate_parameters()
        valid_frequency, valid_bandwidth = self._validated_source_data()
        order = np.argsort(valid_frequency)
        self._fit_frequency = valid_frequency[order]
        self._fit_bandwidth = valid_bandwidth[order]

        self.smoothed_frequency = self._build_smoothing_grid()
        self.smoothed_bandwidth = self._gaussian_smooth(self.smoothed_frequency)
        self._build_interpolator()
        self._save_fit()

    def _validate_parameters(self):
        if self.source_frequency.shape != self.source_bandwidth.shape:
            raise ValueError(
                "frequency and bandwidth must have the same shape; "
                f"got {self.source_frequency.shape} and {self.source_bandwidth.shape}"
            )
        if self.source_frequency.size == 0:
            raise ValueError("frequency and bandwidth must contain at least one value")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be a finite positive value in THz")
        if not np.isfinite(self.low_frequency_cutoff) or self.low_frequency_cutoff <= 0:
            raise ValueError("low_frequency_cutoff must be a finite positive value in THz")
        if self.extrapolation not in self._VALID_EXTRAPOLATION:
            raise ValueError(f"extrapolation must be one of {self._VALID_EXTRAPOLATION}")
        if self.max_bandwidth is not None and (not np.isfinite(self.max_bandwidth) or self.max_bandwidth <= 0):
            raise ValueError("max_bandwidth must be None or a finite positive value in THz")
        if self.n_grid_points < 4:
            raise ValueError("n_grid_points must be at least 4")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

    def _validated_source_data(self):
        frequency = self.source_frequency.reshape(-1)
        bandwidth = self.source_bandwidth.reshape(-1)
        valid = np.ones(frequency.shape, dtype=bool)

        invalid_frequency = ~np.isfinite(frequency)
        invalid_bandwidth = ~np.isfinite(bandwidth)
        nonpositive_frequency = frequency <= 0
        negative_bandwidth = bandwidth < 0
        above_max_bandwidth = (
            np.zeros(bandwidth.shape, dtype=bool)
            if self.max_bandwidth is None
            else bandwidth > self.max_bandwidth
        )

        for key, label, mask in (
            ("non_finite_frequency", "non-finite frequency", invalid_frequency),
            ("non_finite_bandwidth", "non-finite bandwidth", invalid_bandwidth),
            ("nonpositive_frequency", "zero or negative frequency", nonpositive_frequency),
            ("negative_bandwidth", "negative bandwidth", negative_bandwidth),
            ("above_max_bandwidth", "bandwidth above max_bandwidth", above_max_bandwidth),
        ):
            count = int(np.count_nonzero(mask))
            self.excluded_source_counts[key] = count
            if count:
                logging.warning("BandwidthInterpolator excluded %d source modes with %s.", count, label)
                valid &= ~mask

        if np.count_nonzero(valid) < 2:
            raise ValueError("at least two finite positive-frequency modes are required for interpolation")

        valid_frequency = frequency[valid]
        valid_bandwidth = bandwidth[valid]
        if np.all(valid_bandwidth == 0):
            logging.warning("All valid source bandwidths are zero; predictions will be zero.")
        return valid_frequency, valid_bandwidth

    def _build_smoothing_grid(self):
        f_min = float(np.min(self._fit_frequency))
        f_max = float(np.max(self._fit_frequency))
        if f_max <= f_min:
            raise ValueError("source frequencies must span a nonzero range")
        return np.linspace(f_min, f_max, self.n_grid_points)

    def _gaussian_smooth(self, evaluation_frequency):
        evaluation_frequency = np.asarray(evaluation_frequency, dtype=float).reshape(-1)
        smoothed = np.empty(evaluation_frequency.shape, dtype=float)

        sigma2 = self.sigma**2
        for start in range(0, evaluation_frequency.size, self.chunk_size):
            stop = min(start + self.chunk_size, evaluation_frequency.size)
            chunk = evaluation_frequency[start:stop]
            exponent = -0.5 * ((chunk[:, np.newaxis] - self._fit_frequency[np.newaxis, :]) ** 2) / sigma2
            exponent -= np.max(exponent, axis=1)[:, np.newaxis]
            weights = np.exp(exponent)
            denominator = np.sum(weights, axis=1)
            smoothed[start:stop] = weights @ self._fit_bandwidth / denominator
        return smoothed

    def _build_interpolator(self):
        cutoff = self.low_frequency_cutoff
        high_mask = self.smoothed_frequency >= cutoff

        if np.count_nonzero(high_mask) >= 2:
            high_frequency = self.smoothed_frequency[high_mask]
            high_bandwidth = self.smoothed_bandwidth[high_mask]
            if high_frequency[0] > cutoff:
                spline_frequency = np.concatenate(([cutoff], high_frequency))
                spline_bandwidth = np.concatenate((self._gaussian_smooth([cutoff]), high_bandwidth))
            else:
                spline_frequency = high_frequency
                spline_bandwidth = high_bandwidth
        else:
            spline_frequency = self.smoothed_frequency
            spline_bandwidth = self.smoothed_bandwidth

        self._spline = PchipInterpolator(spline_frequency, spline_bandwidth, extrapolate=True)
        cutoff_bandwidth = float(self._spline(cutoff))
        if cutoff_bandwidth < 0 and abs(cutoff_bandwidth) < 1e-14:
            cutoff_bandwidth = 0.0
        if cutoff_bandwidth < 0:
            raise ValueError(
                "The fitted linewidth at low_frequency_cutoff is negative. "
                "Use a larger sigma or different cutoff to avoid unphysical interpolation."
            )
        self._quadratic_coefficient = cutoff_bandwidth / cutoff**2
        self._spline_min_frequency = float(np.min(spline_frequency))
        self._spline_max_frequency = float(np.max(spline_frequency))
        self._source_max_frequency = float(np.max(self._fit_frequency))

    def predict_bandwidth(self, target_frequency, label=None):
        """Predict linewidths at target frequencies.

        Parameters
        ----------
        target_frequency : array_like
            Frequencies in THz. The returned linewidth array has the same shape.
        label : str, optional
            If ``folder`` was provided, store the target frequencies and
            predicted bandwidths under ``folder/predictions/<label>``.

        Returns
        -------
        ndarray
            Interpolated bandwidths in THz, with the same shape as
            ``target_frequency``.
        """
        target = np.asarray(target_frequency, dtype=float)
        if np.any(~np.isfinite(target)):
            raise ValueError("target_frequency contains NaN or infinite values")
        if np.any(target < 0):
            logging.warning(
                "BandwidthInterpolator received negative target frequencies; "
                "setting their predicted bandwidth to zero."
            )

        flat_target = target.reshape(-1)
        flat_bandwidth = np.empty(flat_target.shape, dtype=float)

        negative_mask = flat_target < 0
        flat_bandwidth[negative_mask] = 0.0

        low_mask = (flat_target >= 0) & (flat_target < self.low_frequency_cutoff)
        flat_bandwidth[low_mask] = self._quadratic_coefficient * flat_target[low_mask] ** 2

        spline_mask = ~(negative_mask | low_mask)
        if np.any(spline_mask):
            spline_frequency = flat_target[spline_mask]
            above_source = spline_frequency > self._source_max_frequency
            below_spline = spline_frequency < self._spline_min_frequency

            if self.extrapolation == "error" and (np.any(above_source) or np.any(below_spline)):
                min_target = float(np.min(spline_frequency))
                max_target = float(np.max(spline_frequency))
                raise ValueError(
                    "target_frequency is outside the fitted frequency range "
                    f"[{self._spline_min_frequency}, {self._source_max_frequency}] THz: "
                    f"got [{min_target}, {max_target}] THz"
                )

            if self.extrapolation == "constant":
                spline_frequency = np.clip(spline_frequency, self._spline_min_frequency, self._source_max_frequency)

            predicted = np.asarray(self._spline(spline_frequency), dtype=float)
            small_negative = (predicted < 0) & (predicted > -1e-14)
            if np.any(small_negative):
                predicted[small_negative] = 0.0
            if np.any(predicted < 0):
                raise ValueError(
                    "interpolation produced negative bandwidths; "
                    "increase sigma or use constant extrapolation"
                )
            flat_bandwidth[spline_mask] = predicted

        predicted_bandwidth = flat_bandwidth.reshape(target.shape)
        self.frequency = target.copy()
        self.bandwidth = predicted_bandwidth
        self._last_prediction = (self.frequency, self.bandwidth)
        self._save_prediction(label)
        return predicted_bandwidth

    def predict(self, target_frequency, label=None):
        """Alias for :meth:`predict_bandwidth`."""
        return self.predict_bandwidth(target_frequency, label=label)

    def _save_fit(self):
        if self.folder is None:
            return
        fit_folder = self.folder / "fit"
        fit_folder.mkdir(parents=True, exist_ok=True)
        np.save(fit_folder / "source_frequency.npy", self.source_frequency)
        np.save(fit_folder / "source_bandwidth.npy", self.source_bandwidth)
        np.save(fit_folder / "smoothed_frequency.npy", self.smoothed_frequency)
        np.save(fit_folder / "smoothed_bandwidth.npy", self.smoothed_bandwidth)
        parameters = {
            "sigma": self.sigma,
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "extrapolation": self.extrapolation,
            "max_bandwidth": self.max_bandwidth,
            "excluded_source_counts": self.excluded_source_counts,
            "source_frequency_shape": self.source_shape,
            "source_bandwidth_shape": self.bandwidth_shape,
            "frequency_units": self.frequency_units,
            "bandwidth_units": self.bandwidth_units,
            "n_grid_points": self.n_grid_points,
            "chunk_size": self.chunk_size,
            "interpolator": "scipy.interpolate.PchipInterpolator",
            "low_frequency_model": "Gamma(omega) = A * omega**2 below low_frequency_cutoff",
            "quadratic_coefficient": self._quadratic_coefficient,
        }
        with open(fit_folder / "interpolation_parameters.json", "w") as handle:
            json.dump(parameters, handle, indent=2)

    def _save_prediction(self, label):
        if self.folder is None or label is None:
            return
        label = str(label)
        if os.path.sep in label or (os.path.altsep is not None and os.path.altsep in label):
            raise ValueError("prediction label must not contain path separators")
        prediction_folder = self.folder / "predictions" / label
        prediction_folder.mkdir(parents=True, exist_ok=True)
        np.save(prediction_folder / "frequency.npy", self.frequency)
        np.save(prediction_folder / "bandwidth.npy", self.bandwidth)

    def plot(self, filename="bandwidth_interpolation.png", y_max=None, y_percentile=None):
        """Save a bandwidth-vs-frequency plot of source, fit, and last prediction.

        Parameters
        ----------
        filename : str, optional
            Output filename.
        y_max : float, optional
            Upper y-axis limit for display only. No source or prediction data
            are removed from the interpolation.
        y_percentile : float, optional
            If ``y_max`` is not provided, set the upper y-axis limit from this
            percentile of the finite plotted bandwidth values. This is useful
            for visualizing the fitted curve when a few source linewidths are
            much larger than the rest.
        """
        import matplotlib.pyplot as plt

        output_folder = self.folder if self.folder is not None else Path.cwd()
        output_folder.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.scatter(self._fit_frequency, self._fit_bandwidth, s=8, alpha=0.45, label="source")

        last_prediction = self._last_prediction
        last_frequency = None if self.frequency is None else self.frequency.copy()
        last_bandwidth = None if self.bandwidth is None else self.bandwidth.copy()
        plot_frequency = np.linspace(np.min(self._fit_frequency), np.max(self._fit_frequency), self.n_grid_points)
        plot_bandwidth = self.predict_bandwidth(plot_frequency)
        self._last_prediction = last_prediction
        self.frequency = last_frequency
        self.bandwidth = last_bandwidth
        ax.plot(plot_frequency, plot_bandwidth, label="smoothed interpolation")

        if last_prediction is not None:
            frequency, bandwidth = last_prediction
            ax.scatter(frequency.reshape(-1), bandwidth.reshape(-1), s=8, alpha=0.7, label="prediction")

        if y_max is None and y_percentile is not None:
            plotted_bandwidth = [self._fit_bandwidth, plot_bandwidth]
            if last_prediction is not None:
                plotted_bandwidth.append(last_prediction[1].reshape(-1))
            finite_bandwidth = np.concatenate(plotted_bandwidth)
            finite_bandwidth = finite_bandwidth[np.isfinite(finite_bandwidth)]
            if finite_bandwidth.size:
                y_max = np.percentile(finite_bandwidth, y_percentile)
        if y_max is not None:
            ax.set_ylim(bottom=0, top=y_max)

        ax.axvspan(0, self.low_frequency_cutoff, alpha=0.15, color="gray", label="quadratic region")
        ax.axvline(self.low_frequency_cutoff, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("phonon frequency (THz)")
        ax.set_ylabel("phonon bandwidth (THz)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_folder / filename)
        plt.close(fig)
