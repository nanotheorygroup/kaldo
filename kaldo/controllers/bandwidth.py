"""Explicit QHGK bandwidth providers; outputs always use rad/ps."""

from __future__ import annotations

import numpy as np

from kaldo.helpers.atomic_io import sha256_arrays


def _convert(values, unit):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("linewidth/lifetime values must be finite")
    if unit == "rad/ps":
        result = values
        convention = "direct diffusivity_bandwidth"
    elif unit == "THz":
        result = values * (2 * np.pi)
        convention = "cycle frequency to angular diffusivity_bandwidth"
    elif unit == "lifetime_ps":
        if np.any(values <= 0):
            raise ValueError("lifetimes must be positive")
        result = 1 / (2 * values)
        convention = "diffusivity_bandwidth = 1/(2*lifetime_ps)"
    else:
        raise ValueError("unit must be rad/ps, THz, or lifetime_ps")
    if np.any(result < 0):
        raise ValueError("bandwidth must be nonnegative")
    return result, convention


class DirectBandwidth:
    """Scalar or per-mode values, including optional lifetime conversion."""

    def __init__(self, values, *, unit="rad/ps"):
        self.values = np.asarray(values, dtype=np.float64)
        self.unit = unit
        _, convention = _convert(self.values, unit)
        self.metadata = {"source_unit": unit, "conversion": convention,
                         "interpolation": "none", "source_sha256": sha256_arrays(self.values)}

    def evaluate(self, omega, temperature):
        omega = np.asarray(omega)
        if omega.ndim != 1 or not np.isfinite(omega).all():
            raise ValueError("omega must be a finite vector")
        if self.values.ndim == 0:
            values = np.full(len(omega), self.values)
        elif self.values.size == len(omega):
            values = self.values.reshape(-1)
        else:
            raise ValueError("direct bandwidth must be scalar or one value per mode")
        return np.ascontiguousarray(_convert(values, self.unit)[0])


class TabulatedBandwidth:
    """Bilinear frequency/temperature interpolation with explicit bounds.

    ``values`` has shape ``(n_temperatures, n_frequencies)``. Out-of-range
    frequencies and temperatures raise by default; ``bounds='clamp'`` uses
    the nearest edge value.
    """

    def __init__(self, frequencies, temperatures, values, *,
                 frequency_unit="rad/ps", value_unit="rad/ps", bounds="error"):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        if (self.frequencies.ndim != 1 or self.temperatures.ndim != 1 or
                not len(self.frequencies) or not len(self.temperatures) or
                self.values.shape != (len(self.temperatures), len(self.frequencies)) or
                not np.isfinite(self.frequencies).all() or
                not np.isfinite(self.temperatures).all() or
                np.any(np.diff(self.frequencies) <= 0) or
                np.any(np.diff(self.temperatures) <= 0) or
                np.any(self.temperatures <= 0)):
            raise ValueError("invalid frequency/temperature table")
        if frequency_unit == "THz":
            self.frequencies = self.frequencies * (2 * np.pi)
        elif frequency_unit != "rad/ps":
            raise ValueError("frequency_unit must be rad/ps or THz")
        if bounds not in ("error", "clamp"):
            raise ValueError("bounds must be error or clamp")
        self.value_unit = value_unit
        self.bounds = bounds
        self._converted, convention = _convert(self.values, value_unit)
        self.metadata = {"source_unit": value_unit,
                         "frequency_unit": frequency_unit,
                         "temperature_unit": "K", "conversion": convention,
                         "interpolation": "linear frequency then linear temperature",
                         "bounds": bounds,
                         "source_sha256": sha256_arrays(self.frequencies, self.temperatures,
                                                        self.values)}

    def evaluate(self, omega, temperature):
        omega = np.asarray(omega, dtype=np.float64)
        if (omega.ndim != 1 or not np.isfinite(omega).all() or
                not np.isfinite(temperature)):
            raise ValueError("omega and temperature must be finite")
        if self.bounds == "error" and (np.any(omega < self.frequencies[0]) or
                                        np.any(omega > self.frequencies[-1]) or
                                        not self.temperatures[0] <= temperature <= self.temperatures[-1]):
            raise ValueError("bandwidth interpolation outside table")
        # Interpolation is linear in the table values, so blending the two
        # bracketing temperature rows first equals interpolating every
        # frequency row and then interpolating each mode in temperature.
        if len(self.temperatures) == 1:
            row = self._converted[0]
        else:
            upper = int(np.clip(np.searchsorted(self.temperatures, temperature, side="right"),
                                1, len(self.temperatures) - 1))
            lower = upper - 1
            span = self.temperatures[upper] - self.temperatures[lower]
            weight = (temperature - self.temperatures[lower]) / span
            weight = float(np.clip(weight, 0., 1.))
            row = (1 - weight) * self._converted[lower] + weight * self._converted[upper]
        return np.ascontiguousarray(np.interp(omega, self.frequencies, row))
