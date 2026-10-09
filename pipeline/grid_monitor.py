"""Grid-facing indicators for the console (plan_ui_physics.md).

facility_forcing  what this facility's GPU load does to the grid, from NVML only:
                  electromechanical-band (0.1-2 Hz) modulation of the summed power,
                  steepest 1 s ramp, and how synchronised the GPUs are in that band.
GridSource        grid-side measurements when a facility exposes them. Standard paths:
                  UPS-MIB (RFC 1628) upsInputFrequency 1.3.6.1.2.1.33.1.3.3.1.2 (0.1 Hz),
                  upsInputVoltage 1.3.6.1.2.1.33.1.3.3.1.3 (V); Redfish PowerSupply /
                  PowerSupplyMetrics (input voltage, frequency). This server exposes neither,
                  so only NoGridSource and a CSV replay source are implemented.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BAND_HZ = (0.1, 2.0)
GRID_HZ = 10.0


def _grid_series(rows: dict[int, list[dict]], horizon_s: float) -> pd.DataFrame:
    """Per-GPU power on a common 10 Hz grid (instant power when available)."""
    frames = []
    for gpu, rs in rows.items():
        if not rs:
            continue
        f = pd.DataFrame(rs)[["t_epoch", "power_w", "power_instant_w"]]
        use_inst = f["power_instant_w"].notna().mean() > 0.8
        f["p"] = f["power_instant_w"] if use_inst else f["power_w"]
        f["tick"] = (f["t_epoch"] * GRID_HZ).round().astype("int64")
        frames.append(f.groupby("tick")["p"].mean().rename(gpu))
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide[wide.index >= wide.index.max() - int(horizon_s * GRID_HZ)]
    full = range(int(wide.index.min()), int(wide.index.max()) + 1)
    return wide.reindex(full).interpolate(limit_direction="both")


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(len(x), 1.0 / GRID_HZ)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, len(x))


def facility_forcing(rows: dict[int, list[dict]], *, tdp_w: float = 450.0, horizon_s: float = 120.0) -> dict:
    wide = _grid_series(rows, horizon_s)
    if len(wide) < GRID_HZ * 20:
        return {}
    n = wide.shape[1]
    total = wide.sum(axis=1).to_numpy()
    band = _band(total, *BAND_HZ)
    sec = pd.Series(total).rolling(int(GRID_HZ), min_periods=1).mean().to_numpy()
    ramp = np.abs(np.diff(sec[:: int(GRID_HZ)])) if len(sec) > GRID_HZ else np.array([0.0])
    corr = None
    if n >= 2:
        parts = [_band(wide[c].to_numpy(), *BAND_HZ) for c in wide]
        pairs = [np.corrcoef(parts[i], parts[j])[0, 1] for i in range(n) for j in range(i + 1, n)
                 if parts[i].std() > 0 and parts[j].std() > 0]
        corr = float(np.mean(pairs)) if pairs else None
    spec = np.abs(np.fft.rfft(band))
    freqs = np.fft.rfftfreq(len(band), 1.0 / GRID_HZ)
    return {
        "n_gpus": n,
        "seconds": len(total) / GRID_HZ,
        "mean_w": float(total.mean()),
        "band_amp_pu": float(2 * np.sqrt(2) * band.std() / (n * tdp_w)),  # peak-to-peak of a sinusoid, / TDP
        "band_peak_hz": float(freqs[np.argmax(spec)]) if spec.max() > 0 else None,
        "max_ramp_pu_per_s": float(ramp.max() / (n * tdp_w)),
        "gpu_band_corr": corr,
    }


def amplitude_spectrum(x: np.ndarray, sample_hz: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum (Hann window, mean removed) restricted to ``band``."""
    x = np.asarray(x, float) - np.mean(x)
    w = np.hanning(len(x))
    amp = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_hz)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[keep], amp[keep]


def facility_spectrum(rows: dict[int, list[dict]], *, tdp_w: float = 450.0, horizon_s: float = 120.0,
                      band: tuple[float, float] = (0.05, 2.0)) -> dict:
    """Spectrum of the summed facility power as a fraction of total TDP, for the mode-overlap plot."""
    wide = _grid_series(rows, horizon_s)
    if len(wide) < GRID_HZ * 20:
        return {}
    freqs, amp = amplitude_spectrum(wide.sum(axis=1).to_numpy(), GRID_HZ, band)
    return {"freq_hz": freqs, "amp_pu": amp / (wide.shape[1] * tdp_w)}


def facility_series(rows: dict[int, list[dict]], *, horizon_s: float = 900.0) -> dict:
    """Summed facility power on the 10 Hz grid and its 0.1-2 Hz component, for the time-aligned plot."""
    wide = _grid_series(rows, horizon_s)
    if len(wide) < GRID_HZ * 20:
        return {}
    total = wide.sum(axis=1).to_numpy()
    return {"t_epoch": wide.index.to_numpy() / GRID_HZ, "total_w": total, "band_w": _band(total, *BAND_HZ)}


class NoGridSource:
    label = "미연결"

    def snapshot(self, t_epoch: float | None = None) -> dict | None:
        return None

    def spectrum(self, t_epoch: float | None = None, band: tuple[float, float] = (0.05, 2.0)) -> dict | None:
        return None

    def series(self, t_epoch: float | None, horizon_s: float, max_hz: float = 10.0) -> dict | None:
        return None


OSC_BAND_HZ = (0.05, 2.0)  # electromechanical band shown in the simulated panel
ROCOF_WINDOW_S = 0.5      # RoCoF over a 500 ms window, not sample-to-sample noise
MIN_OSC_SAMPLE_HZ = 4.0   # below this the 0.05-2 Hz band cannot be resolved


def grid_indicators(t: np.ndarray, freq_hz: np.ndarray, nominal_hz: float) -> dict:
    """Measured counterparts of the simulated panel: swing (mHz), RoCoF, dominant oscillation."""
    t, freq_hz = np.asarray(t, float), np.asarray(freq_hz, float)
    out = {"freq_hz": float(freq_hz[-1]), "freq_dev_hz": float(freq_hz[-1] - nominal_hz),
           "max_abs_dev_hz": float(np.max(np.abs(freq_hz - nominal_hz))),
           "swing_mhz": float((freq_hz.max() - freq_hz.min()) * 1000), "osc_freq_hz": None,
           "rocof_max_hz_s": None}
    sample_hz = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
    out["sample_hz"] = float(sample_hz)
    if sample_hz <= 0:
        return out
    grid = t[0] + np.arange(int(round((t[-1] - t[0]) * sample_hz)) + 1) / sample_hz  # keeps the last sample
    f = np.interp(grid, t, freq_hz)
    k = max(1, int(round(ROCOF_WINDOW_S * sample_hz)))
    if len(f) > k:
        out["rocof_max_hz_s"] = float(np.max(np.abs(f[k:] - f[:-k])) / (k / sample_hz))
    if sample_hz >= MIN_OSC_SAMPLE_HZ and len(f) >= 32:
        x = f - np.polyval(np.polyfit(grid - grid[0], f, 1), grid - grid[0])  # remove slow drift
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / sample_hz)
        band = (freqs >= OSC_BAND_HZ[0]) & (freqs <= min(OSC_BAND_HZ[1], sample_hz / 2))
        if band.any() and spec[band].max() > 0:
            out["osc_freq_hz"] = float(freqs[band][np.argmax(spec[band])])
    return out


class CsvGridSource:
    """Grid-side samples measured at the facility's point of connection (PQ meter / PMU export).

    Columns: ``freq_hz`` (+ optional ``voltage_v``) and a time column, either ``t_epoch``
    (absolute, matched to the pipeline's data time) or ``t_s`` (seconds from the file start,
    replayed in a loop from the first snapshot; used by the example file).
    """

    def __init__(self, path: Path, nominal_hz: float = 60.0, horizon_s: float = 60.0):
        frame = pd.read_csv(path)
        self.relative = "t_epoch" not in frame and "t_s" in frame
        self.tcol = "t_s" if self.relative else "t_epoch"
        self.frame = frame.sort_values(self.tcol).reset_index(drop=True)
        self.nominal_hz = nominal_hz
        self.horizon_s = horizon_s
        self.label = f"수전점 계측 {Path(path).name}"
        self._anchor: float | None = None

    def _file_time(self, t_epoch: float | None) -> float | None:
        """Relative files start one horizon in (a full window at once) and advance with data time, looping."""
        if t_epoch is None or not self.relative:
            return t_epoch
        if self._anchor is None:
            self._anchor = t_epoch
        t0, t1 = float(self.frame[self.tcol].iloc[0]), float(self.frame[self.tcol].iloc[-1])
        return t0 + (self.horizon_s + t_epoch - self._anchor) % max(t1 - t0, 1e-9)

    def _window(self, t_epoch: float | None) -> pd.DataFrame:
        now = self._file_time(t_epoch)
        f = self.frame if now is None else self.frame[self.frame[self.tcol] <= now]
        return f[f[self.tcol] >= f[self.tcol].max() - self.horizon_s] if len(f) else f

    def spectrum(self, t_epoch: float | None = None, band: tuple[float, float] = (0.05, 2.0)) -> dict | None:
        """Measured frequency spectrum (mHz) over the same horizon, None if too sparse for the band."""
        f = self._window(t_epoch)
        if len(f) < 32:
            return None
        t, freq = f[self.tcol].to_numpy(float), f["freq_hz"].to_numpy(float)
        sample_hz = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
        if sample_hz < MIN_OSC_SAMPLE_HZ:
            return None
        grid = t[0] + np.arange(int(round((t[-1] - t[0]) * sample_hz)) + 1) / sample_hz
        y = np.interp(grid, t, freq)
        y = y - np.polyval(np.polyfit(grid - grid[0], y, 1), grid - grid[0])  # remove slow drift
        freqs, amp = amplitude_spectrum(y, sample_hz, (band[0], min(band[1], sample_hz / 2)))
        return {"freq_hz": freqs, "amp_mhz": amp * 1000}

    def series(self, t_epoch: float | None, horizon_s: float, max_hz: float = 10.0) -> dict | None:
        """Frequency deviation (mHz) on the pipeline's data clock over ``horizon_s``, at most ``max_hz``."""
        if t_epoch is None or len(self.frame) < 2:
            return None
        tf, ff = self.frame[self.tcol].to_numpy(float), self.frame["freq_hz"].to_numpy(float)
        hz = min(max_hz, (len(tf) - 1) / (tf[-1] - tf[0])) if tf[-1] > tf[0] else 0.0
        if hz <= 0:
            return None
        times = np.arange(t_epoch - horizon_s, t_epoch, 1.0 / hz)
        if self.relative:  # same mapping as _file_time, vectorised
            self._file_time(t_epoch)
            ft = tf[0] + (self.horizon_s + times - self._anchor) % max(tf[-1] - tf[0], 1e-9)
        else:
            keep = (times >= tf[0]) & (times <= tf[-1])
            times, ft = times[keep], times[keep]
        if len(times) < 2:
            return None
        return {"t_epoch": times, "freq_dev_mhz": (np.interp(ft, tf, ff) - self.nominal_hz) * 1000.0}

    def snapshot(self, t_epoch: float | None = None) -> dict | None:
        f = self._window(t_epoch)
        if len(f) < 3:
            return None
        out = grid_indicators(f[self.tcol].to_numpy(), f["freq_hz"].to_numpy(), self.nominal_hz)
        if "voltage_v" in f:
            out["voltage_v"] = float(f["voltage_v"].iloc[-1])
        return out
