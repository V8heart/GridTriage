"""Recover power_instant_w written by the pre-fix collector.

The old collector read the NVML field-value union through ``dVal`` even though
NVML_FI_DEV_POWER_INSTANT is an unsigned int in milliwatts, so every value was
stored as the denormal double sharing those bits (~1e-319). Reinterpreting the
double's bits as uint64 gives the milliwatt count back. CSV keeps six
significant digits, which is enough for 1 mW resolution below 1 kW.

Values written by the fixed collector (normal watts) pass through unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Any genuine board power is far above this; corrupted values are < 5e-318.
_DENORMAL_MAX = 1e-300


def recover_power_instant(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return instant power in watts, decoding corrupted denormal values."""
    data = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=np.float64)
    out = data.copy()
    corrupted = np.isfinite(data) & (data > 0) & (data < _DENORMAL_MAX)
    if corrupted.any():
        bits = data[corrupted].view(np.uint64)
        out[corrupted] = bits.astype(np.float64) / 1000.0
    return out
