"""Write examples/grid/pcc_measurement_example.csv: a SYNTHETIC point-of-connection measurement.

    .venv/bin/python scripts/make_grid_example.py

It only shows the input format a facility PQ meter / PMU export would have (t_s, freq_hz,
voltage_v at 30 samples/s, 60 Hz system) so the console's grid panel can be switched from the
test-system simulation to "measured" mode in a demo. It is not a measurement of any real grid
and has no relation to the replayed GPU sessions (its 0.37 Hz component is a generic
inter-area-like oscillation, deliberately not tied to any workload period).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "grid" / "pcc_measurement_example.csv"


def main(seconds: float = 600.0, hz: float = 30.0, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / hz)
    drift = np.cumsum(rng.normal(0.0, 0.00004, len(t)))  # slow load–generation imbalance, Hz
    drift -= np.linspace(drift[0], drift[-1], len(t))     # keep the file stationary for looping
    osc = 0.008 * np.sin(2 * np.pi * 0.37 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 240.0))
    freq = 60.0 + drift + osc + rng.normal(0.0, 0.001, len(t))
    voltage = 22_900.0 * (1 + 0.002 * np.sin(2 * np.pi * 0.37 * t + 0.6)) + rng.normal(0.0, 8.0, len(t))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_s": t.round(4), "freq_hz": freq.round(5), "voltage_v": voltage.round(1)}).to_csv(OUT, index=False)
    print("wrote", OUT, len(t), "rows")


if __name__ == "__main__":
    main()
