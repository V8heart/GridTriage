import numpy as np

from dataset.recover_instant_power import recover_power_instant


def test_recovers_denormal_milliwatts_and_keeps_watts():
    corrupted = np.array([331596], dtype=np.uint64).view(np.float64)[0]
    # The CSV round-trip keeps six significant digits.
    corrupted = float(f"{corrupted:.6g}")
    out = recover_power_instant(np.array([corrupted, 287.5, np.nan, 0.0]))
    assert abs(out[0] - 331.596) < 0.002
    assert out[1] == 287.5
    assert np.isnan(out[2])
    assert out[3] == 0.0
