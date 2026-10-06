"""One observed-waveform replay as a subprocess (runs in .venv-physics, ANDES needs Python 3.11).

    echo '{"power": [...], "sample_hz": 10, "test_system": "wecc",
           "facility": {"tdp_w": 450, "participation": 1.0, "facility_share": 0.01}}' \
      | .venv-physics/bin/python -m bit2watt_impl.physics.replay_worker

Reads one JSON request on stdin and writes one JSON result on stdout (last line).
The cyber pipeline (.venv, Python 3.10) calls this through pipeline/physics_bridge.py.
"""
from __future__ import annotations

import json
import logging
import sys

import numpy as np


def main() -> None:
    logging.disable(logging.WARNING)  # ANDES logs to stderr/stdout; keep stdout parseable
    req = json.loads(sys.stdin.read())
    from bit2watt_impl.physics.simulation import inject_observed_waveform_with_timeout

    out = inject_observed_waveform_with_timeout(
        np.asarray(req["power"], dtype=float),
        float(req["sample_hz"]),
        timeout_s=float(req.get("timeout_s", 60.0)),
        test_system=str(req.get("test_system", "wecc")),
        facility=req.get("facility"),
    )
    print("\n" + json.dumps(out, default=float))


if __name__ == "__main__":
    main()
