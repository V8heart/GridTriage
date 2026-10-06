"""Cyber pipeline -> physics replay bridge (plan_ui_physics.md).

The pipeline runs in .venv (Python 3.10); ANDES lives in .venv-physics (3.11), so a
replay is a subprocess (bit2watt_impl/physics/replay_worker.py) with a JSON request.
Results are cached by request. The output answers "how would a public test system
(Kundur / WECC) respond if this pattern ran at facility scale", not "the grid now".
"""
from __future__ import annotations

import functools
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHYSICS_PYTHON = ROOT / ".venv-physics" / "bin" / "python"
MODAL = ROOT / "dataset" / "eval" / "modal_analysis.json"
MODE_TOL = 0.10  # relative frequency tolerance for "forcing overlaps a mode"


@dataclass
class PhysicsSettings:
    enabled: bool = True
    test_system: str = "wecc"
    tdp_w: float = 450.0  # RTX 4090
    participation: float = 1.0
    facility_share: float = 0.01
    timeout_s: float = 60.0

    def facility(self) -> dict:
        return {"tdp_w": self.tdp_w, "participation": self.participation, "facility_share": self.facility_share}


def available() -> bool:
    return PHYSICS_PYTHON.exists()


@functools.lru_cache(maxsize=1)
def system_modes() -> list[dict]:
    """WECC electromechanical modes (IBR 0%) from the offline modal analysis."""
    if not MODAL.exists():
        return []
    report = json.loads(MODAL.read_text())
    base = next((r for r in report.get("results", []) if r.get("target_penetration") == 0.0), None)
    return [m for m in (base or {}).get("modes", []) if m.get("freq_hz")]


def forcing_frequency(power: np.ndarray, sample_hz: float) -> float | None:
    """Dominant frequency of the GPU power waveform in 0.05 Hz .. Nyquist."""
    x = np.asarray(power, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 16:
        return None
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_hz)
    band = freqs >= 0.05
    if not band.any() or spec[band].max() <= 0:
        return None
    return float(freqs[band][np.argmax(spec[band])])


def mode_overlap(freq_hz: float | None, modes: list[dict] | None = None) -> dict | None:
    """Closest system mode within MODE_TOL of the forcing frequency (or of its 2nd/3rd harmonic)."""
    if not freq_hz:
        return None
    modes = system_modes() if modes is None else modes
    best = None
    for harmonic in (1, 2, 3):
        f = freq_hz * harmonic
        for m in modes:
            rel = abs(m["freq_hz"] - f) / m["freq_hz"]
            if rel <= MODE_TOL and (best is None or rel < best["rel_gap"]):
                best = {"mode_hz": m["freq_hz"], "damping_ratio": m.get("damping_ratio"), "harmonic": harmonic,
                        "rel_gap": rel}
    return best


def _request(power: np.ndarray, sample_hz: float, settings: PhysicsSettings) -> dict:
    return {"power": [round(float(v), 3) for v in power], "sample_hz": float(sample_hz),
            "test_system": settings.test_system, "facility": settings.facility(), "timeout_s": settings.timeout_s}


_CACHE: dict[str, dict] = {}


def replay(power: np.ndarray, sample_hz: float, settings: PhysicsSettings | None = None, *,
           runner=None) -> dict:
    """Run one replay (cached). ``runner(request) -> dict`` replaces the subprocess in tests."""
    settings = settings or PhysicsSettings()
    power = np.asarray(power, dtype=float)
    power = power[np.isfinite(power)]
    req = _request(power, sample_hz, settings)
    key = hashlib.sha1(json.dumps(req, sort_keys=True).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    forcing = forcing_frequency(power, sample_hz)
    try:
        out = (runner or _subprocess_runner)(req)
    except Exception as exc:  # a failed replay is reported, never fatal to the cyber pipeline
        out = {"converged": False, "failure_reason": f"{type(exc).__name__}: {exc}"}
    out = {**out, "forcing_freq_hz": forcing, "mode_overlap": mode_overlap(forcing),
           "settings": asdict(settings), "scope": "public test system response, not the present grid state"}
    _CACHE[key] = out
    return out


def _subprocess_runner(req: dict) -> dict:
    if not available():
        raise FileNotFoundError(f"{PHYSICS_PYTHON} (run ./setup_physics_env.sh)")
    proc = subprocess.run([str(PHYSICS_PYTHON), "-m", "bit2watt_impl.physics.replay_worker"], cwd=ROOT,
                          input=json.dumps(req), capture_output=True, text=True, timeout=req["timeout_s"] + 30)
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError((proc.stderr or proc.stdout).strip().splitlines()[-1:] or f"exit {proc.returncode}")
    return json.loads(lines[-1])
