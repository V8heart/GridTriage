"""Real-time v4 pipeline engine (UI-independent).

A source yields one telemetry row per GPU per tick. Every ``stride_s`` of data
time the engine scores the latest ``window_s`` window per GPU with the frozen
Stage1 model. When a GPU has ``trigger_windows`` consecutive candidate
windows, the candidate run is summarised into an evidence bundle and sent to
retrieval + LLM judgment on a worker thread. The first time a run raises the
alarm, its power waveform is also replayed on a public test system (physics
bridge, own worker thread, optional). The UI only reads ``snapshot()``.

Sources:
  NvmlSource    polls NVML directly (power average + instant, util, processes)
  ReplaySource  streams a recorded session at ``speed`` x real time
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.declared_context import DECLARED_POOL
from dataset.recover_instant_power import recover_power_instant
from pipeline.physics_bridge import PhysicsSettings, replay
from pipeline.stage1_v2 import build_windows, score_window
from pipeline.v4_core import ROOT, Stage1Model, analyze_segment, load_stage1, segment_bundle, session_gate_windows

SESSIONS = ROOT / "dataset" / "real" / "sessions"
ROW_KEYS = ("t_epoch", "gpu_id", "gpu_model", "power_w", "power_instant_w", "util_gpu_pct",
            "observed_n_procs", "declared_job_type", "declared_job_family")


def declared_fields(job_type: str) -> dict:
    return {"declared_job_type": job_type, "declared_job_family": DECLARED_POOL[job_type]["family"]}


class NvmlSource:
    """Live NVML reader. ``declared`` maps gpu_id -> scheduler-declared job type."""

    def __init__(self, declared: dict[int, str], interval_s: float = 0.1):
        import pynvml

        from bit2watt_impl.collect_telemetry import field_value_to_float

        self._nvml = pynvml
        self._field = field_value_to_float
        pynvml.nvmlInit()
        self.handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in declared}
        self.names = {g: pynvml.nvmlDeviceGetName(h) for g, h in self.handles.items()}
        self.declared = declared
        self.interval_s = interval_s
        self.label = "live NVML"

    def poll(self) -> list[dict]:
        nv = self._nvml
        now = time.time()
        rows = []
        for g, h in self.handles.items():
            inst = None
            try:
                inst_mw = self._field(nv.nvmlDeviceGetFieldValues(h, [nv.NVML_FI_DEV_POWER_INSTANT])[0])
                inst = None if inst_mw is None else inst_mw / 1000.0
            except nv.NVMLError:
                pass
            try:
                procs = len(nv.nvmlDeviceGetComputeRunningProcesses(h))
            except nv.NVMLError:
                procs = np.nan
            name = self.names[g]
            rows.append({
                "t_epoch": now, "gpu_id": g, "gpu_model": name.decode() if isinstance(name, bytes) else name,
                "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0, "power_instant_w": inst,
                "util_gpu_pct": nv.nvmlDeviceGetUtilizationRates(h).gpu, "observed_n_procs": procs,
                **declared_fields(self.declared[g]),
            })
        return rows

    def sleep(self) -> None:
        time.sleep(self.interval_s)


class ReplaySource:
    """Replays a recorded session. Ground truth stays out of the rows the engine sees."""

    def __init__(self, session_id: str, speed: float = 10.0):
        frame = pd.read_csv(SESSIONS / session_id / "telemetry.csv", low_memory=False)
        frame["power_instant_w"] = recover_power_instant(frame["power_instant_w"]) if "power_instant_w" in frame else np.nan
        self.truth = frame.groupby("gpu_id")[["gt_label", "gt_variant"]].first().to_dict("index")
        frame = frame[[c for c in ROW_KEYS if c in frame]].sort_values("t_epoch")
        self.ticks = [g.to_dict("records") for _, g in frame.groupby((frame["t_epoch"] * 10).round())]
        self.speed = speed
        self.label = f"replay {session_id} x{speed:g}"
        self.i = 0
        self.t0 = self.ticks[0][0]["t_epoch"] if self.ticks else 0.0
        self.wall0: float | None = None

    def poll(self) -> list[dict]:
        if self.i >= len(self.ticks):
            return []
        if self.wall0 is None:  # clock starts when the engine starts pulling
            self.wall0 = time.monotonic()
        target = self.t0 + (time.monotonic() - self.wall0) * self.speed
        out = []
        while self.i < len(self.ticks) and self.ticks[self.i][0]["t_epoch"] <= target:
            out.extend(self.ticks[self.i])
            self.i += 1
        return out

    @property
    def finished(self) -> bool:
        return self.i >= len(self.ticks)

    def sleep(self) -> None:
        time.sleep(0.05)


@dataclass
class EngineSettings:
    window_s: float = 30.0
    stride_s: float = 15.0
    trigger_windows: int = field(default_factory=session_gate_windows)  # session-calibrated gate
    reanalyze_every: int = 4
    llm_model: str = "gemma3:12b"
    retrieval_mode: str = "hybrid_rerank"
    k: int = 8
    ollama_host: str | None = None
    buffer_s: float = 900.0
    physics: PhysicsSettings | None = None  # None = no physics replay


@dataclass
class EngineState:
    rows: dict[int, deque] = field(default_factory=dict)
    windows: list[dict] = field(default_factory=list)
    analyses: list[dict] = field(default_factory=list)
    physics: dict[str, dict] = field(default_factory=dict)  # alarm_id -> replay result (or {"status": "running"})
    pending: int = 0
    pending_alarms: dict[str, int] = field(default_factory=dict)  # alarm_id -> gpu, judgment in flight
    error: str | None = None


class LiveEngine:
    def __init__(self, source, retriever, settings: EngineSettings | None = None, stage1: Stage1Model | None = None):
        self.source = source
        self.retriever = retriever
        self.settings = settings or EngineSettings()
        self.stage1 = stage1 or load_stage1()
        self.state = EngineState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._physics_pool = ThreadPoolExecutor(max_workers=1)
        self.physics_runner = None  # tests replace the subprocess
        self._next_score_t: float | None = None
        self._runs: dict[int, list[dict]] = {}
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                self.ingest(self.source.poll())
                if getattr(self.source, "finished", False):
                    break
                self.source.sleep()
        except Exception as exc:  # surfaced in the UI instead of dying silently
            self.state.error = f"{type(exc).__name__}: {exc}"

    # -- data path -------------------------------------------------------
    def ingest(self, rows: list[dict]) -> None:
        if not rows:
            return
        maxlen = int(self.settings.buffer_s * 12)
        with self._lock:
            for r in rows:
                self.state.rows.setdefault(int(r["gpu_id"]), deque(maxlen=maxlen)).append(r)
        t = max(r["t_epoch"] for r in rows)
        if self._next_score_t is None:
            self._next_score_t = min(r["t_epoch"] for r in rows) + self.settings.window_s
        # A large batch (slow consumer, fast replay) still gets every stride scored.
        while t >= self._next_score_t:
            self.score_latest(self._next_score_t)
            self._next_score_t += self.settings.stride_s

    def _window_frame(self, t_end: float) -> pd.DataFrame:
        with self._lock:
            rows = [r for buf in self.state.rows.values() for r in buf if t_end - self.settings.window_s <= r["t_epoch"] <= t_end]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["session_id"] = "live"
        frame["timestamp"] = frame["t_epoch"] - frame["t_epoch"].min()
        hz = 1.0 / max(frame.groupby("gpu_id")["t_epoch"].diff().median(), 1e-3)
        frame["sample_hz"] = hz
        return frame

    def score_latest(self, t_end: float) -> list[dict]:
        frame = self._window_frame(t_end)
        if frame.empty:
            return []
        s1 = self.stage1
        hz = float(frame["sample_hz"].iloc[0])
        windows = build_windows(frame, window_s=self.settings.window_s - 1.0 / hz, stride_s=self.settings.window_s,
                                config=s1.config)
        out = []
        for _, row in windows.iterrows():
            scored = score_window(row.to_dict(), s1.baseline, s1.calibration, s1.config)
            rec = {**{k: row.get(k) for k in ("gpu_id", "mean_w", "swing_abs_w", "inst_step_mad_w", "dominant_freq_hz",
                                              "declared_job_type", "observed_n_procs")},
                   "t_end": t_end, "window_row": row.to_dict(),
                   **{k: scored[k] for k in ("u_score", "p_value", "is_candidate", "idle_status", "active_evidence",
                                              "evidence_bool", "driving_feature", "impact_level")}}
            out.append(rec)
            self._track_run(rec)
        with self._lock:
            self.state.windows.extend(out)
            self.state.windows = self.state.windows[-400:]
        return out

    def _track_run(self, rec: dict) -> None:
        gpu = int(rec["gpu_id"])
        if not rec["is_candidate"]:
            self._runs[gpu] = []
            return
        run = self._runs.setdefault(gpu, [])
        run.append(rec)
        n = len(run)
        trig, every = self.settings.trigger_windows, self.settings.reanalyze_every
        if n == trig or (n > trig and (n - trig) % every == 0):
            self._submit(gpu, list(run))
        if n == trig and self.settings.physics and self.settings.physics.enabled:
            self._submit_physics(gpu, list(run))

    @staticmethod
    def alarm_id(gpu: int, run: list[dict]) -> str:
        return f"{gpu}:{run[0]['t_end']:.1f}"

    def run_waveform(self, gpu: int, run: list[dict]) -> tuple[np.ndarray, float]:
        """Power over the alarmed run (instant power when recorded), and its sample rate."""
        t0, t1 = run[0]["t_end"] - self.settings.window_s, run[-1]["t_end"]
        with self._lock:
            rows = [r for r in self.state.rows.get(gpu, ()) if t0 <= r["t_epoch"] <= t1]
        frame = pd.DataFrame(rows)
        inst = frame["power_instant_w"] if "power_instant_w" in frame else pd.Series(dtype=float)
        power = inst if inst.notna().mean() > 0.8 else frame["power_w"]
        hz = 1.0 / max(float(frame["t_epoch"].diff().median()), 1e-3)
        return power.interpolate(limit_direction="both").to_numpy(dtype=float), hz

    def _submit_physics(self, gpu: int, run: list[dict]) -> None:
        aid = self.alarm_id(gpu, run)
        power, hz = self.run_waveform(gpu, run)
        with self._lock:
            self.state.physics[aid] = {"status": "running"}

        def task():
            result = replay(power, hz, self.settings.physics, runner=self.physics_runner)
            with self._lock:
                self.state.physics[aid] = {"status": "done", **result}

        self._physics_pool.submit(task)

    def _submit(self, gpu: int, run: list[dict]) -> None:
        part = pd.DataFrame([{**r["window_row"], "u_score": r["u_score"], "p_value": r["p_value"],
                              "evidence_bool": r["evidence_bool"]} for r in run])
        seg = segment_bundle(part, self.stage1.config)
        aid = self.alarm_id(gpu, run)
        with self._lock:
            self.state.pending += 1
            self.state.pending_alarms[aid] = gpu

        def task():
            st = self.settings
            try:
                result = analyze_segment(seg["bundle"], self.retriever, model=st.llm_model, k=st.k,
                                         mode=st.retrieval_mode, host=st.ollama_host)
            except Exception as exc:
                result = {"query": None, "hits": [], "judgment": {"verdict": "invalid", "errors": [str(exc)]}}
            result.update(gpu_id=gpu, t_start=run[0]["t_end"] - st.window_s, t_end=run[-1]["t_end"],
                          n_windows=len(run), bundle=seg["bundle"], alarm_id=self.alarm_id(gpu, run))
            with self._lock:
                self.state.analyses.append(result)
                self.state.pending -= 1
                self.state.pending_alarms.pop(aid, None)

        self._pool.submit(task)

    # -- read side -------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            rows = {g: list(buf) for g, buf in self.state.rows.items()}
            return {"rows": rows, "windows": list(self.state.windows), "analyses": list(self.state.analyses),
                    "physics": dict(self.state.physics), "pending": self.state.pending,
                    "pending_alarms": dict(self.state.pending_alarms),
                    "run_len": {g: len(r) for g, r in self._runs.items()},
                    "trigger_windows": self.settings.trigger_windows, "error": self.state.error,
                    "source": getattr(self.source, "label", ""), "stage1": self.stage1.name}


def observability(rows: list[dict]) -> dict:
    """Effective refresh from the buffer: interval between value changes of power_w."""
    if len(rows) < 20:
        return {}
    frame = pd.DataFrame(rows[-600:])
    t = frame["t_epoch"].to_numpy()
    p = frame["power_w"].to_numpy()
    changed = np.r_[True, p[1:] != p[:-1]]
    ct = t[changed]
    refresh = float(np.median(np.diff(ct))) if len(ct) > 2 else float("nan")
    poll = float(np.median(np.diff(t)))
    return {"poll_ms": poll * 1000, "refresh_ms": refresh * 1000, "change_ratio": float(changed.mean()),
            "nyquist_hz": 0.5 / refresh if refresh and refresh > 0 else float("nan")}
