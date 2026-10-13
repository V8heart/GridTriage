"""Live engine v5: v4 scoring and session gate, v5 judgment, automatic research on 'unknown', enrolment queue.

Differences from pipeline.live.LiveEngine (unchanged, still used by app.py / app_v2.py):
  * sources also record GPU framebuffer memory (the v5 case features need it)
  * an alarm is judged by core_v5 (case memory fit + channels + gate (a)+(b))
  * every analysed alarm is saved as a pending case (an analyst may enrol it, also as a new normal)
  * the first 'unknown' verdict of an alarm starts research_v5 on its own worker; the result is written
    to the analyst queue and linked from the pending case. Research never changes the verdict.
  * when the approval ledger changes (an analyst enrolled or revoked a case), the case memory is reloaded
    before the next judgment, and enrolled classes are matched by case (judge_v5 rule c), so a confirmed
    threat is recognised on its next occurrence.

  python -m pipeline.v5.live_v5 --replay <session_id> [--speed 20] [--no-web]   (headless check)
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.recover_instant_power import recover_power_instant
from pipeline.live import ROW_KEYS, SESSIONS, EngineSettings, LiveEngine, NvmlSource, ReplaySource
from pipeline.v4_core import segment_bundle
from pipeline.v5.case_store import DEFAULT_DIR, CaseStore, safe_id
from pipeline.v5.core_v5 import analyze_run_v5, run_windows

ROW_KEYS_V5 = (*ROW_KEYS, "fb_used_mb")
QUEUE = Path(__file__).resolve().parents[2] / "dataset" / "v5" / "research_queue"


class NvmlSourceV5(NvmlSource):
    """NvmlSource plus framebuffer memory used (MB)."""

    def poll(self) -> list[dict]:
        rows = super().poll()
        for r in rows:
            try:
                r["fb_used_mb"] = self._nvml.nvmlDeviceGetMemoryInfo(self.handles[r["gpu_id"]]).used / 1024 ** 2
            except self._nvml.NVMLError:
                r["fb_used_mb"] = np.nan
        return rows


class ReplaySourceV5(ReplaySource):
    """ReplaySource that keeps fb_used_mb. Ground truth still never reaches the engine rows."""

    def __init__(self, session_id: str, speed: float = 10.0):
        frame = pd.read_csv(SESSIONS / session_id / "telemetry.csv", low_memory=False)
        frame["power_instant_w"] = recover_power_instant(frame["power_instant_w"]) if "power_instant_w" in frame else np.nan
        self.truth = frame.groupby("gpu_id")[["gt_label", "gt_variant"]].first().to_dict("index")
        frame = frame[[c for c in ROW_KEYS_V5 if c in frame]].sort_values("t_epoch")
        self.ticks = [g.to_dict("records") for _, g in frame.groupby((frame["t_epoch"] * 10).round())]
        self.speed = speed
        self.label = f"replay {session_id} x{speed:g}"
        self.session_id = session_id
        self.i = 0
        self.t0 = self.ticks[0][0]["t_epoch"] if self.ticks else 0.0
        self.wall0 = None


@dataclass
class EngineSettingsV5(EngineSettings):
    normal_rule: bool = True          # gate (b)
    research: bool = True             # research on the first 'unknown' of an alarm
    research_web: bool = True         # arXiv / Semantic Scholar (generic keywords only); False = local ATT&CK/ATLAS only
    store_dir: Path | None = None     # case memory + ledger (default dataset/v5/case_memory)
    queue_dir: Path = QUEUE


@dataclass
class ResearchState:
    jobs: dict[str, dict] = field(default_factory=dict)  # alarm_id -> {"status", "path", "top", ...}


class LiveEngineV5(LiveEngine):
    def __init__(self, source, retriever, settings: EngineSettingsV5 | None = None, stage1=None,
                 store: CaseStore | None = None, researcher=None):
        super().__init__(source, retriever, settings or EngineSettingsV5(), stage1)
        self.store = store or CaseStore(self.settings.store_dir or DEFAULT_DIR)
        self._memory = None
        self._memory_version = None
        self._memory_lock = threading.Lock()
        self.research_state = ResearchState()
        self._research_pool = ThreadPoolExecutor(max_workers=1)
        self.researcher = researcher  # tests replace research_v5

    # -- case memory -----------------------------------------------------
    def memory(self):
        """Current case memory; rebuilt when the approval ledger changed since the last load."""
        with self._memory_lock:
            version = self.store.version()
            if self._memory is None or version != self._memory_version:
                self._memory = self.store.memory()
                self._memory_version = version
            return self._memory

    def _run_rows(self, gpu: int, run: list[dict]) -> pd.DataFrame:
        t0, t1 = run[0]["t_end"] - self.settings.window_s, run[-1]["t_end"]
        with self._lock:
            rows = [r for r in self.state.rows.get(gpu, ()) if t0 <= r["t_epoch"] <= t1]
        return pd.DataFrame(rows)

    # -- judgment ----------------------------------------------------------
    def _submit(self, gpu: int, run: list[dict]) -> None:
        part = pd.DataFrame([{**r["window_row"], "u_score": r["u_score"], "p_value": r["p_value"],
                              "evidence_bool": r["evidence_bool"]} for r in run])
        seg = segment_bundle(part, self.stage1.config)
        aid = self.alarm_id(gpu, run)
        raw = self._run_rows(gpu, run)
        with self._lock:
            self.state.pending += 1
            self.state.pending_alarms[aid] = gpu

        def task():
            st = self.settings
            windows = pd.DataFrame()
            try:
                windows = run_windows(raw)
                result = analyze_run_v5(seg["bundle"], windows, self.retriever, self.memory(), model=st.llm_model,
                                        k=st.k, mode=st.retrieval_mode, host=st.ollama_host, n_windows=len(run),
                                        normal_rule=st.normal_rule, enrolled=self.store.enrolled_classes())
            except Exception as exc:
                result = {"query": None, "hits": [], "judgment": {"verdict": "invalid", "errors": [str(exc)]}}
            result.update(gpu_id=gpu, t_start=run[0]["t_end"] - st.window_s, t_end=run[-1]["t_end"],
                          n_windows=len(run), bundle=seg["bundle"], alarm_id=aid, engine="v5",
                          memory_version=self._memory_version)
            result["pending_case"] = self._save_pending(aid, gpu, run, windows, result)
            with self._lock:
                self.state.analyses.append(result)
                self.state.pending -= 1
                self.state.pending_alarms.pop(aid, None)
            if result["judgment"].get("verdict") == "unknown":
                self._maybe_research(aid, seg["bundle"], result, windows)

        self._pool.submit(task)

    def pending_case_id(self, aid: str) -> str:
        src = getattr(self.source, "session_id", None) or "live"
        return safe_id(f"{src}-gpu{aid.replace(':', '-t')}")

    def _save_pending(self, aid: str, gpu: int, run: list[dict], windows: pd.DataFrame, result: dict) -> str | None:
        if windows.empty:
            return None
        j = result["judgment"]
        cid = self.pending_case_id(aid)
        if cid in self.store.status():  # already enrolled or dismissed by an analyst: do not re-queue
            return cid
        w = windows.assign(session_id=getattr(self.source, "session_id", None) or "live", gpu_id=gpu)
        with self._lock:
            research = self.research_state.jobs.get(aid, {}).get("path")
        # Re-analyses of a growing run can change the LLM's pick; keep every verdict and make escalation sticky:
        # once an alarm was 'unknown' it stays flagged for the analyst whatever later re-analyses say.
        prev = self.store.pending_dir / f"{cid}.json"
        history = json.loads(prev.read_text(encoding="utf-8")).get("history", []) if prev.exists() else []
        history.append({"n_windows": len(run), "verdict": j.get("verdict"), "threat_card": j.get("threat_card"),
                        "gate_rule": (j.get("gate") or {}).get("rule")})
        escalated = any(h["verdict"] == "unknown" for h in history)
        self.store.save_pending(cid, w, {
            "escalated": escalated, "history": history,
            "alarm_id": aid, "gpu_id": gpu, "source": getattr(self.source, "label", ""),
            "t_start": result["t_start"], "t_end": result["t_end"], "n_windows": len(run),
            "verdict": j.get("verdict"), "threat_card": j.get("threat_card"), "gate": j.get("gate"),
            "explanation": j.get("explanation"), "case_fit": result.get("case", {}).get("class_p"),
            "research_queue": research, "memory_version": self._memory_version})
        return cid

    # -- research ----------------------------------------------------------
    def _maybe_research(self, aid: str, bundle: dict, result: dict, windows: pd.DataFrame) -> None:
        if not self.settings.research:
            return
        with self._lock:
            if aid in self.research_state.jobs:  # once per alarm; re-analyses of the same run do not re-trigger
                return
            self.research_state.jobs[aid] = {"status": "running", "started": time.time()}
        case, channels = result.get("case") or {}, result.get("channels") or {}

        def task():
            from pipeline.v5.research_v5 import research_v5, write_queue_item

            st = self.settings
            try:
                fn = self.researcher or research_v5
                out = fn(bundle, channels, case, model=st.llm_model, host=st.ollama_host, with_web=st.research_web)
                path = write_queue_item(self.pending_case_id(aid), out, st.queue_dir)
                top = [(h["name"], h["score"]) for h in out.get("hypotheses", [])[:4]]
                job = {"status": "done", "path": str(path), "top": top, "observed_tokens": out.get("observed_tokens")}
            except Exception as exc:
                job = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            with self._lock:
                self.research_state.jobs[aid].update(job)
            if job["status"] == "done":
                self._link_research(aid, job["path"])

        self._research_pool.submit(task)

    def _link_research(self, aid: str, path: str) -> None:
        """Add the research queue path to the alarm's pending case (it may have been saved first)."""
        pfile = self.store.pending_dir / f"{self.pending_case_id(aid)}.json"
        if pfile.exists():
            d = json.loads(pfile.read_text(encoding="utf-8"))
            d["research_queue"] = path
            tmp = pfile.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(pfile)

    def wait_idle(self, timeout: float = 3600.0) -> None:
        """Block until queued judgments and research jobs finished (headless runs and tests)."""
        self._pool.shutdown(wait=True)
        self._research_pool.shutdown(wait=True)

    def snapshot(self) -> dict:
        snap = super().snapshot()
        with self._lock:
            snap["research"] = {k: dict(v) for k, v in self.research_state.jobs.items()}
        snap["memory_version"] = self._memory_version
        return snap


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless v5 replay: alarms, verdicts, research, pending cases")
    parser.add_argument("--replay", required=True, help="session id under dataset/real/sessions")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--no-research", action="store_true")
    parser.add_argument("--no-web", action="store_true", help="research with the local ATT&CK/ATLAS only")
    parser.add_argument("--store", type=Path, default=DEFAULT_DIR, help="case memory + ledger directory")
    parser.add_argument("--queue", type=Path, default=QUEUE, help="research queue directory")
    args = parser.parse_args()
    from pipeline.kb.retriever import HybridRetriever

    src = ReplaySourceV5(args.replay, speed=args.speed)
    eng = LiveEngineV5(src, HybridRetriever(), EngineSettingsV5(llm_model=args.model, ollama_host=args.host,
                                                                 research=not args.no_research,
                                                                 research_web=not args.no_web,
                                                                 store_dir=args.store, queue_dir=args.queue))
    eng._loop()  # runs the whole replay in this thread
    eng.wait_idle()
    snap = eng.snapshot()
    for a in snap["analyses"]:
        j = a["judgment"]
        print(json.dumps({"alarm": a["alarm_id"], "n_windows": a["n_windows"], "verdict": j.get("verdict"),
                          "card": j.get("threat_card"), "gate": j.get("gate"), "errors": j.get("errors"),
                          "case_fit": a.get("case", {}).get("class_p"), "pending_case": a.get("pending_case")},
                         ensure_ascii=False))
    for aid, job in snap["research"].items():
        print("research", aid, json.dumps(job, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
