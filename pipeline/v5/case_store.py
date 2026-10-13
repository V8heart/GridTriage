"""Persistent case memory with an analyst approval ledger.

Layout (local, under dataset/ so it is never uploaded with the code):

  dataset/v5/case_memory/
    cases/<case_id>.csv        30 s window features of one approved case (no raw telemetry, no labels from gt_*)
    pending/<case_id>.json     an alarm waiting for an analyst: features, verdict, research queue path
    ledger.jsonl               append-only, hash-chained approval record

Ledger entries (one JSON per line):
  action      seed | enroll | revoke | dismiss (alarm reviewed, not enrolled)
  case_id, label, analyst, reason, ts (UTC ISO), source (alarm / session / seed split),
  prior       the pipeline verdict the analyst saw (enroll from pending), research queue path
  n_windows, features_sha256   hash of the case CSV, so a swapped file is detected
  prev_hash, hash              sha256 chain over the entry; `verify()` recomputes it

Only cases whose latest ledger action is seed/enroll are active. Revoking never deletes the CSV,
so the record of what was believed and when stays reconstructible.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.v5.raw_features import ALL_RAW

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "dataset" / "v5" / "case_memory"
GENESIS = "0" * 64
CASE_COLUMNS = ["case_id", "session_id", "gpu_id", "win", "declared_job_family", "cls", *ALL_RAW]
ACTIONS = ("seed", "enroll", "revoke", "dismiss")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:120]


def entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False).encode())


@dataclass
class CaseStore:
    root: Path = DEFAULT_DIR

    def __post_init__(self):
        self.root = Path(self.root)
        self.cases_dir = self.root / "cases"
        self.pending_dir = self.root / "pending"
        self.ledger_path = self.root / "ledger.jsonl"
        for d in (self.cases_dir, self.pending_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- ledger ------------------------------------------------------------
    @contextmanager
    def _locked(self):
        with (self.root / ".lock").open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def ledger(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        return [json.loads(l) for l in self.ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _append(self, entry: dict) -> dict:
        """Caller holds the lock."""
        if entry["action"] not in ACTIONS:
            raise ValueError(f"unknown action {entry['action']!r}")
        led = self.ledger()
        entry = {"seq": len(led), "ts": _now(), **entry, "prev_hash": led[-1]["hash"] if led else GENESIS}
        entry["hash"] = entry_hash(entry)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def version(self) -> str:
        """Changes whenever the ledger changes (the live engine reloads its memory on change)."""
        led = self.ledger()
        return led[-1]["hash"] if led else GENESIS

    def verify(self) -> list[str]:
        """Problems found in the hash chain or the case files (empty = intact)."""
        problems, prev = [], GENESIS
        for i, e in enumerate(self.ledger()):
            if e.get("seq") != i:
                problems.append(f"entry {i}: seq {e.get('seq')}")
            if e.get("prev_hash") != prev:
                problems.append(f"entry {i}: prev_hash does not chain")
            if entry_hash(e) != e.get("hash"):
                problems.append(f"entry {i}: hash mismatch (entry edited)")
            prev = e.get("hash")
            if e["action"] in ("seed", "enroll"):
                path = self.cases_dir / f"{e['case_id']}.csv"
                if not path.exists():
                    problems.append(f"entry {i}: case file missing {path.name}")
                elif _sha(path.read_bytes()) != e.get("features_sha256"):
                    problems.append(f"entry {i}: case file {path.name} changed after approval")
        return problems

    # -- cases -------------------------------------------------------------
    def status(self) -> dict[str, dict]:
        """case_id -> latest ledger entry."""
        latest = {}
        for e in self.ledger():
            latest[e["case_id"]] = e
        return latest

    def enrolled_classes(self) -> frozenset:
        """Labels with at least one active analyst-enrolled (not seed) case: judge_v5 rule (c) applies to them."""
        return frozenset(e["label"] for e in self.status().values() if e["action"] == "enroll" and e.get("label"))

    def active_ids(self) -> list[str]:
        return sorted(c for c, e in self.status().items() if e["action"] in ("seed", "enroll"))

    def _write_case(self, case_id: str, windows: pd.DataFrame, label: str) -> tuple[Path, str, int]:
        frame = windows.copy()
        frame["case_id"], frame["cls"] = case_id, label
        for col in CASE_COLUMNS:
            if col not in frame:
                frame[col] = None
        path = self.cases_dir / f"{case_id}.csv"
        frame[CASE_COLUMNS].to_csv(path, index=False)
        return path, _sha(path.read_bytes()), len(frame)

    def add_case(self, case_id: str, windows: pd.DataFrame, label: str, *, action: str, analyst: str, reason: str,
                 source: dict, prior: dict | None = None) -> dict:
        if action not in ("seed", "enroll"):
            raise ValueError("add_case is for seed/enroll")
        if not analyst or not reason:
            raise ValueError("analyst and reason are required for an approval record")
        if windows.empty:
            raise ValueError("a case needs at least one feature window")
        case_id = safe_id(case_id)
        with self._locked():
            current = self.status().get(case_id)
            if current and current["action"] in ("seed", "enroll"):
                raise ValueError(f"case {case_id} is already active (revoke it first)")
            _, digest, n = self._write_case(case_id, windows, label)
            return self._append({"action": action, "case_id": case_id, "label": label, "analyst": analyst,
                                 "reason": reason, "source": source, "prior": prior, "n_windows": n,
                                 "features_sha256": digest})

    def revoke(self, case_id: str, *, analyst: str, reason: str) -> dict:
        if not analyst or not reason:
            raise ValueError("analyst and reason are required")
        with self._locked():
            current = self.status().get(case_id)
            if not current or current["action"] == "revoke":
                raise ValueError(f"case {case_id} is not active")
            return self._append({"action": "revoke", "case_id": case_id, "label": current["label"],
                                 "analyst": analyst, "reason": reason, "source": {"revokes_seq": current["seq"]}})

    def windows(self) -> pd.DataFrame:
        frames = [pd.read_csv(self.cases_dir / f"{c}.csv") for c in self.active_ids()]
        if not frames:
            return pd.DataFrame(columns=CASE_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        out["session_id"] = out["session_id"].astype(str)  # live cases share "live"; keep cases apart for LOSO
        out.loc[out.session_id.isin(["live", "nan", "None"]), "session_id"] = out["case_id"]
        out["gpu_id"] = pd.to_numeric(out["gpu_id"], errors="coerce").fillna(0).astype(int)
        return out

    def memory(self):
        from pipeline.v5.exemplar_kb import CaseMemory

        win = self.windows()
        if win.empty:
            raise RuntimeError(f"case memory {self.root} is empty: run `python -m pipeline.v5.enroll seed` first")
        return CaseMemory(win)

    # -- pending (alarms waiting for an analyst) ---------------------------
    def save_pending(self, case_id: str, windows: pd.DataFrame, info: dict) -> Path:
        path = self.pending_dir / f"{safe_id(case_id)}.json"
        payload = {"case_id": safe_id(case_id), "saved": _now(), **info,
                   "windows": windows.to_dict("records")}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def pending(self) -> list[dict]:
        out = []
        for p in sorted(self.pending_dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            d.pop("windows", None)
            out.append(d)
        return out

    def approve_pending(self, case_id: str, label: str, *, analyst: str, reason: str) -> dict:
        path = self.pending_dir / f"{safe_id(case_id)}.json"
        if not path.exists():
            raise FileNotFoundError(f"no pending case {case_id}")
        d = json.loads(path.read_text(encoding="utf-8"))
        windows = pd.DataFrame(d["windows"])
        prior = {k: d.get(k) for k in ("verdict", "threat_card", "gate", "research_queue", "alarm_id", "escalated", "history")}
        entry = self.add_case(case_id, windows, label, action="enroll", analyst=analyst, reason=reason,
                              source={"kind": "alarm", "alarm_id": d.get("alarm_id"), "source": d.get("source"),
                                      "t_start": d.get("t_start"), "t_end": d.get("t_end")}, prior=prior)
        path.rename(path.with_suffix(".approved"))
        return entry

    def dismiss_pending(self, case_id: str, *, analyst: str, reason: str) -> dict:
        """The analyst reviewed the alarm and chose not to enroll it; that decision is recorded too."""
        if not analyst or not reason:
            raise ValueError("analyst and reason are required")
        path = self.pending_dir / f"{safe_id(case_id)}.json"
        if not path.exists():
            raise FileNotFoundError(f"no pending case {case_id}")
        d = json.loads(path.read_text(encoding="utf-8"))
        with self._locked():
            entry = self._append({"action": "dismiss", "case_id": safe_id(case_id), "label": None, "analyst": analyst,
                                  "reason": reason, "source": {"kind": "alarm", "alarm_id": d.get("alarm_id")},
                                  "prior": {k: d.get(k) for k in ("verdict", "threat_card", "gate", "research_queue", "escalated", "history")}})
        path.rename(path.with_suffix(".dismissed"))
        return entry
