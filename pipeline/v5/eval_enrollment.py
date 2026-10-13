"""Case enrollment curve: after an analyst confirms n sessions of a new threat, does case memory recognise the rest?

No LLM. Memory starts as round5b fold_00 train+cal (normal, swma, crypto, unobservable cases). For each class
that memory has never seen, n of its sessions are enrolled (every choice of n, averaged) and the remaining
sessions of that class are judged by the case-based rule; the eval-set normal segments measure the
false-alarm cost of each enrolment.

Case-based rule (deterministic, alpha fixed 0.05):
  top = class with the largest share of the k nearest windows
  if fit_p[top] >= alpha (or top has < 2 sessions, so no reference): verdict = top
  else: unknown

  python -m pipeline.v5.eval_enrollment
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from pipeline.v5.eval_v5 import md, memory_sessions
from pipeline.v5.exemplar_kb import CaseMemory

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "eval" / "v5"
NEW = ["ltma", "sponge_attack", "hidden_ml_training", "llmjacking", "burst_ramp_attack"]
ALPHA = 0.05


def case_verdict(memory: CaseMemory, part: pd.DataFrame) -> str:
    sc = memory.score(part, part.declared_job_family.iloc[0])
    top = max(sc["nearest_case_votes"], key=sc["nearest_case_votes"].get)
    p = sc["class_p"].get(top)
    return top if p is None or p >= ALPHA else "unknown"


def main() -> None:
    win = pd.read_pickle(OUT / "windows_raw.pkl")
    mem_s = memory_sessions()
    base = win[win.session_id.isin(mem_s)]
    rest = win[~win.session_id.isin(mem_s)]
    normals = rest[rest.cls == "normal"]
    rows = []
    for cls in NEW:
        sessions = sorted(rest.loc[rest.cls == cls, "session_id"].unique())
        for n in range(0, min(3, len(sessions) - 1) + 1):
            for enrolled in itertools.combinations(sessions, n):
                memory = CaseMemory(pd.concat([base, rest[rest.session_id.isin(enrolled)]]))
                held = rest[(rest.cls == cls) & ~rest.session_id.isin(enrolled)]
                for (sid, gid), part in held.groupby(["session_id", "gpu_id"]):
                    rows.append({"cls": cls, "n_enrolled": n, "kind": "new", "verdict": case_verdict(memory, part)})
                for (sid, gid), part in normals.groupby(["session_id", "gpu_id"]):
                    rows.append({"cls": cls, "n_enrolled": n, "kind": "normal", "verdict": case_verdict(memory, part)})
            print(cls, n, flush=True)
    df = pd.DataFrame(rows)
    new = df[df.kind == "new"].assign(correct=lambda d: d.verdict == d.cls, unknown=lambda d: d.verdict == "unknown",
                                       called_normal=lambda d: d.verdict == "normal")
    nrm = df[df.kind == "normal"].assign(ok=lambda d: d.verdict == "normal",
                                          as_new=lambda d: d.verdict == d.cls)
    t1 = new.groupby(["cls", "n_enrolled"])[["correct", "unknown", "called_normal"]].mean().round(2)
    t2 = nrm.groupby(["cls", "n_enrolled"])[["ok", "as_new"]].mean().round(3)
    text = ("# 사례 등록 곡선 (LLM 없음, 사례 기억 규칙)\n\n새 클래스 세션 n개를 분석가가 확인해 등록한 뒤, 나머지 세션 판정.\n\n"
            "## 새 클래스의 나머지 세션\n\n" + md(t1) + "\n\n## 평가 세트 정상 세그먼트 (등록의 오탐 비용)\n\n" + md(t2) + "\n")
    (OUT / "eval_enrollment.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
