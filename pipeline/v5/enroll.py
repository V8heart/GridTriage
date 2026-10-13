"""Analyst CLI for the v5 case memory: review alarms, approve them as cases, keep an approval record.

  python -m pipeline.v5.enroll seed                         # first run: approved cases = Stage1 fit split
  python -m pipeline.v5.enroll list                         # pending alarms + active cases per label
  python -m pipeline.v5.enroll show <case_id>               # pending detail (verdict, case fit, research) or ledger history
  python -m pipeline.v5.enroll approve <case_id> --label sponge_attack --analyst kim --reason "..."
  python -m pipeline.v5.enroll approve <case_id> --label normal --analyst kim --reason "new serving workload"
  python -m pipeline.v5.enroll dismiss <case_id> --analyst kim --reason "maintenance window"
  python -m pipeline.v5.enroll enroll-session <session_id> --gpu 0 --label X --analyst kim --reason "..."
  python -m pipeline.v5.enroll revoke <case_id> --analyst kim --reason "mislabelled"
  python -m pipeline.v5.enroll verify                       # hash chain + case file integrity
  python -m pipeline.v5.enroll history [--case <case_id>]

Labels: 'normal', 'unobservable', a threat card name (corpus/*.md), or a new class with --new-class
(every enrolled class is matched from its cases: judge_v5 rule (c)). A running LiveEngineV5
reloads the memory on its next judgment after any approve / revoke.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pipeline.v5.case_store import DEFAULT_DIR, CaseStore, safe_id
from pipeline.v5.core_v5 import NON_THREAT, card_names, run_windows

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ROOT / "dataset" / "real" / "sessions"
MANIFEST = ROOT / "dataset/real/pipeline/refit/{model}/{fold}/split_manifest.json"
OBSERVED = ["t_epoch", "gpu_id", "power_w", "power_instant_w", "util_gpu_pct", "fb_used_mb", "observed_n_procs",
            "declared_job_type", "declared_job_family"]


def known_labels(store: CaseStore) -> set[str]:
    return NON_THREAT | (card_names() - BENIGN_CARDS) | {e["label"] for e in store.status().values() if e.get("label")}


BENIGN_CARDS = {"normal_workloads", "benign_periodic"}


def check_label(store: CaseStore, label: str, new_class: bool) -> str:
    """Validated memory class for ``label`` (benign card names are the 'normal' class)."""
    if label in BENIGN_CARDS:
        print(f"label {label!r} is a benign card; enrolling as 'normal'")
        return "normal"
    if label in known_labels(store):
        return label
        return
    if not new_class:
        sys.exit(f"unknown label {label!r}. Known: {sorted(known_labels(store))}. "
                 "Use --new-class to enrol a threat that has no card yet.")
    if safe_id(label) != label or not label.islower():
        sys.exit("a new class label must be lowercase snake_case")
    return label


def cmd_seed(store: CaseStore, args) -> None:
    from pipeline.v5.raw_features import TELEMETRY, build_windows

    manifest = json.loads(Path(str(MANIFEST).format(model=args.model, fold=args.fold)).read_text())
    sessions = set(manifest["train"]) | set(manifest["cal"])
    cache = ROOT / "dataset/eval/v5/windows_raw.pkl"
    win = pd.read_pickle(cache) if cache.exists() and not args.rebuild else build_windows(TELEMETRY)
    win = win[win.session_id.isin(sessions)]
    active = set(store.active_ids())
    n = 0
    for (sid, gid), part in win.groupby(["session_id", "gpu_id"]):
        cid = safe_id(f"seed-{sid}-gpu{gid}")
        if cid in active:
            continue
        store.add_case(cid, part, str(part.cls.iloc[0]), action="seed", analyst=args.analyst,
                       reason=f"{args.model}/{args.fold} train+cal split (the sessions Stage1 was fitted on)",
                       source={"kind": "seed", "model": args.model, "fold": args.fold, "session_id": sid, "gpu_id": int(gid)})
        n += 1
    print(f"seeded {n} cases ({len(active)} already active) -> {store.root}")


def cmd_list(store: CaseStore, args) -> None:
    pend = store.pending()
    print(f"# pending alarms ({len(pend)})")
    for d in pend:
        fit = {k: v for k, v in (d.get("case_fit") or {}).items() if "@" not in k}
        flag = "UNKNOWN " if d.get("escalated") else ""
        print(f"  {flag}{d['case_id']}  latest={d.get('verdict')}:{d.get('threat_card')}  windows={d.get('n_windows')}  "
              f"fit={fit}  research={'yes' if d.get('research_queue') else 'no'}")
    win = store.windows()
    print(f"\n# active cases ({win.case_id.nunique() if len(win) else 0})")
    if len(win):
        print(win.groupby("cls").agg(cases=("case_id", "nunique"), windows=("case_id", "size")).to_string())


def cmd_show(store: CaseStore, args) -> None:
    pfile = store.pending_dir / f"{safe_id(args.case_id)}.json"
    if pfile.exists():
        d = json.loads(pfile.read_text(encoding="utf-8"))
        d.pop("windows", None)
        print(json.dumps(d, ensure_ascii=False, indent=1, default=str))
        if d.get("research_queue") and Path(d["research_queue"]).exists():
            print("\n" + Path(d["research_queue"]).read_text(encoding="utf-8"))
        return
    hist = [e for e in store.ledger() if e["case_id"] == args.case_id]
    if not hist:
        sys.exit(f"no pending case or ledger entry {args.case_id!r}")
    for e in hist:
        print(json.dumps(e, ensure_ascii=False, indent=1))


def cmd_approve(store: CaseStore, args) -> None:
    label = check_label(store, args.label, args.new_class)
    e = store.approve_pending(args.case_id, label, analyst=args.analyst, reason=args.reason)
    print(f"enrolled {e['case_id']} as {e['label']} (seq {e['seq']}, {e['n_windows']} windows, hash {e['hash'][:12]})")


def cmd_dismiss(store: CaseStore, args) -> None:
    e = store.dismiss_pending(args.case_id, analyst=args.analyst, reason=args.reason)
    print(f"dismissed {e['case_id']} (seq {e['seq']})")


def cmd_enroll_session(store: CaseStore, args) -> None:
    label = check_label(store, args.label, args.new_class)
    path = SESSIONS / args.session_id / "telemetry.csv"
    frame = pd.read_csv(path, usecols=lambda c: c in OBSERVED + ["warmup"], low_memory=False)  # no gt_* columns
    frame = frame[frame.gpu_id == args.gpu]
    if "warmup" in frame and not args.keep_warmup:
        frame = frame[~frame["warmup"].astype(str).str.lower().eq("true")]
    if args.t_start is not None:
        frame = frame[frame.t_epoch >= args.t_start]
    if args.t_end is not None:
        frame = frame[frame.t_epoch <= args.t_end]
    windows = run_windows(frame)
    if windows.empty:
        sys.exit("no complete 30 s window in the selected span")
    windows = windows.assign(session_id=args.session_id, gpu_id=args.gpu)
    cid = args.case_id or safe_id(f"session-{args.session_id}-gpu{args.gpu}")
    e = store.add_case(cid, windows, label, action="enroll", analyst=args.analyst, reason=args.reason,
                       source={"kind": "session", "session_id": args.session_id, "gpu_id": args.gpu,
                               "t_start": args.t_start, "t_end": args.t_end})
    print(f"enrolled {e['case_id']} as {e['label']} (seq {e['seq']}, {e['n_windows']} windows)")


def cmd_revoke(store: CaseStore, args) -> None:
    e = store.revoke(args.case_id, analyst=args.analyst, reason=args.reason)
    print(f"revoked {e['case_id']} (seq {e['seq']})")


def cmd_verify(store: CaseStore, args) -> None:
    problems = store.verify()
    led = store.ledger()
    print(f"ledger entries: {len(led)}, active cases: {len(store.active_ids())}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("ledger chain and case files intact")


def cmd_history(store: CaseStore, args) -> None:
    for e in store.ledger():
        if args.case and e["case_id"] != args.case:
            continue
        if not args.case and e["action"] == "seed" and not args.with_seed:
            continue
        print(f"{e['seq']:>4} {e['ts']} {e['action']:<7} {e['case_id']:<45} {str(e.get('label')):<20} "
              f"{e['analyst']}: {e['reason']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=DEFAULT_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seed")
    p.add_argument("--model", default="round5b")
    p.add_argument("--fold", default="fold_00")
    p.add_argument("--analyst", default="system:seed")
    p.add_argument("--rebuild", action="store_true", help="recompute features instead of the diag cache")
    sub.add_parser("list")
    p = sub.add_parser("show")
    p.add_argument("case_id")
    for name in ("approve", "dismiss", "revoke"):
        p = sub.add_parser(name)
        p.add_argument("case_id")
        if name == "approve":
            p.add_argument("--label", required=True)
            p.add_argument("--new-class", action="store_true")
        p.add_argument("--analyst", required=True)
        p.add_argument("--reason", required=True)
    p = sub.add_parser("enroll-session")
    p.add_argument("session_id")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--new-class", action="store_true")
    p.add_argument("--analyst", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--t-start", type=float)
    p.add_argument("--t-end", type=float)
    p.add_argument("--case-id")
    p.add_argument("--keep-warmup", action="store_true")
    sub.add_parser("verify")
    p = sub.add_parser("history")
    p.add_argument("--case")
    p.add_argument("--with-seed", action="store_true")

    args = parser.parse_args(argv)
    store = CaseStore(args.store)
    {"seed": cmd_seed, "list": cmd_list, "show": cmd_show, "approve": cmd_approve, "dismiss": cmd_dismiss,
     "enroll-session": cmd_enroll_session, "revoke": cmd_revoke, "verify": cmd_verify,
     "history": cmd_history}[args.cmd](store, args)


if __name__ == "__main__":
    main()
