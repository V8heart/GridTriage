"""Short GPU occupancy/power sweep. Not a capture. Requires an idle box."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _busy_compute() -> list[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ["nvidia-smi missing"]
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _one_point(args, *, kind: str, extra: list[str], seconds: float) -> dict:
    argv = [
        sys.executable,
        "-m",
        {
            "train": "workloads.llm_workloads",
            "infer": "workloads.llm_inference",
            "vision": "workloads.vision_workloads",
        }[kind],
        "--max-seconds",
        str(seconds),
        "--gpu-id",
        str(args.gpu_id),
        *extra,
    ]
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT, check=False, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    oom = "out of memory" in (proc.stderr or "").lower() or "runtime_oom" in (proc.stdout or "")
    summary = {}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    return {
        "kind": kind,
        "argv": extra,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "oom": oom,
        "summary": summary,
        "stderr_tail": (proc.stderr or "")[-400:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-shared-gpu-safe", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=35.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dataset" / "real" / "private" / "probes" / "capacity_sweep.json",
    )
    args = parser.parse_args()
    if not args.confirm_shared_gpu_safe:
        raise SystemExit("--confirm-shared-gpu-safe is required")
    busy = _busy_compute()
    if busy:
        raise SystemExit("GPU compute apps present; not starting sweep: " + json.dumps(busy))

    points = []
    for vocab, seq, batch in (
        (4096, 256, 8),
        (4096, 256, 16),
        (4096, 512, 8),
        (4096, 512, 16),
        (4096, 512, 32),
        (8192, 256, 8),
        (8192, 512, 8),
        (8192, 512, 16),
    ):
        extra = [
            "--mode",
            "finetune",
            "--preset",
            "small",
            "--seq-len",
            str(seq),
            "--batch-size",
            str(batch),
            "--vocab-size",
            str(vocab),
        ]
        point = _one_point(args, kind="train", extra=extra, seconds=min(args.seconds, 45.0))
        point["intended_vocab"] = vocab
        point["seq_len"] = seq
        point["batch_size"] = batch
        points.append(point)
    points.append(
        _one_point(
            args,
            kind="infer",
            extra=["--mode", "batch", "--preset", "small", "--seq-len", "256", "--batch-size", "8"],
            seconds=25.0,
        )
    )
    points.append(
        _one_point(
            args,
            kind="vision",
            extra=["--mode", "single", "--batch-size", "32"],
            seconds=25.0,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(points), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
