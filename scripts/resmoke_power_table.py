"""120s power table for the eight resmoke workloads. Does not write labels.csv."""
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


WORKLOADS = [
    (
        "llm_pretrain_ddp",
        [
            str(Path(sys.executable).parent / "torchrun"),
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "workloads.llm_workloads",
            "--mode",
            "ddp",
            "--preset",
            "small",
            "--seq-len",
            "256",
            "--probe",
            "--probe-target-w",
            "300",
        ],
    ),
    (
        "llm_pretrain_fsdp",
        [
            str(Path(sys.executable).parent / "torchrun"),
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "workloads.llm_workloads",
            "--mode",
            "fsdp",
            "--preset",
            "small",
            "--seq-len",
            "256",
            "--probe",
            "--probe-target-w",
            "300",
        ],
    ),
    (
        "llm_finetune",
        [
            sys.executable,
            "-m",
            "workloads.llm_workloads",
            "--mode",
            "finetune",
            "--preset",
            "small",
            "--seq-len",
            "256",
            "--gpu-id",
            "0",
            "--probe",
            "--probe-target-w",
            "300",
        ],
    ),
    (
        "swma_basic",
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "basic",
            "--period",
            "1.0",
            "--matrix-size",
            "4096",
            "--gpu-id",
            "0",
        ],
    ),
    (
        "swma_piggyback",
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "piggyback",
            "--host",
            "llm",
            "--gpu-id",
            "0",
        ],
    ),
    (
        "swma_mimicry",
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "mimicry",
            "--host",
            "llm",
            "--gpu-id",
            "0",
        ],
    ),
    (
        "ltma",
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "ltma",
            "--host",
            "llm",
            "--aux-width",
            "2048",
            "--gpu-id",
            "0",
        ],
    ),
    (
        "cryptojacking",
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "cryptojacking",
            "--matrix-size",
            "4096",
            "--gpu-id",
            "0",
        ],
    ),
]


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


def _nvml_snapshot() -> list[dict]:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            rows = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                rows.append(
                    {
                        "gpu_id": index,
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        "util": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        "mem_used_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024),
                    }
                )
            return rows
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return []


def _summarize(samples: list[list[dict]]) -> dict:
    by_gpu: dict[int, list[dict]] = {}
    for snap in samples:
        for row in snap:
            by_gpu.setdefault(int(row["gpu_id"]), []).append(row)
    out = {}
    for gpu_id, rows in by_gpu.items():
        watts = [r["power_w"] for r in rows]
        utils = [r["util"] for r in rows]
        mems = [r["mem_used_mb"] for r in rows]
        out[str(gpu_id)] = {
            "n": len(rows),
            "mean_w": sum(watts) / len(watts) if watts else None,
            "max_w": max(watts) if watts else None,
            "min_w": min(watts) if watts else None,
            "mean_util": sum(utils) / len(utils) if utils else None,
            "max_mem_mb": max(mems) if mems else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-shared-gpu-safe", action="store_true")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--only", default=None, help="comma-separated workload names")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dataset" / "real" / "private" / "probes" / "resmoke_power.json",
    )
    args = parser.parse_args()
    if not args.confirm_shared_gpu_safe:
        raise SystemExit("--confirm-shared-gpu-safe is required")
    busy = _busy_compute()
    if busy:
        raise SystemExit("GPU compute apps present; not starting resmoke: " + json.dumps(busy))

    selected = {name.strip() for name in args.only.split(",")} if args.only else None
    rows = []
    for name, argv in WORKLOADS:
        if selected is not None and name not in selected:
            continue
        extra = (
            ["--max-seconds", str(args.duration)]
            if "workloads.llm_workloads" in argv
            else ["--duration", str(args.duration)]
        )
        if name in {"swma_piggyback", "swma_mimicry"}:
            extra.extend(["--progress-log", str(args.out.with_suffix(f".{name}.jsonl"))])
        cmd = [*argv, *extra]
        print(f"starting {name}", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        samples: list[list[dict]] = []
        try:
            while proc.poll() is None:
                snap = _nvml_snapshot()
                if snap:
                    samples.append(snap)
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        stdout, _ = proc.communicate() if proc.stdout else ("", None)
        summary = {}
        for line in reversed((stdout or "").splitlines()):
            if line.startswith("{"):
                try:
                    summary = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
        stats = _summarize(samples)
        rows.append(
            {
                "workload": name,
                "returncode": proc.returncode,
                "stats": stats,
                "summary": {
                    key: summary.get(key)
                    for key in (
                        "period_requested_s",
                        "period_actual_s",
                        "period_floored",
                        "host_period_s",
                        "ltma_mode",
                        "baseline_w",
                        "aux_width",
                        "micro_batch",
                        "mean_w",
                    )
                    if key in summary
                },
            }
        )
        print(json.dumps({"workload": name, "returncode": proc.returncode, "stats": stats}), flush=True)
        time.sleep(5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
