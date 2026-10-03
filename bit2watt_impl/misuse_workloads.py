"""Misuse attacks that reuse real GPU workloads under a false or abusive context.

Each variant runs an actual workload on this machine (nothing synthetic) and
reports its whole run as the attack interval, so window labels follow the
standard gt_attack_intervals_epoch path:

  llmjacking_serving  1.3B LLM serving on a GPU declared as a notebook
  hidden_finetune     LLM fine-tuning on a GPU declared as inference
  sponge_long_gen     serving where every request forces long generation (energy-latency inputs)
  burst_irregular     irregular full-load / idle steps of random length (load ramps, no period)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time

VARIANTS = ("llmjacking_serving", "hidden_finetune", "sponge_long_gen", "burst_irregular")


def _argv(args) -> list[str]:
    py = sys.executable
    common = ["--gpu-id", str(args.gpu_id), "--seed", str(args.seed)]
    if args.progress_log:
        common += ["--progress-log", args.progress_log]
    if args.variant == "llmjacking_serving":
        return [py, "-m", "workloads.llm_inference", "--mode", "bursty", "--preset", "serve_1b", "--dtype", "fp16",
                "--rps", "2", "--batch-size", "4", "--seq-len", "512", "--max-new-tokens", "128",
                "--max-seconds", str(args.duration), *common]
    if args.variant == "hidden_finetune":
        return [py, "-m", "workloads.llm_workloads", "--mode", "finetune", "--preset", "small",
                "--batch-size", "64", "--seq-len", "512",
                "--duration", str(args.duration), *common]
    if args.variant == "sponge_long_gen":
        return [py, "-m", "workloads.llm_inference", "--mode", "serving", "--preset", "serve_1b", "--dtype", "fp16",
                "--rps", "1", "--batch-size", "8", "--seq-len", "512", "--max-new-tokens", "480",
                "--max-seconds", str(args.duration), *common]
    return [py, "-m", "bit2watt_impl.misuse_workloads", "--burst-child", "--variant", args.variant,
            "--duration", str(args.duration), "--gpu-id", str(args.gpu_id), "--seed", str(args.seed)]


def burst_child(args) -> None:
    import torch

    torch.cuda.set_device(args.gpu_id)
    rng = random.Random(args.seed)
    a = torch.randn(4096, 4096, device=f"cuda:{args.gpu_id}", dtype=torch.float16)
    b = torch.randn_like(a)
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        on_until = min(deadline, time.monotonic() + rng.uniform(1.0, 12.0))
        while time.monotonic() < on_until:
            for _ in range(4):
                torch.mm(a, b)
            torch.cuda.synchronize()  # keep the host clock and GPU work aligned
        time.sleep(max(0.0, min(deadline - time.monotonic(), rng.uniform(1.0, 12.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--burst-child", action="store_true")
    args, _ = parser.parse_known_args()
    if args.burst_child:
        burst_child(args)
        return
    start = time.time()
    proc = subprocess.run(_argv(args), check=False)
    end = time.time()
    print(json.dumps({"ok": proc.returncode == 0, "variant": args.variant, "duration_s": args.duration,
                      "gt_attack_intervals_epoch": {str(args.gpu_id): [[start, end]]}}))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
