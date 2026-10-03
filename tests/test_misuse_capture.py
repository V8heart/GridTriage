"""Misuse workloads: real workload argv, forced false declaration, attack labels."""
import argparse

from dataset import run_capture as rc


def _plan(workload):
    ns = argparse.Namespace(
        workload=workload, gpu_id=0, gpu_ids_override=None, duration=480.0, interval_ms=100.0, period=1.0,
        duty_cycle=0.5, preset=None, seq_len=None, grad_accum=None, batch_size=None, rps=None, dtype=None,
        max_new_tokens=None, dataloader=None, host="llm", matrix_size=None, active_streams=None, aux_width=None,
        probe_target_w=300.0, session_id=None, run_id=None, group_id=None, capture_key=None, seed=7,
        pre_capture_s=10.0, post_capture_s=10.0, warmup_s=0.0, preflight_s=5.0, probe_cmd=None, proc_poll_s=1.0,
        progress_mask_seed=7, progress_mask_drop_prob=0.4, declared_policy="pool_random", declared_job_type=None,
        dry_run=True, allow_busy=False, confirm_shared_gpu_safe=False, skip_finalize=True)
    return rc.build_plan(ns)


def test_llmjacking_is_serving_declared_as_notebook():
    plan = _plan("llmjacking")
    assert plan["private"]["gt_label"] == "llmjacking"
    assert "bit2watt_impl.misuse_workloads" in plan["workload"]
    assert plan["declared_by_gpu"]["0"]["declared_job_type"] == "notebook"
