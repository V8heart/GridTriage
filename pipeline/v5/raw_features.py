"""Window features from raw NVML telemetry, including channels the v4 bundle drops.

v4 sends the LLM a power-only pattern summary plus boolean evidence. NVML also records
GPU utilisation, framebuffer memory and the process count, which separate workloads that
have the same power shape (e.g. a hidden fine-tune holds optimizer state in memory, a
miner does not). Features here are observed-only: no gt_* column and no declared string
except the declared job family, which the scheduler provides at inference time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TELEMETRY = [ROOT / "dataset/real/pipeline/telemetry_v4_inst.csv",
             ROOT / "dataset/real/pipeline/telemetry_holdout_v2.csv",
             ROOT / "dataset/real/pipeline/telemetry_misuse_v1.csv"]
LABELS = ROOT / "dataset/real/index/labels.csv"
COLS = ["t_epoch", "session_id", "gpu_id", "power_w", "power_instant_w", "util_gpu_pct", "fb_used_mb",
        "observed_n_procs", "declared_job_type", "declared_job_family", "gt_label", "gt_is_attack", "gt_variant", "warmup"]
WINDOW_S, STRIDE_S, HZ = 30.0, 15.0, 10.0
TDP_W = 450.0

POWER_FEATURES = ["p_mean", "p_std", "p_p05", "p_p95", "p_swing", "p_ramp95", "p_fdom", "p_fstrength",
                  "p_spec_entropy", "p_lowfreq_frac", "p_acf1s", "p_acf5s", "i_std", "i_minus_p_std", "i_step_mad"]
UTIL_FEATURES = ["u_mean", "u_std", "u_frac_hi", "u_frac_lo", "up_corr", "w_per_util"]
MEM_FEATURES = ["m_mean_gb", "m_std_mb", "m_range_mb"]
PROC_FEATURES = ["n_procs"]
ALL_RAW = POWER_FEATURES + UTIL_FEATURES + MEM_FEATURES + PROC_FEATURES

NORMAL_VARIANTS_UNOBS = {"swma_piggyback", "swma_mimicry"}


def class_of(gt_label: str, variant: str | None) -> str:
    """Evaluation class: normal | unobservable | attack card name (same key as stage2_v4_dataset)."""
    if str(gt_label).startswith("normal"):
        return "normal"
    if variant in NORMAL_VARIANTS_UNOBS:
        return "unobservable"
    return {"swma": "swma", "ltma": "ltma", "cryptojacking": "cryptojacking"}.get(str(gt_label), str(gt_label))


def _spectrum(x: np.ndarray, hz: float):
    x = x - x.mean()
    if np.allclose(x, 0):
        return None, None
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / hz)
    return freqs[1:], spec[1:]


def _acf(x: np.ndarray, lag: int) -> float:
    if lag >= len(x) or np.std(x) < 1e-9:
        return 0.0
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    return float(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12))


def window_features(w: pd.DataFrame) -> dict:
    p = w["power_w"].to_numpy(float)
    inst = w["power_instant_w"].to_numpy(float) if "power_instant_w" in w else p
    inst = np.where(np.isfinite(inst), inst, p)
    u = w["util_gpu_pct"].to_numpy(float)
    m = w["fb_used_mb"].to_numpy(float)
    out = {
        "p_mean": p.mean(), "p_std": p.std(), "p_p05": np.percentile(p, 5), "p_p95": np.percentile(p, 95),
        "p_swing": np.percentile(p, 95) - np.percentile(p, 5),
    }
    sec = p[: len(p) // int(HZ) * int(HZ)].reshape(-1, int(HZ)).mean(axis=1) if len(p) >= 2 * HZ else p
    out["p_ramp95"] = float(np.percentile(np.abs(np.diff(sec)), 95)) if len(sec) > 1 else 0.0
    freqs, spec = _spectrum(p, HZ)
    if spec is None:
        out.update(p_fdom=0.0, p_fstrength=0.0, p_spec_entropy=0.0, p_lowfreq_frac=0.0)
    else:
        prob = spec / spec.sum()
        out["p_fdom"] = float(freqs[np.argmax(spec)])
        out["p_fstrength"] = float(spec.max() / spec.sum())
        out["p_spec_entropy"] = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(len(prob)))
        out["p_lowfreq_frac"] = float(spec[freqs < 0.1].sum() / spec.sum())
    out["p_acf1s"], out["p_acf5s"] = _acf(p, int(HZ)), _acf(p, int(5 * HZ))
    out["i_std"] = inst.std()
    out["i_minus_p_std"] = float(np.std(inst - p))
    out["i_step_mad"] = float(np.median(np.abs(np.diff(inst)))) if len(inst) > 1 else 0.0
    out["u_mean"], out["u_std"] = u.mean(), u.std()
    out["u_frac_hi"], out["u_frac_lo"] = float((u >= 90).mean()), float((u <= 5).mean())
    out["up_corr"] = float(np.corrcoef(u, p)[0, 1]) if u.std() > 1e-6 and p.std() > 1e-6 else 0.0
    out["w_per_util"] = float(p.mean() / (u.mean() + 5.0))
    out["m_mean_gb"], out["m_std_mb"], out["m_range_mb"] = m.mean() / 1024, m.std(), m.max() - m.min()
    out["n_procs"] = float(np.nanmedian(w["observed_n_procs"].to_numpy(float)))
    return {k: float(v) for k, v in out.items()}


def target_gpus() -> set[tuple[str, int]]:
    lab = pd.read_csv(LABELS, usecols=["session_id", "gpu_id", "gpu_role", "valid"])
    lab = lab[(lab.gpu_role == "target") & (lab.valid.astype(str) != "False")]
    return {(str(s), int(g)) for s, g in zip(lab.session_id, lab.gpu_id)}


def build_windows(paths=TELEMETRY) -> pd.DataFrame:
    """One row per 30 s window of each target GPU's ground-truth span (attack span / whole normal run)."""
    targets = target_gpus()
    rows = []
    for path in paths:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
        for (sid, gid), g in df.groupby(["session_id", "gpu_id"], sort=False):
            if (str(sid), int(gid)) not in targets:
                continue
            labels = g["gt_label"].astype(str)
            session_label = next((l for l in labels.unique() if l != "normal_idle"), None)
            if session_label is None:
                continue
            g = g[~g["warmup"].astype(str).str.lower().eq("true")] if "warmup" in g else g
            is_atk = g["gt_is_attack"].astype(str).str.lower().eq("true")
            span = g[is_atk] if not session_label.startswith("normal") else g[labels.loc[g.index] == session_label]
            if span.empty:
                continue
            variant = span["gt_variant"].dropna().astype(str)
            variant = variant.iloc[0] if len(variant) else None
            for feats in frame_windows(span):
                rows.append({"session_id": str(sid), "gpu_id": int(gid),
                             "declared_job_family": str(span["declared_job_family"].iloc[0]),
                             "declared_job_type": str(span["declared_job_type"].iloc[0]),
                             "gt_label": session_label, "gt_variant": variant,
                             "cls": class_of(session_label, variant), **feats})
    return pd.DataFrame(rows)


def frame_windows(frame: pd.DataFrame, window_s: float = WINDOW_S, stride_s: float = STRIDE_S,
                  hz: float = HZ) -> list[dict]:
    """Feature dicts of every full window of one GPU's rows (needs >= 80% of the expected samples)."""
    if frame.empty:
        return []
    frame = frame.sort_values("t_epoch")
    t = frame["t_epoch"].to_numpy(float)
    t0, out, k = t.min(), [], 0
    while t0 + k * stride_s + window_s <= t.max() + 1e-6:
        lo = t0 + k * stride_s
        w = frame[(t >= lo) & (t < lo + window_s)]
        k += 1
        if len(w) >= window_s * hz * 0.8:
            out.append({"win": k - 1, "t_start": float(lo), **window_features(w)})
    return out
