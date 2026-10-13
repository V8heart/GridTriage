"""Case memory: labelled 30 s windows of approved sessions, with conformal novelty and class-fit scores.

MalRAG (arXiv 2511.14129) retrieves labelled samples rather than prose and prunes each class
by its own distance spread; deep kNN OOD (Sun et al., ICML 2022) scores novelty by the
distance to the k-th nearest training feature without a distribution assumption. Here both
are calibrated per session, leave-one-session-out, so a p-value means "how unusual is this
segment's distance compared with held-out sessions that really belong to the memory / class".

  novelty_p   distance to all memory windows      -> small = unlike anything approved
  class_p[c]  distance to class c memory windows  -> small = does not look like a known c case

Classes with few sessions cannot carry their own conformal reference (4 sessions -> p >= 0.2),
so as in MalRAG each class distance is first scaled by that class's own leave-one-session-out
median distance, and the scaled ratios of all classes are pooled into one reference.
Normal is also compared within the declared job family ("normal@training"): a job that does
not look like honest jobs of the type it was declared as (a Mondrian, declared-context check).

Distances use observed channels only (power, instant power, utilisation, memory, processes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from pipeline.v5.raw_features import ALL_RAW

HEAVY = ("p_std", "p_swing", "p_ramp95", "i_std", "i_minus_p_std", "i_step_mad", "u_std", "m_std_mb", "m_range_mb")
K = 5
MIN_CLASS_SESSIONS = 2  # a class needs >= 2 sessions for a leave-one-session-out reference


def transform(df: pd.DataFrame, features=ALL_RAW) -> np.ndarray:
    X = df[list(features)].astype(float).fillna(0.0).copy()
    for c in HEAVY:
        if c in X:
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X.to_numpy()


def _seg_dist(nn: NearestNeighbors, Xq: np.ndarray, k: int) -> float:
    d = nn.kneighbors(Xq, n_neighbors=min(k, nn.n_samples_fit_))[0].mean(axis=1)
    return float(np.median(d))


def conformal_p(value: float, reference: np.ndarray) -> float:
    return float((1 + np.sum(reference >= value)) / (len(reference) + 1))


@dataclass
class CaseMemory:
    windows: pd.DataFrame                   # memory windows with 'cls', 'session_id', 'gpu_id'
    features: tuple = tuple(ALL_RAW)
    k: int = K
    scaler: RobustScaler = field(init=False)
    X: np.ndarray = field(init=False)
    ref_all: np.ndarray = field(init=False)
    ref_cls: dict = field(init=False)
    _nn_all: NearestNeighbors = field(init=False)
    _nn_cls: dict = field(init=False)

    def __post_init__(self):
        self.windows = self.windows.reset_index(drop=True)
        fam = self.windows.get("declared_job_family", pd.Series("", index=self.windows.index)).astype(str)
        self.windows["fam_key"] = None
        normal = self.windows.cls == "normal"
        self.windows.loc[normal, "fam_key"] = "normal@" + fam[normal]
        self.scaler = RobustScaler().fit(transform(self.windows, self.features))
        self.X = self.scaler.transform(transform(self.windows, self.features))
        self._nn_all = NearestNeighbors().fit(self.X)
        self._nn_cls = {c: NearestNeighbors().fit(self.X[self._mask(c)]) for c in self.keys}
        self._calibrate()

    @property
    def classes(self) -> list[str]:
        return sorted(self.windows.cls.unique())

    @property
    def keys(self) -> list[str]:
        fam = self.windows.dropna(subset=["fam_key"]).groupby("fam_key").session_id.nunique()
        return self.classes + sorted(fam[fam >= MIN_CLASS_SESSIONS].index)

    def _mask(self, key: str) -> np.ndarray:
        col = "fam_key" if "@" in key else "cls"
        return (self.windows[col] == key).to_numpy()

    def sessions_of(self, key: str) -> int:
        return int(self.windows.loc[self._mask(key), "session_id"].nunique())

    def _segments(self):
        return self.windows.groupby(["session_id", "gpu_id"]).groups.items()

    def _calibrate(self) -> None:
        """Leave-one-session-out distances of memory segments (novelty and per-class references)."""
        ref_all, ref_cls = [], {c: [] for c in self.keys}
        sess = self.windows.session_id.to_numpy()
        for (sid, _gid), idx in self._segments():
            idx = np.asarray(list(idx))
            keep = sess != sid
            Xq = self.X[idx]
            ref_all.append(_seg_dist(NearestNeighbors().fit(self.X[keep]), Xq, self.k))
            row = self.windows.iloc[idx[0]]
            for key in (row.cls, row.fam_key):
                if not isinstance(key, str) or key not in ref_cls:
                    continue
                own = keep & self._mask(key)
                if own.sum() >= self.k and self.sessions_of(key) >= MIN_CLASS_SESSIONS:
                    ref_cls[key].append(_seg_dist(NearestNeighbors().fit(self.X[own]), Xq, self.k))
        self.ref_all = np.asarray(ref_all)
        self.ref_cls = {c: np.asarray(v) for c, v in ref_cls.items() if len(v) >= MIN_CLASS_SESSIONS}
        self.scale = {c: float(np.median(v)) for c, v in self.ref_cls.items()}
        self.ref_ratio = np.concatenate([v / self.scale[c] for c, v in self.ref_cls.items()])

    def score(self, seg_windows: pd.DataFrame, declared_family: str | None = None) -> dict:
        """Novelty / class-fit p-values and nearest cases for one segment's windows."""
        Xq = self.scaler.transform(transform(seg_windows, self.features))
        d_all = _seg_dist(self._nn_all, Xq, self.k)
        class_p, class_d = {}, {}
        for c, nn in self._nn_cls.items():
            class_d[c] = _seg_dist(nn, Xq, self.k)
            if c in self.ref_cls:
                class_p[c] = round(conformal_p(class_d[c] / self.scale[c], self.ref_ratio), 3)
        # nearest memory sessions by window vote (for the prompt and the analyst)
        idx = self._nn_all.kneighbors(Xq, n_neighbors=self.k)[1].ravel()
        votes = self.windows.iloc[idx].groupby("cls").size().sort_values(ascending=False)
        fam_key = f"normal@{declared_family}" if declared_family else None
        return {
            "declared_normal_key": fam_key if fam_key in class_p else None,
            "declared_normal_p": class_p.get(fam_key) if fam_key else None,
            "novelty_p": round(conformal_p(d_all, self.ref_all), 3),
            "class_p": class_p,
            "class_dist": {c: round(v, 3) for c, v in class_d.items()},
            "nearest_case_votes": {c: round(int(n) / len(idx), 3) for c, n in votes.items()},
            "no_case_support": sorted(c for c in self.classes if c not in self.ref_cls),
        }

    def normal_quantiles(self, seg_windows: pd.DataFrame, cols: list[str]) -> dict:
        """Where the segment median of each column sits among memory normal windows (0..1)."""
        ref = self.windows[self.windows.cls == "normal"]
        out = {}
        for c in cols:
            v = float(seg_windows[c].median())
            out[c] = {"value": round(v, 3), "normal_q": round(float((ref[c] <= v).mean()), 2)}
        return out
