"""Figures for the project brief (docs/figures/*.png). Runs in .venv-physics (matplotlib).

    .venv-physics/bin/python scripts/make_figures.py

Inputs are repository outputs only:
  docs/figures/_swma_power.csv      SWMA 1 s period, s-fdb3c79ef8d14b9a GPU0, 120-132 s (power_w, recovered power_instant_w)
  docs/figures/_timeline_data.json  official Stage1 (round5b/fold_00) windows on fold_00 test sessions
  dataset/eval/ibr_frequency_2d_sweep.csv, dataset/eval/modal_analysis.json   Physics Tier B
  RESULTS_v4.md §2.2                Stage2 ablation accuracies (dev 107 segments, gemma3:12b)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
for f in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
    path = Path("/usr/share/fonts/opentype/noto") / f
    if path.exists():
        font_manager.fontManager.addfont(str(path))
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})

BLUE, ORANGE, RED, GREEN, GREY, PURPLE = "#2166ac", "#e08214", "#d62728", "#1a9850", "#7f7f7f", "#6a51a3"


def box(ax, x, y, w, h, text, color=BLUE, dashed=False, fill=0.10, size=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.6,
                                edgecolor=color, facecolor=matplotlib.colors.to_rgba(color, fill),
                                linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color="#444", dashed=False, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
                                 color=color, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8, color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def fig_concept():
    fig, ax = canvas(10.4, 3.6)
    box(ax, 0.1, 0.9, 2.2, 1.9, "GPU 서버\n\nNVML 텔레메트리\n진행 로그\n스케줄러 신고 작업", GREY)
    ax.add_patch(FancyBboxPatch((2.9, 0.35), 4.3, 3.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=2, edgecolor=BLUE, facecolor="white"))
    ax.text(5.05, 3.05, "GridPulse", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    for x, text in ((3.12, "① 선별\n정상 기준선\n대비 이상"), (4.47, "② 검색\n위협 지식\n1,091건"),
                    (5.82, "③ 판정·설명\n로컬 LLM\n근거 인용")):
        box(ax, x, 1.85, 1.15, 0.95, text, BLUE, size=8.5)
    arrow(ax, 4.29, 2.32, 4.45, 2.32)
    arrow(ax, 5.64, 2.32, 5.80, 2.32)
    box(ax, 3.12, 0.6, 3.85, 0.95, "관측 한계를 스스로 선언\n판정 가능 ≤ 약 1 Hz · 더 빠른 변조는 '외부 계측 필요'",
        GREY, fill=0.06, size=8.2)
    box(ax, 8.2, 2.0, 2.1, 1.25, "관제 · 보안팀\n조사 우선순위\n+ 근거 문서", RED)
    box(ax, 8.2, 0.45, 2.1, 1.25, "계통 검증\nANDES Kundur·WECC\n(공개 테스트 계통)", GREEN)
    arrow(ax, 2.3, 1.85, 2.9, 1.85)
    arrow(ax, 7.2, 2.6, 8.2, 2.6)
    ax.text(7.7, 2.72, "경보·판정", ha="center", va="bottom", fontsize=8, color="#444")
    arrow(ax, 7.2, 1.1, 8.2, 1.1)
    ax.text(7.7, 1.22, "경보 구간\n파형", ha="center", va="bottom", fontsize=8, color="#444")
    ax.text(1.2, 0.55, "추가 계측기 없음", ha="center", fontsize=8.5, color=GREY)
    fig.savefig(OUT / "fig1_concept.png")


def fig_architecture():
    fig, ax = canvas(12.8, 5.2)
    y, w, gap = 3.35, 1.55, 0.27
    steps = [("Stage 0\n데이터 품질\n갱신 간격·값 변경률", GREY), ("Stage 1\n코호트 robust-z\n+ conformal (α=0.05)", BLUE),
             ("세션 게이트\n후보 5창 연속\n(약 90초)", BLUE), ("Stage 2\n증거 번들\n(고정 어휘)", BLUE),
             ("Stage 4\n하이브리드 검색\n+ 재정렬 (상위 8)", BLUE), ("Stage 5\nLLM 판정\n인용·증거 검증", BLUE)]
    xs = [0.15 + i * (w + gap) for i in range(len(steps))]
    for x, (text, color) in zip(xs, steps):
        box(ax, x, y, w, 1.25, text, color, size=8.5)
    for a in xs[:-1]:
        arrow(ax, a + w, y + 0.62, a + w + gap, y + 0.62)
    box(ax, 0.15, 4.85, 4.4, 0.32, "NVML 100 ms · 평균/순시 전력 · 사용률 · 프로세스 · 신고 작업", GREY, fill=0.05, size=8)
    arrow(ax, 0.9, 4.85, 0.9, y + 1.25)
    # row 2
    box(ax, 0.15, 1.7, 2.8, 1.1, "오프라인 적합 (train/cal만)\n코호트 기준선 · 증거 임계\nα · 세션 게이트 · KB 구축",
        GREY, dashed=True, fill=0.04, size=8.2)
    arrow(ax, 1.55, 2.8, xs[1] + w / 2, y, color=GREY, dashed=True)
    box(ax, xs[3], 1.7, xs[4] + w - xs[3], 1.1, "위협 지식베이스 1,091건\n위협 카드 10 (관측 시그니처)\n+ ATT&CK·ICS·ATLAS 1,021",
        PURPLE, size=8.2)
    arrow(ax, xs[4] + w / 2, 2.8, xs[4] + w / 2, y, color=PURPLE)
    box(ax, xs[5], 1.7, w, 1.1, "콘솔\n상태 카드·근거\n(조작 기능 없음)", RED, size=8.5)
    arrow(ax, xs[5] + w / 2, y, xs[5] + w / 2, 2.8)
    box(ax, xs[5] + w + 0.25, 1.7, 1.45, 1.1, "Stage 3\n프로세스·스케줄러\n(2차 과제)", GREY, dashed=True,
        fill=0.03, size=8)
    # row 3
    box(ax, 3.3, 0.2, 3.3, 1.1, "Physics 계층 (ANDES, 별도 환경)\n경보 구간 파형 → WECC/Kundur 재생\n회전속도 진동·RoCoF·모드 겹침",
        GREEN, size=8.2)
    arrow(ax, xs[2] + w / 2, y, xs[2] + w / 2, 1.3, color=GREEN)
    arrow(ax, 6.6, 0.75, xs[5] + 0.5, 1.7, color=GREEN)
    fig.savefig(OUT / "fig2_architecture.png")


def fig_observability():
    f = pd.read_csv(OUT / "_swma_power.csv")
    t = f["t"] - f["t"].min()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(t, f["power_instant_w"], color=ORANGE, lw=1.1, label="순시 전력 (power_instant_w)")
    ax.plot(t, f["power_w"], color=BLUE, lw=2.2, label="1초 평균 전력 (power_w)")
    sw_i = f["power_instant_w"].max() - f["power_instant_w"].min()
    sw_a = f["power_w"].max() - f["power_w"].min()
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("GPU 전력 (W)")
    ax.set_title(f"같은 SWMA 구간 (1초 주기, RTX 4090): 진폭 순시 {sw_i:.0f} W vs 1초 평균 {sw_a:.0f} W", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)
    ax.set_ylim(0, 520)
    fig.savefig(OUT / "fig3_observability.png")


def fig_session_gate():
    data = json.loads((OUT / "_timeline_data.json").read_text())
    picks = [("s-6ee2464205af7764", "정상 DDP 학습"), ("s-f9da550c34ee781e", "LTMA"),
             ("s-e20f23bb1450db38", "SWMA (jitter)")]
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 2.7), sharex=True)
    for ax, (sid, name) in zip(axes, picks):
        d = data[sid]
        pat = d["pattern"]
        in_alarm, alarm_at, i = set(), None, 0
        while i < len(pat):  # windows inside runs of >= 5 consecutive candidates
            j = i
            while j < len(pat) and pat[j] == "X":
                j += 1
            if j - i >= 5:
                in_alarm.update(range(i, j))
                alarm_at = i + 4 if alarm_at is None else alarm_at
            i = max(j, i + 1)
        for i, c in enumerate(pat):
            color = RED if i in in_alarm else (ORANGE if c == "X" else "#e6e6e6")
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=color))
        if alarm_at is not None:
            ax.annotate("경보", (alarm_at + 0.45, 1.0), (alarm_at + 0.45, 1.55), ha="center", fontsize=8.5,
                        color=RED, arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
        ax.set_xlim(0, 48)
        ax.set_ylim(0, 1.7)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylabel(f"{name}\n{'경보' if alarm_at is not None else '경보 없음'}", rotation=0, ha="right", va="center",
                      fontsize=8.5)
    axes[-1].set_xlabel("30초 창 번호 (15초 간격) · 회색 정상 / 주황 후보 / 빨강 5창 연속 → 세션 경보")
    fig.savefig(OUT / "fig4_session_gate.png")


def fig_ibr():
    f = pd.read_csv(ROOT / "dataset/eval/ibr_frequency_2d_sweep.csv")
    f = f[(f["kind"] == "square") & (f["converged"].astype(str) == "True")]
    modal = json.loads((ROOT / "dataset/eval/modal_analysis.json").read_text())
    modes = next(r for r in modal["results"] if r["target_penetration"] == 0.0)["modes"]
    lo, hi = min(m["freq_hz"] for m in modes), max(m["freq_hz"] for m in modes)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axvspan(0.1, 1.0, color=ORANGE, alpha=0.12, label="AI 학습 iteration 기본 주파수 (0.1–1 Hz)")
    ax.axvspan(lo, hi, ymin=0, ymax=0.08, color=GREEN, alpha=0.6, label=f"WECC 전기기계 모드 ({lo:.2f}–{hi:.2f} Hz)")
    for pen, color in ((0.0, BLUE), (0.3, PURPLE), (0.5, RED)):
        g = f[f["target_penetration"].round(2) == pen].sort_values("frequency_hz")
        inertia = g["inertia_reduction_frac"].iloc[0]
        ax.plot(g["frequency_hz"], g["osc_std"] * 1e4, marker="o", ms=3.5, lw=1.6, color=color,
                label=f"IBR {pen:.0%} (관성 −{inertia:.0%})")
    ax.set_xlabel("부하 변동 주파수 (Hz), 사각파")
    ax.set_ylabel("동기기 회전속도 표준편차\n(×1e-4 p.u.)", fontsize=9)
    ax.set_title("WECC 179-bus: IBR 비중별 같은 부하 변동(계통 부하의 1.2%)에 대한 동기기 응답", fontsize=10)
    ax.legend(fontsize=7.8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_ylim(bottom=0)
    fig.savefig(OUT / "fig5_ibr_response.png")


def fig_stage2():
    rows = [("문서 없음", 0.579, GREY), ("이전 RAG\n(설명문+TF-IDF)", 0.533, GREY), ("무작위 문서 8건", 0.570, GREY),
            ("KB 1,091건\n시그니처 질의+하이브리드+재정렬", 0.757, BLUE)]
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], width=0.6)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.012, f"{r[1]:.2f}", ha="center", fontsize=10,
                fontweight="bold" if r[2] == BLUE else "normal")
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("판정 정확도")
    ax.set_title("실측 107 세그먼트, gemma3:12b · KB 대비 나머지 모두 McNemar p < 0.001", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    fig.savefig(OUT / "fig6_stage2_accuracy.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_concept, fig_architecture, fig_observability, fig_session_gate, fig_ibr, fig_stage2):
        fn()
        print("wrote", fn.__name__)
