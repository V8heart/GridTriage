"""GridTriage console v2: one-screen layout of the same v4 pipeline as app.py.

    .venv/bin/streamlit run app_v2.py

The first screen holds everything needed to follow an alarm: a status strip, per-GPU tiles,
and three bordered panels (GPU power, latest Agent verdict, grid coupling). History, window
scores, retrieved documents and physics details sit below the fold in tabs.
Like app.py, the page only renders what pipeline/live.py computes and has no workload-control
buttons. Physics replay on a public test system is a sensitivity check, not the present grid state.
"""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dataset.declared_context import DECLARED_POOL
from pipeline import physics_bridge
from pipeline.grid_monitor import (CsvGridSource, NoGridSource, facility_forcing, facility_series,
                                   facility_spectrum)
from pipeline.kb.retriever import MODES, HybridRetriever, default_device
from pipeline.live import EngineSettings, LiveEngine, NvmlSource, ReplaySource, observability

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "dataset" / "real" / "index" / "labels.csv"
EXAMPLE_GRID = ROOT / "examples" / "grid" / "pcc_measurement_example.csv"
ICON_DIR = ROOT / "assets" / "llm_icons"  # LobeHub icons (MIT), identify the model only
PANEL_H = 450  # px; the three main panels share this height so the first screen stays aligned
TDP_W = physics_bridge.PhysicsSettings().tdp_w
LOW_DAMPING = 0.05  # modes below this damping ratio are drawn as the ones to watch

RED, GREEN, ORANGE, GREY, PURPLE, BLUE = "#d62728", "#2ca02c", "#ff7f0e", "#7f7f7f", "#9467bd", "#2166ac"
VERDICT_COLOR = {"attack": RED, "normal": GREEN, "unobservable": GREY, "unknown": ORANGE, "invalid": PURPLE}
VERDICT_TEXT = {"attack": "공격 의심", "unknown": "미확인 이상", "normal": "정상 (경보 후 판정)",
                "unobservable": "관측 대역 밖", "invalid": "판정 실패"}
SYSTEM_NAME = {"wecc": "WECC 179-bus", "kundur_ieeest": "Kundur 2-area"}
EMPTY = {"rows": {}, "windows": [], "analyses": [], "physics": {}, "pending": 0, "pending_alarms": {},
         "run_len": {}, "trigger_windows": 5, "error": None, "source": "", "stage1": ""}

st.set_page_config(page_title="GridTriage Console", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 1rem; max-width: 100%;}
div[data-testid="stVerticalBlock"] {gap: 0.55rem;}
.gp-bar {display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:6px 2px 2px 2px;}
.gp-title {font-size:1.35rem; font-weight:800; margin-right:6px;}
.gp-chip {font-size:0.8rem; padding:2px 9px; border-radius:999px; border:1px solid rgba(128,128,128,.35);
          white-space:nowrap;}
.gp-chip b {font-weight:700;}
.gp-tile {box-sizing:border-box; border:1px solid var(--c); border-left:6px solid var(--c); border-radius:8px;
          padding:7px 12px; background:color-mix(in srgb, var(--c) 9%, transparent); height:84px; overflow:hidden;
          margin-bottom:14px;}
.gp-tl {font-size:0.75rem; opacity:.75; letter-spacing:.02em;}
.gp-tv {font-size:1.12rem; font-weight:800; color:var(--c); line-height:1.35; white-space:nowrap;
        overflow:hidden; text-overflow:ellipsis;}
.gp-ts {font-size:0.78rem; opacity:.85; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.gp-h {display:flex; align-items:baseline; gap:8px; font-weight:800; font-size:1.02rem; margin:0 0 4px 0;}
.gp-n {display:inline-block; width:6px; height:1.05rem; align-self:center; border-radius:3px; background:var(--c);}
.gp-sub {font-weight:400; font-size:0.78rem; opacity:.7;}
.gp-m {display:flex; justify-content:space-between; align-items:baseline; padding:3px 0;
       border-bottom:1px dashed rgba(128,128,128,.25); font-size:0.88rem;}
.gp-m span:last-child {font-weight:700; font-size:1.0rem;}
.gp-sec {font-size:0.8rem; font-weight:700; opacity:.75; margin:6px 0 2px 0;}
.gp-note {font-size:0.74rem; opacity:.65; line-height:1.4; margin-top:6px;}
.gp-v {border-left:6px solid var(--c); border-radius:8px; padding:8px 12px;
       background:color-mix(in srgb, var(--c) 9%, transparent);}
.gp-vt {font-size:1.25rem; font-weight:800; color:var(--c);}
.gp-tag {display:inline-block; font-size:0.74rem; padding:1px 7px; margin:2px 3px 0 0; border-radius:5px;
         background:rgba(128,128,128,.15);}
.gp-doc {font-size:0.8rem; padding:2px 0;}
.gp-flow {display:flex; align-items:stretch; gap:6px; margin:0 0 12px 0;}
.gp-node {flex:1; box-sizing:border-box; border:1px solid rgba(128,128,128,.35); border-radius:8px; padding:5px 10px;}
.gp-node .k {font-size:0.72rem; opacity:.7;}
.gp-node .v {font-size:1.0rem; font-weight:800;}
.gp-node .s {font-size:0.72rem; opacity:.75; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.gp-arrow {align-self:center; font-size:1.2rem; opacity:.5;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="검색 인덱스와 모델을 불러오는 중")
def get_retriever(device: str) -> HybridRetriever:
    return HybridRetriever(device=device)


@st.cache_data
def session_table() -> pd.DataFrame:
    if not LABELS.exists():  # recorded captures (dataset/real) are not part of the repository
        return pd.DataFrame(columns=["session_id", "label", "declared_job_type"])
    lab = pd.read_csv(LABELS)
    lab = lab[lab["valid"].astype(str).str.lower().isin({"true", "1"}) & lab["gpu_role"].eq("target")]
    lab = lab[[(ROOT / "dataset/real/sessions" / s / "telemetry.csv").exists() for s in lab["session_id"]]]
    lab["label"] = lab["gt_variant"].fillna(lab["gt_label"]).astype(str)
    return lab.drop_duplicates("session_id")[["session_id", "label", "declared_job_type"]].sort_values("label")


# -- sidebar: same inputs as app.py ------------------------------------------
st.sidebar.title("GridTriage")
mode = st.sidebar.radio("입력", ["Replay (저장 세션)", "Live (NVML)"], index=0 if len(session_table()) else 1)
llm = st.sidebar.selectbox("판정 LLM", ["gemma3:12b", "deepseek-r1:14b", "claude-opus-5-5"])
retrieval = st.sidebar.selectbox("검색 방식", list(MODES), index=list(MODES).index("hybrid_rerank"))
k = st.sidebar.slider("LLM에 넘길 문서 수 k", 3, 12, 8)
host = st.sidebar.text_input("Ollama 주소", "http://127.0.0.1:11434")
devices = [default_device()] + [d for d in ("cuda:1", "cuda:0", "cpu") if d != default_device()]
pipe_device = st.sidebar.selectbox("파이프라인 장치 (검색 모델)", devices)
pipe_gpu = int(pipe_device.split(":")[1]) if pipe_device.startswith("cuda") else None

if mode.startswith("Replay") and not len(session_table()):
    st.sidebar.info("저장 세션이 없습니다. 실측 캡처(dataset/real)는 저장소에 포함되지 않습니다. "
                    "Live 모드를 쓰거나 dataset/run_capture로 세션을 수집하세요.")
    source_key = ("replay", None, None)
elif mode.startswith("Replay"):
    sessions = session_table()
    pick = st.sidebar.selectbox("세션", sessions.index,
                                format_func=lambda i: f"{sessions.at[i, 'label']} · {sessions.at[i, 'session_id']}")
    speed = st.sidebar.select_slider("재생 배속", [1, 5, 10, 20, 50], value=20)
    source_key = ("replay", sessions.at[pick, "session_id"], speed)
else:
    watched = st.sidebar.multiselect("감시할 GPU", [0, 1], default=[g for g in (0, 1) if g != pipe_gpu])
    if pipe_gpu in watched:
        st.sidebar.warning(f"GPU{pipe_gpu}는 파이프라인 장치이기도 해서 검색 모델 부하가 감시 데이터에 섞입니다.")
    declared = {}
    for gpu in watched:
        declared[gpu] = st.sidebar.selectbox(f"GPU{gpu} 신고 작업 (스케줄러)", list(DECLARED_POOL), index=1, key=f"d{gpu}")
    source_key = ("live", tuple(sorted(declared.items())))

with st.sidebar.expander("계통 연계 (물리 재생 · 계통 지표)", expanded=False):
    phys_on = st.checkbox("경보 구간을 테스트 계통에 재생", value=physics_bridge.available(),
                          disabled=not physics_bridge.available(),
                          help="ANDES 환경(.venv-physics)이 있어야 합니다. 경보 1건당 약 6–7초.")
    test_system = st.selectbox("테스트 계통", ["wecc", "kundur_ieeest"], format_func=SYSTEM_NAME.get)
    participation = st.select_slider("동기화 참여율", [0.1, 0.25, 0.5, 1.0], value=1.0)
    share_pct = st.select_slider("시설 비중 (계통 부하 대비 %)", [0.5, 1, 2, 5, 10], value=1)
    grid_input = st.radio("계통 상태 입력", ["시험 계통(시뮬레이션)", "수전점 계측 예시 파일(합성)", "수전점 계측 CSV 경로"],
                          help="수전점 전력품질 계측기·PMU 값을 CSV로 연결합니다.")
    grid_csv = ""
    if grid_input.startswith("수전점 계측 예시"):
        grid_csv = str(EXAMPLE_GRID)
    elif grid_input.startswith("수전점 계측 CSV"):
        grid_csv = st.text_input("CSV 경로 (t_epoch 또는 t_s, freq_hz, voltage_v)", "")
physics = physics_bridge.PhysicsSettings(enabled=phys_on, test_system=test_system, participation=participation,
                                         facility_share=share_pct / 100) if phys_on else None

c1, c2 = st.sidebar.columns(2)
if c1.button("시작", type="primary") and source_key[1]:
    old = st.session_state.get("engine")
    if old:
        old.stop()
    src = ReplaySource(source_key[1], speed=source_key[2]) if source_key[0] == "replay" else NvmlSource(dict(source_key[1]))
    engine = LiveEngine(src, get_retriever(pipe_device), EngineSettings(llm_model=llm, retrieval_mode=retrieval, k=k,
                                                                        ollama_host=host, physics=physics))
    engine.start()
    st.session_state["engine"] = engine
    st.session_state["grid"] = CsvGridSource(Path(grid_csv)) if grid_csv and Path(grid_csv).exists() else NoGridSource()
    st.session_state["seen_events"] = set()
if c2.button("정지") and st.session_state.get("engine"):
    st.session_state["engine"].stop()



# -- small HTML pieces -------------------------------------------------------
def panel_head(n: str, title: str, sub: str = "", color: str = BLUE) -> None:
    st.markdown(f"<div class='gp-h'><span class='gp-n' style='--c:{color}'></span>{title}"
                f"<span class='gp-sub'>{sub}</span></div>", unsafe_allow_html=True)


def tile(label: str, value: str, sub: str, color: str) -> str:
    return (f"<div class='gp-tile' style='--c:{color}'><div class='gp-tl'>{label}</div>"
            f"<div class='gp-tv'>{value}</div><div class='gp-ts'>{sub}</div></div>")


def metric(name: str, value: str) -> str:
    return f"<div class='gp-m'><span>{name}</span><span>{value}</span></div>"


def llm_icon(model: str, px: int = 20) -> str:
    if model.startswith("claude-"):
        return "✳️"
    name = "gemma-color" if model.startswith("gemma") else "deepseek-color" if model.startswith("deepseek") else "ollama"
    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        return "🤖"
    svg = path.read_text().replace('height="1em"', f'height="{px}"').replace('width="1em"', f'width="{px}"')
    return (f"<img src='data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}' width='{px}' "
            f"height='{px}' style='vertical-align:middle'>")


def latest_per_alarm(snap: dict) -> tuple[dict, dict]:
    """Newest judgment per alarm, and how many times each alarm was (re)analysed."""
    latest, count = {}, {}
    for res in snap["analyses"]:
        latest[res.get("alarm_id")] = res
        count[res.get("alarm_id")] = count.get(res.get("alarm_id"), 0) + 1
    return latest, count


def verdict_title(j: dict) -> str:
    v = j.get("verdict", "invalid")
    return VERDICT_TEXT.get(v, v) + (f" · {j['threat_card']}" if v == "attack" and j.get("threat_card") else "")


# -- status strip ------------------------------------------------------------
def status_bar(snap: dict, engine: LiveEngine | None) -> None:
    rows0 = next(iter(snap["rows"].values()), [])
    obs = observability(rows0)
    state = ("<b style='color:%s'>● 실행 중</b>" % GREEN) if engine and engine.running else (
        "● 정지" if engine else "● 대기")
    chips = [state]
    if snap["source"]:
        chips.append(f"입력 <b>{escape(str(snap['source']))}</b>")
    if obs:
        chips.append(f"판정 가능 대역 <b>≤ {obs['nyquist_hz']:.1f} Hz</b> · 갱신 {obs['refresh_ms']:.0f} ms")
    st.markdown("<div class='gp-bar'><span class='gp-title'>GridTriage Console</span>"
                + "".join(f"<span class='gp-chip'>{c}</span>" for c in chips) + "</div>", unsafe_allow_html=True)
    if snap["error"]:
        st.error(snap["error"])


def gpu_tile(gpu: int, snap: dict) -> str:
    """watching -> suspicion building -> alarm (analysing) -> verdict, as one tile."""
    trig = snap["trigger_windows"]
    run = snap["run_len"].get(gpu, 0)
    pending = [a for a, g in snap["pending_alarms"].items() if g == gpu]
    last = next((a for a in reversed(snap["analyses"]) if a["gpu_id"] == gpu), None)
    judged = {a.get("alarm_id") for a in snap["analyses"]}
    if pending and not judged & set(pending):
        return tile(f"GPU{gpu}", "🚨 경보 · 분석 중", f"의심 창 {run}개 연속", RED)
    if run >= trig and last is not None:
        v = last["judgment"].get("verdict", "invalid")
        return tile(f"GPU{gpu}", verdict_title(last["judgment"]), f"의심 창 {run}개 연속",
                    VERDICT_COLOR.get(v, GREY))
    if run > 0:
        return tile(f"GPU{gpu}", f"⚠ 의심 누적 {run}/{trig}", "의심 창 연속 발생", ORANGE)
    latest_w = next((w for w in reversed(snap["windows"]) if w["gpu_id"] == gpu), None)
    if latest_w and latest_w.get("grid_watch"):
        return tile(f"GPU{gpu}", "계통 주의", "통계상 정상 · 모드 대역 변동 큼", PURPLE)
    recent = f"최근 판정: {VERDICT_TEXT.get(last['judgment'].get('verdict'), '')}" if last else "정상 기준선 안"
    return tile(f"GPU{gpu}", "정상 감시 중", recent, GREEN)


def tiles_row(snap: dict, engine: LiveEngine | None) -> None:
    gpus = sorted(snap["rows"]) or [0, 1]
    latest, _ = latest_per_alarm(snap)
    n_attack = sum(r["judgment"].get("verdict") in ("attack", "unknown") for r in latest.values())
    busy = bool(snap["pending_alarms"])
    model = engine.settings.llm_model if engine else llm
    cards = [gpu_tile(g, snap) if snap["rows"] else tile(f"GPU{g}", "—", "데이터 대기", GREY) for g in gpus]
    cards.append(tile("세션 경보", f"{len(latest)}건", f"공격·미확인 {n_attack}건",
                      RED if n_attack else (ORANGE if latest else GREY)))
    cards.append(tile("Agent", "분석 중" if busy else "대기", model, ORANGE if busy else BLUE))
    phys = [p for p in snap["physics"].values() if p.get("status") == "done" and p.get("converged")]
    g = grid_reading(snap)
    if g:
        cards.append(tile("계통 상태", f"{g['swing_mhz']:.1f} mHz", f"주파수 변동 폭 · RoCoF {g['rocof_max_hz_s'] or 0:.3f} Hz/s", PURPLE))
    elif phys:
        mo = phys[-1].get("mode_overlap")
        cards.append(tile(f"계통 응답 · {SYSTEM_NAME.get(phys[-1].get('test_system'), '')}", f"RoCoF {phys[-1].get('rocof_hz_s') or 0:.3f} Hz/s",
                          f"모드 {mo['mode_hz']:.2f} Hz와 겹침" if mo else "고유 모드와 겹치지 않음", RED if mo else PURPLE))
    else:
        cards.append(tile("계통 상태", "—", "입력 없음", GREY))
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)


# -- panel 1: power ----------------------------------------------------------
def power_figure(rows: list[dict], windows: pd.DataFrame, gpu: int, height: int, legend: bool) -> go.Figure:
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=22, b=4), showlegend=legend,
                      legend=dict(orientation="h", y=1.18, x=1, xanchor="right", font=dict(size=10)),
                      title=dict(text=f"GPU{gpu}", x=0.01, y=0.97, font=dict(size=12)),
                      yaxis=dict(title=None, ticksuffix=" W"), xaxis=dict(title=None, ticksuffix="s"))
    if frame.empty:
        return fig
    t0 = frame["t_epoch"].min()
    x = frame["t_epoch"] - t0
    if frame["power_instant_w"].notna().any():
        fig.add_scatter(x=x, y=frame["power_instant_w"], name="순시 전력", line=dict(width=1, color="#f4a582"))
    fig.add_scatter(x=x, y=frame["power_w"], name="평균 전력 (1초)", line=dict(width=2, color=BLUE))
    if not windows.empty:
        for _, w in windows[(windows.gpu_id == gpu) & windows.is_candidate].iterrows():
            fig.add_vrect(x0=w.t_end - 30 - t0, x1=w.t_end - t0, fillcolor=RED, opacity=0.08, line_width=0)
        if "grid_watch" in windows:
            for _, w in windows[(windows.gpu_id == gpu) & windows.grid_watch.fillna(False).astype(bool)].iterrows():
                fig.add_vrect(x0=w.t_end - 30 - t0, x1=w.t_end - t0, fillcolor=PURPLE, opacity=0.10, line_width=0)
    return fig


def power_panel(snap: dict) -> None:
    with st.container(height=PANEL_H, border=True):
        panel_head("1", "GPU 전력", "빨강: 의심 창 · 보라: 계통 주의 창")
        gpus = sorted(snap["rows"])
        if not gpus:
            st.caption("데이터 없음")
            return
        windows = pd.DataFrame(snap["windows"])
        h = int((PANEL_H - 70) / len(gpus))
        for i, gpu in enumerate(gpus):
            st.plotly_chart(power_figure(snap["rows"][gpu], windows, gpu, h, legend=i == 0), width="stretch",
                            key=f"p{gpu}", config={"displayModeBar": False})


# -- panel 2: latest verdict -------------------------------------------------
def verdict_panel(snap: dict, model: str) -> None:
    with st.container(height=PANEL_H, border=True):
        panel_head("2", "최신 판정", model, RED)
        for gpu in sorted(set(snap["pending_alarms"].values())):
            st.markdown(f"{llm_icon(model, 16)} &nbsp;**GPU{gpu}** 경보 구간 분석 중…",
                        unsafe_allow_html=True)
        if not snap["analyses"]:
            st.caption("판정 없음")
            return
        res = snap["analyses"][-1]
        j = res["judgment"]
        color = VERDICT_COLOR.get(j.get("verdict"), GREY)
        expl = escape(j.get("explanation") or "; ".join(j.get("errors", [])))
        st.markdown(
            f"<div class='gp-v' style='--c:{color}'><div class='gp-vt'>{llm_icon(j.get('model') or model, 20)} "
            f"{escape(verdict_title(j))}</div><div style='font-size:0.8rem;opacity:.75'>GPU{res['gpu_id']} · 의심 창 "
            f"{res['n_windows']}개 · {j.get('latency_s', '')}s</div><div style='font-size:0.86rem;margin-top:4px'>{expl}</div>"
            "</div>", unsafe_allow_html=True)
        used = j.get("evidence_used") or []
        if used:
            st.markdown("<div class='gp-sec'>사용한 관측 증거</div>"
                        + "".join(f"<span class='gp-tag'>{escape(e)}</span>" for e in used), unsafe_allow_html=True)
        cited = set(j.get("cited_docs") or [])
        hits = res.get("hits") or []
        if hits:
            lines = "".join(
                f"<div class='gp-doc'>{'<b>●</b>' if h['doc_id'] in cited else '○'} {h['rank']}. "
                f"{'<b>' if h['doc_id'] in cited else ''}{escape(h['title'])}{'</b>' if h['doc_id'] in cited else ''}</div>"
                for h in hits[:5])
            st.markdown(f"<div class='gp-sec'>검색 문서 상위 5 / {len(hits)} (● 인용)</div>{lines}", unsafe_allow_html=True)
        ph = snap["physics"].get(res.get("alarm_id")) or {}
        if ph.get("status") == "running":
            st.markdown("<div class='gp-note'>계통 시험 재생 중…</div>", unsafe_allow_html=True)
        elif ph.get("converged"):
            mo = ph.get("mode_overlap")
            st.markdown(f"<div class='gp-note'>계통 시험 응답: RoCoF {ph.get('rocof_hz_s') or 0:.3f} Hz/s · "
                        + (f"고유 모드 {mo['mode_hz']:.2f} Hz와 겹침" if mo else "고유 모드와 겹치지 않음")
                        + "</div>", unsafe_allow_html=True)


# -- panel 3: grid coupling --------------------------------------------------
def grid_reading(snap: dict) -> dict | None:
    grid = st.session_state.get("grid") or NoGridSource()
    rows_all = [r for rs in snap["rows"].values() for r in rs]
    return grid.snapshot(max((r["t_epoch"] for r in rows_all), default=None)) if rows_all else None


def spectrum_figure(snap: dict, height: int) -> go.Figure:
    """Facility load spectrum against the system's electromechanical modes (and the measured grid, if connected)."""
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4), showlegend=True,
                      legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=9)),
                      xaxis=dict(range=[0.05, 2.0], ticksuffix=" Hz", tickfont=dict(size=9)),
                      yaxis=dict(ticksuffix="%", tickfont=dict(size=9), rangemode="tozero"),
                      yaxis2=dict(overlaying="y", side="right", ticksuffix=" mHz", tickfont=dict(size=9),
                                  showgrid=False, rangemode="tozero"))
    for m in physics_bridge.system_modes():
        low = (m.get("damping_ratio") or 1) < LOW_DAMPING
        fig.add_vline(x=m["freq_hz"], line_width=1.2 if low else 0.8, line_color=RED if low else GREY,
                      opacity=0.55 if low else 0.25)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=RED, width=1.2),
                    name=f"계통 모드 (감쇠 < {LOW_DAMPING:.0%})")
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=GREY, width=0.8), name="계통 모드")
    obs = observability(next(iter(snap["rows"].values()), []))
    nyq = obs.get("nyquist_hz") if obs else None
    if nyq and nyq == nyq and nyq < 2.0:  # NaN-safe
        fig.add_vrect(x0=nyq, x1=2.0, fillcolor=GREY, opacity=0.18, line_width=0,
                      annotation_text="관측 불가", annotation_position="top left", annotation_font_size=9)
    fs = facility_spectrum(snap["rows"], tdp_w=TDP_W) if snap["rows"] else {}
    if fs:
        amp = fs["amp_pu"] * 100
        fig.add_scatter(x=fs["freq_hz"], y=amp, name="시설 부하 (% TDP)", fill="tozeroy",
                        line=dict(color=BLUE, width=1.5))
        ok = fs["freq_hz"] <= (nyq if nyq and nyq == nyq else 2.0)
        if ok.any():
            i = int(amp[ok].argmax())
            peak_hz, peak = float(fs["freq_hz"][ok][i]), float(amp[ok][i])
            mo = physics_bridge.mode_overlap(peak_hz)
            fig.add_scatter(x=[peak_hz], y=[peak], mode="markers", showlegend=False,
                            marker=dict(size=9, color=RED if mo else BLUE, symbol="diamond"))
            label = f"{peak_hz:.2f} Hz"
            if mo:
                label += ("" if mo["harmonic"] == 1 else f" ({mo['harmonic']}차 고조파)") + f" · 모드 {mo['mode_hz']:.2f} Hz와 겹침"
            fig.add_annotation(x=peak_hz, y=peak, yshift=12, showarrow=False, text=label,
                               font=dict(size=9, color=RED if mo else None))
    grid = st.session_state.get("grid") or NoGridSource()
    rows_all = [r for rs in snap["rows"].values() for r in rs]
    gs = grid.spectrum(max(r["t_epoch"] for r in rows_all)) if rows_all else None
    if gs:
        fig.add_scatter(x=gs["freq_hz"], y=gs["amp_mhz"], name="수전점 주파수 (mHz)", yaxis="y2",
                        line=dict(color=PURPLE, width=1.3))
    return fig


def grid_panel(snap: dict) -> None:
    with st.container(height=PANEL_H, border=True):
        panel_head("3", "계통 연계", "", PURPLE)
        g = grid_reading(snap)
        if g:  # a point-of-connection feed (PQ meter / PMU export) replaces the simulated values
            st.markdown("<div class='gp-sec'>계통 상태 · 수전점 계측 (60초)</div>", unsafe_allow_html=True)
            osc = f"{g['osc_freq_hz']:.2f} Hz" if g.get("osc_freq_hz") is not None else f"계산 불가 ({g['sample_hz']:.1f}/s)"
            st.markdown(metric("주파수 · 변동 폭", f"{g['freq_hz']:.3f} Hz · {g['swing_mhz']:.1f} mHz")
                        + metric("RoCoF", f"{g['rocof_max_hz_s'] or 0:.3f} Hz/s")
                        + metric("진동 주파수", osc)
                        + (metric("수전점 전압", f"{g['voltage_v']:,.0f} V") if "voltage_v" in g else ""),
                        unsafe_allow_html=True)
        else:
            done = [p for p in snap["physics"].values() if p.get("status") == "done" and p.get("converged")]
            name = SYSTEM_NAME.get(done[-1].get("test_system"), "") if done else ""
            st.markdown(f"<div class='gp-sec'>계통 상태 · 시험 계통 {name}</div>", unsafe_allow_html=True)
            if done:
                ph = done[-1]
                st.markdown(metric("주파수 변동 폭 (경보 구간)", f"{(ph.get('osc_ptp') or 0) * 60.0 * 1000:.1f} mHz")
                            + metric("RoCoF", f"{ph.get('rocof_hz_s') or 0:.3f} Hz/s")
                            + metric("진동 주파수", f"{ph.get('dominant_freq_hz') or 0:.2f} Hz"), unsafe_allow_html=True)
            else:
                st.caption("입력 없음")
        st.markdown("<div class='gp-sec'>부하 스펙트럼 · 계통 고유 모드 (최근 120초)</div>", unsafe_allow_html=True)
        st.plotly_chart(spectrum_figure(snap, 215), width="stretch", key="spectrum", config={"displayModeBar": False})


# -- scale chain: GPU -> facility -> share of system load -> grid response --------
def scale_flow(snap: dict, engine: LiveEngine | None) -> None:
    ps = engine.settings.physics if engine and engine.settings.physics else None
    part = ps.participation if ps else participation
    share = ps.facility_share if ps else share_pct / 100
    last = {w["gpu_id"]: w for w in snap["windows"]}
    swings = [float(w["swing_abs_w"]) for w in last.values() if pd.notna(w.get("swing_abs_w"))]
    ff = facility_forcing(snap["rows"], tdp_w=TDP_W) if snap["rows"] else {}
    nodes = [("GPU 전력 변동 (최근 창 최대)", f"{max(swings):.0f} W" if swings else "—",
              f"TDP의 {max(swings) / TDP_W:.0%}" if swings else "")]
    if ff:
        corr = "—" if ff["gpu_band_corr"] is None else f"{ff['gpu_band_corr']:.2f}"
        nodes.append(("시설 합산 · 0.1–2 Hz 변동", f"{ff['band_amp_pu']:.1%} TDP",
                      f"GPU {ff['n_gpus']}장 동기화 {corr} · 램프 {ff['max_ramp_pu_per_s']:.1%}/s"))
        nodes.append(("계통 부하 대비 (추정)", f"{ff['band_amp_pu'] * part * share:.3%}",
                      f"× 참여율 {part:g} × 시설 비중 {share:.1%}"))
    else:
        nodes += [("시설 합산 · 0.1–2 Hz 변동", "—", "데이터 수집 중"), ("계통 부하 대비 (추정)", "—", "")]
    g = grid_reading(snap)
    done = [p for p in snap["physics"].values() if p.get("status") == "done" and p.get("converged")]
    if g:
        nodes.append(("계통 응답 · 수전점", f"{g['swing_mhz']:.1f} mHz", f"RoCoF {g['rocof_max_hz_s'] or 0:.3f} Hz/s"))
    elif done:
        ph = done[-1]
        nodes.append((f"계통 응답 · {SYSTEM_NAME.get(ph.get('test_system'), '')}",
                      f"{(ph.get('osc_ptp') or 0) * 60.0 * 1000:.1f} mHz", f"RoCoF {ph.get('rocof_hz_s') or 0:.3f} Hz/s"))
    else:
        nodes.append(("계통 응답", "—", "입력 없음"))
    html = "<span class='gp-arrow'>→</span>".join(
        f"<div class='gp-node'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{sub}</div></div>"
        for k, v, sub in nodes)
    st.markdown(f"<div class='gp-flow'>{html}</div>", unsafe_allow_html=True)


# -- load -> grid, on one time axis -------------------------------------------
def aligned_figure(snap: dict) -> go.Figure | None:
    """Facility load (top) and grid frequency deviation (bottom) on the same data clock.

    Bottom: the point-of-connection feed when connected, plus each alarm's test-system replay
    placed at the data time of the waveform it replayed.
    """
    rows_all = [r for rs in snap["rows"].values() for r in rs]
    fs = facility_series(snap["rows"]) if snap["rows"] else {}
    if not rows_all or not fs:
        return None
    t0 = min(r["t_epoch"] for r in rows_all)
    t_now = max(r["t_epoch"] for r in rows_all)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("시설 합산 전력", "계통 주파수 편차"))
    x = fs["t_epoch"] - t0
    fig.add_scatter(x=x, y=fs["total_w"], name="합산 전력", line=dict(color=BLUE, width=1.2), row=1, col=1)
    fig.add_scatter(x=x, y=fs["band_w"] + fs["total_w"].mean(), name="0.1–2 Hz 성분 (+평균)",
                    line=dict(color=ORANGE, width=1.2), row=1, col=1)
    grid = st.session_state.get("grid") or NoGridSource()
    gs = grid.series(t_now, t_now - t0)
    if gs:
        fig.add_scatter(x=gs["t_epoch"] - t0, y=gs["freq_dev_mhz"], name="수전점 계측",
                        line=dict(color=PURPLE, width=1.2), row=2, col=1)
    shown = False
    for aid, ph in snap["physics"].items():
        tr = ph.get("omega_trace") if ph.get("status") == "done" and ph.get("converged") else None
        if not tr or ph.get("data_t0") is None:
            continue
        start = ph.get("waveform_t_start_s") or tr["t_s"][0]
        fig.add_scatter(x=[ph["data_t0"] + t - start - t0 for t in tr["t_s"]], y=tr["freq_dev_mhz"],
                        name=f"시험 계통 응답 ({SYSTEM_NAME.get(ph.get('test_system'), '')})", legendgroup="sim",
                        showlegend=not shown, line=dict(color=RED, width=1.2, dash="dot"), row=2, col=1)
        shown = True
    runs: dict[str, list[float]] = {}
    for res in snap["analyses"]:
        runs.setdefault(res["alarm_id"], []).append(res["t_end"])
    for aid, ends in runs.items():  # alarmed run: first window start .. last analysed window end
        fig.add_vrect(x0=alarm_time(aid) - 30 - t0, x1=max(ends) - t0, fillcolor=RED, opacity=0.07, line_width=0,
                      row="all", col=1)
    fig.update_layout(height=420, margin=dict(l=4, r=4, t=26, b=4),
                      legend=dict(orientation="h", y=-0.12, font=dict(size=10)))
    fig.update_yaxes(ticksuffix=" W", row=1, col=1)
    fig.update_yaxes(ticksuffix=" mHz", row=2, col=1)
    fig.update_xaxes(ticksuffix="s", range=[0, t_now - t0], row=2, col=1)
    fig.update_annotations(font_size=11)
    if not gs and not shown:
        fig.add_annotation(text="계통 입력 없음", xref="x2 domain", yref="y2 domain", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=GREY))
    return fig


# -- cyber / grid event timeline ---------------------------------------------
def alarm_time(aid: str) -> float:
    return float(aid.split(":", 1)[1])


def timeline_figure(snap: dict) -> go.Figure | None:
    rows_all = [r for rs in snap["rows"].values() for r in rs]
    if not rows_all:
        return None
    t0 = min(r["t_epoch"] for r in rows_all)
    t_now = max(r["t_epoch"] for r in rows_all) - t0
    fig = go.Figure()
    gpus = sorted(snap["rows"])
    lanes = [f"GPU{g} 의심 창" for g in gpus] + ["계통 주의 창", "세션 경보", "LLM 판정", "계통 재생"]
    cand = [w for w in snap["windows"] if w["is_candidate"]]
    watch = [w for w in snap["windows"] if w.get("grid_watch")]
    if watch:  # statistically ordinary but high electromechanical-band impact: kept apart from security candidates
        fig.add_scatter(x=[w["t_end"] - t0 for w in watch], y=["계통 주의 창"] * len(watch), mode="markers",
                        marker=dict(symbol="line-ns-open", size=12, color=PURPLE), showlegend=False,
                        hovertext=[f"GPU{w['gpu_id']} · impact {w.get('impact_level')}" for w in watch],
                        hoverinfo="x+text")
    for g in gpus:
        ws = [w for w in cand if w["gpu_id"] == g]
        if ws:
            fig.add_scatter(x=[w["t_end"] - t0 for w in ws], y=[f"GPU{g} 의심 창"] * len(ws), mode="markers",
                            marker=dict(symbol="line-ns-open", size=12, color=ORANGE), showlegend=False,
                            hovertext=[f"p={w['p_value']:.3f}" for w in ws], hoverinfo="x+text")
    alarms = {}
    for res in snap["analyses"]:
        alarms.setdefault(res["alarm_id"], res)
    for aid, gpu in snap["pending_alarms"].items():
        alarms.setdefault(aid, {"gpu_id": gpu, "t_end": None})
    if alarms:  # an alarm id carries the end time of the run's first window; the alarm fires at the gate
        xs = [(a["t_end"] if a.get("t_end") is not None else t0 + t_now) - t0 for a in alarms.values()]
        fig.add_scatter(x=xs, y=["세션 경보"] * len(xs), mode="markers", showlegend=False,
                        marker=dict(symbol="triangle-up", size=13, color=RED),
                        hovertext=[f"GPU{a['gpu_id']}" for a in alarms.values()], hoverinfo="x+text")
    for res in snap["analyses"]:
        j = res["judgment"]
        fig.add_scatter(x=[res["t_end"] - t0], y=["LLM 판정"], mode="markers", showlegend=False,
                        marker=dict(size=11, color=VERDICT_COLOR.get(j.get("verdict"), GREY)),
                        hovertext=f"GPU{res['gpu_id']} {verdict_title(j)}", hoverinfo="x+text")
    for aid, ph in snap["physics"].items():
        if ph.get("status") != "done":
            continue
        first = next((a for a in snap["analyses"] if a["alarm_id"] == aid), None)
        x = (first["t_end"] if first else alarm_time(aid)) - t0
        mo = ph.get("mode_overlap")
        fig.add_scatter(x=[x], y=["계통 재생"], mode="markers", showlegend=False,
                        marker=dict(symbol="square", size=11, color=RED if mo else PURPLE),
                        hovertext=(f"RoCoF {ph.get('rocof_hz_s') or 0:.3f} Hz/s"
                                   + (f" · 모드 {mo['mode_hz']:.2f} Hz 겹침" if mo else "")
                                   if ph.get("converged") else "재생 실패"), hoverinfo="x+text")
    fig.update_layout(height=60 + 34 * len(lanes), margin=dict(l=4, r=4, t=8, b=4),
                      xaxis=dict(range=[0, max(t_now, 1)], ticksuffix="s"),
                      yaxis=dict(categoryorder="array", categoryarray=lanes[::-1], type="category"))
    return fig


# -- below the fold ----------------------------------------------------------
def physics_detail(ph: dict | None) -> None:
    if not ph:
        st.caption("결과 없음")
        return
    if ph.get("status") == "running":
        st.caption("물리 재생 중 (테스트 계통)…")
        return
    s = ph.get("settings") or {}
    name = SYSTEM_NAME.get(ph.get("test_system"), ph.get("test_system"))
    if not ph.get("converged"):
        st.caption(f"물리 재생 실패 ({name}): {ph.get('failure_reason')}")
        return
    sc = ph.get("scaling") or {}
    mo = ph.get("mode_overlap")
    overlap = (f"계통 고유 모드 {mo['mode_hz']:.2f} Hz(감쇠비 {mo['damping_ratio']:.1%})와 "
               f"{'기본파' if mo['harmonic'] == 1 else str(mo['harmonic']) + '차 고조파'}가 겹침"
               if mo else "계통 고유 모드(0.27–1.88 Hz)와 겹치지 않음")
    st.markdown(
        f"**{name}** · 참여율 {s.get('participation')} · 시설 비중 {s.get('facility_share', 0):.1%}  \n"
        f"주입 진폭 계통 부하의 {sc.get('injected_frac_of_system_load', 0):.2%}"
        f"{' (모선 한도로 잘림)' if sc.get('clipped') else ''} · 회전속도 진동 {ph.get('osc_std', 0):.2e} p.u. · "
        f"RoCoF {ph.get('rocof_hz_s') or 0:.3f} Hz/s · 응답 주파수 {ph.get('dominant_freq_hz') or 0:.2f} Hz  \n"
        f"부하 변동 {ph.get('forcing_freq_hz') or 0:.2f} Hz → {overlap}")


def details(snap: dict, model: str) -> None:
    st.markdown("<div class='gp-h' style='margin-top:14px'><span class='gp-n' style='--c:#555'></span>상세</div>",
                unsafe_allow_html=True)
    t_hist, t_time, t_win, t_phys = st.tabs(["판정 이력과 근거", "계통 흐름 · 이벤트", "창 분석", "물리 응답"])
    latest, count = latest_per_alarm(snap)
    with t_hist:
        if not latest:
            st.caption("판정 없음")
        for aid, res in list(latest.items())[::-1]:
            j = res["judgment"]
            color = VERDICT_COLOR.get(j.get("verdict"), GREY)
            again = f" · 재분석 {count[aid] - 1}회" if count[aid] > 1 else ""
            st.markdown(
                f"<div class='gp-v' style='--c:{color};margin-top:6px'><b style='color:{color}'>"
                f"{escape(verdict_title(j))}</b><span style='font-size:0.8rem;opacity:.7'> · GPU{res['gpu_id']} · 의심 창 "
                f"{res['n_windows']}개 · {j.get('latency_s', '')}s{again}</span><br>"
                f"<span style='font-size:0.86rem'>{escape(j.get('explanation') or '; '.join(j.get('errors', [])))}</span></div>",
                unsafe_allow_html=True)
            with st.expander("근거: 질의 · 검색 문서 · 증거"):
                st.code(res.get("query") or "", language=None)
                cited = set(j.get("cited_docs") or [])
                hits = pd.DataFrame(res["hits"])
                if not hits.empty:
                    hits["인용"] = hits["doc_id"].isin(cited).map({True: "●", False: ""})
                    st.dataframe(hits[["rank", "인용", "score", "layer", "title"]], hide_index=True, width="stretch")
                st.json({"evidence_used": j.get("evidence_used"),
                         "evidence_true": [k for k, v in res["bundle"]["unexplainedness"]["evidence"].items() if v],
                         "pattern": res["bundle"].get("pattern")}, expanded=False)
    with t_time:
        aligned, events = aligned_figure(snap), timeline_figure(snap)
        if aligned is None and events is None:
            st.caption("데이터 없음")
        if aligned is not None:
            st.plotly_chart(aligned, width="stretch", key="aligned", config={"displayModeBar": False})
        if events is not None:
            st.plotly_chart(events, width="stretch", key="timeline", config={"displayModeBar": False})
    with t_win:
        windows = pd.DataFrame(snap["windows"])
        if windows.empty:
            st.caption("데이터 없음")
        else:
            view = windows.tail(40).iloc[::-1][["gpu_id", "p_value", "is_candidate", "mean_w", "swing_abs_w",
                                                 "dominant_freq_hz", "grid_watch", "active_evidence"]]
            view["active_evidence"] = view["active_evidence"].map(lambda v: ", ".join(v or []))
            st.dataframe(view.round(3), hide_index=True, width="stretch", height=360)
    with t_phys:
        if not snap["physics"]:
            st.caption("결과 없음")
        for aid, ph in list(snap["physics"].items())[::-1]:
            st.markdown(f"**경보 {aid}**")
            physics_detail(ph)


def notify(snap: dict) -> None:
    """Toast once per new alarm and once per new verdict."""
    seen = st.session_state.setdefault("seen_events", set())
    for aid, gpu in snap["pending_alarms"].items():
        if ("alarm", aid) not in seen:
            seen.add(("alarm", aid))
            st.toast(f"GPU{gpu} 이상 경보 — 자동 분석을 시작합니다", icon="🚨")
    for a in snap["analyses"]:
        key = ("verdict", a.get("alarm_id"), a["t_end"])
        if key not in seen:
            seen.add(key)
            st.toast(f"GPU{a['gpu_id']} {verdict_title(a['judgment'])}", icon="🔎")


@st.fragment(run_every=1.0)
def dashboard() -> None:
    engine: LiveEngine | None = st.session_state.get("engine")
    snap = engine.snapshot() if engine else EMPTY
    model = engine.settings.llm_model if engine else llm
    status_bar(snap, engine)
    tiles_row(snap, engine)
    scale_flow(snap, engine)
    notify(snap)
    left, mid, right = st.columns([5, 4, 3], gap="small")
    with left:
        power_panel(snap)
    with mid:
        verdict_panel(snap, model)
    with right:
        grid_panel(snap)
    details(snap, model)


dashboard()
