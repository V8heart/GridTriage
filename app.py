"""GridTriage console: real-time view of the v4 pipeline.

    .venv/bin/streamlit run app.py

Live mode polls NVML on this server; Replay mode streams a recorded session.
The page only renders what pipeline/live.py computes (no analysis logic here)
and has no workload-control buttons: output is triage priority, not enforcement.
Physics: each session alarm is replayed once on a public test system (Kundur / WECC)
at a chosen facility scale. That is a sensitivity check, not the present grid state.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset.declared_context import DECLARED_POOL
from pipeline import physics_bridge
from pipeline.grid_monitor import CsvGridSource, NoGridSource, facility_forcing
from pipeline.kb.retriever import MODES, HybridRetriever, default_device
from pipeline.live import EngineSettings, LiveEngine, NvmlSource, ReplaySource, observability

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "dataset" / "real" / "index" / "labels.csv"
EXAMPLE_GRID = ROOT / "examples" / "grid" / "pcc_measurement_example.csv"
VERDICT_COLOR = {"attack": "#d62728", "normal": "#2ca02c", "unobservable": "#7f7f7f",
                 "unknown": "#ff7f0e", "invalid": "#9467bd"}

st.set_page_config(page_title="GridTriage Console", layout="wide")


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


# -- sidebar: source and settings ------------------------------------------
st.sidebar.title("GridTriage")
mode = st.sidebar.radio("입력", ["Replay (저장 세션)", "Live (NVML)"], index=0 if len(session_table()) else 1)
llm = st.sidebar.selectbox("판정 LLM", ["gemma3:12b", "deepseek-r1:14b"])
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
    pick = st.sidebar.selectbox("세션 (라벨은 평가용 표기)", sessions.index,
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
    test_system = st.selectbox("테스트 계통", ["wecc", "kundur_ieeest"],
                               format_func={"wecc": "WECC 179-bus", "kundur_ieeest": "Kundur 2-area"}.get)
    participation = st.select_slider("동기화 참여율", [0.1, 0.25, 0.5, 1.0], value=1.0)
    share_pct = st.select_slider("시설 비중 (계통 부하 대비 %)", [0.5, 1, 2, 5, 10], value=1)
    grid_input = st.radio("계통 상태 입력", ["시험 계통(시뮬레이션)", "수전점 계측 예시 파일(합성)", "수전점 계측 CSV 경로"],
                          help="운영 시에는 수전점 전력품질 계측기·PMU 값을 연결합니다. 예시 파일은 형식 시연용 합성 데이터입니다.")
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
    st.session_state["truth"] = getattr(src, "truth", None)
if c2.button("정지") and st.session_state.get("engine"):
    st.session_state["engine"].stop()

st.sidebar.caption("조작(종료·차단) 기능은 없습니다. 출력은 조사 우선순위와 근거입니다.")


# -- main view ---------------------------------------------------------------
def power_figure(rows: list[dict], windows: pd.DataFrame, gpu: int) -> go.Figure:
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    if frame.empty:
        return fig
    t0 = frame["t_epoch"].min()
    x = frame["t_epoch"] - t0
    if frame["power_instant_w"].notna().any():
        fig.add_scatter(x=x, y=frame["power_instant_w"], name="순시 전력", line=dict(width=1, color="#f4a582"))
    fig.add_scatter(x=x, y=frame["power_w"], name="평균 전력 (1초)", line=dict(width=2, color="#2166ac"))
    if not windows.empty:
        for _, w in windows[(windows.gpu_id == gpu) & windows.is_candidate].iterrows():
            fig.add_vrect(x0=w.t_end - 30 - t0, x1=w.t_end - t0, fillcolor="#d62728", opacity=0.08, line_width=0)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title=f"GPU{gpu}",
                      xaxis_title="초", yaxis_title="W", legend=dict(orientation="h", y=1.15))
    return fig


VERDICT_TEXT = {"attack": "공격 의심", "unknown": "미확인 이상 (등록 위협과 불일치)", "normal": "경보 후 판정: 정상",
                "unobservable": "관측 대역 밖 (외부 계측 필요)", "invalid": "판정 실패"}


def _box(color: str, title: str, body: str = "") -> str:
    return (f"<div style='border-left:6px solid {color};background:{color}1a;padding:8px 12px;border-radius:6px;"
            f"margin-bottom:6px'><b style='color:{color}'>{title}</b><br><span style='font-size:0.92em'>{body}</span></div>")


def gpu_status(gpu: int, snap: dict) -> str:
    """One monitoring card per GPU: watching -> suspicion building -> alarm (analysing) -> verdict."""
    trig = snap["trigger_windows"]
    run = snap["run_len"].get(gpu, 0)
    pending = [a for a, g in snap["pending_alarms"].items() if g == gpu]
    last = next((a for a in reversed(snap["analyses"]) if a["gpu_id"] == gpu), None)
    judged = {a.get("alarm_id") for a in snap["analyses"]}
    if pending and not judged & set(pending):  # first judgment of this alarm still running
        return _box("#d62728", f"🚨 GPU{gpu} 이상 경보 — 자동 분석 중…",
                    f"의심 창 {run}개 연속(약 {15 * run + 15}초). 위협 지식 검색 → LLM 판정을 진행하고 있습니다.")
    if run >= trig and last is not None:
        j = last["judgment"]
        v = j.get("verdict", "invalid")
        color = VERDICT_COLOR.get(v, "#333")
        head = f"{'🚨' if v in ('attack', 'unknown') else '✅'} GPU{gpu} {VERDICT_TEXT.get(v, v)}"
        head += f": {j.get('threat_card')}" if j.get("threat_card") and v == "attack" else ""
        expl = (j.get("explanation") or "; ".join(j.get("errors", [])))
        expl = expl.split(". ")[0][:220]
        ph = snap["physics"].get(last.get("alarm_id")) or {}
        if ph.get("status") == "done" and ph.get("converged"):
            mo = ph.get("mode_overlap")
            expl += (f"<br>계통 시험 응답: RoCoF {ph.get('rocof_hz_s') or 0:.2f} Hz/s"
                     + (f", 고유 모드 {mo['mode_hz']:.2f} Hz와 겹침" if mo else ""))
        elif ph.get("status") == "running":
            expl += "<br>계통 시험 재생 중…"
        return _box(color, head, f"의심 창 {run}개 연속 · {expl}")
    if run > 0:
        return _box("#ff7f0e", f"⚠️ GPU{gpu} 의심 창 누적 {run}/{trig}",
                    f"정상 기준선에서 벗어난 창이 이어지고 있습니다. {trig}개 연속이면 자동 분석합니다.")
    recent = f" · 최근 판정: {VERDICT_TEXT.get(last['judgment'].get('verdict'), '')}" if last else ""
    return _box("#2ca02c", f"🟢 GPU{gpu} 정상 감시 중", f"최근 창이 정상 기준선 안에 있습니다{recent}.")


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
            j = a["judgment"]
            st.toast(f"GPU{a['gpu_id']} {VERDICT_TEXT.get(j.get('verdict'), j.get('verdict'))} "
                     f"{j.get('threat_card') or ''}", icon="🔎")


def grid_panel(snap: dict) -> None:
    st.subheader("계통 연계")
    left, right = st.columns(2)
    with left:
        st.markdown("**우리 부하가 계통에 주는 자극** (NVML, 최근 120초 합산)")
        ff = facility_forcing(snap["rows"])
        if ff:
            corr = "—" if ff["gpu_band_corr"] is None else f"{ff['gpu_band_corr']:.2f}"
            st.markdown(
                f"- 모드 대역(0.1–2 Hz) 변동 진폭 **{ff['band_amp_pu']:.1%}** (GPU {ff['n_gpus']}장 TDP 대비), "
                f"주 성분 {ff['band_peak_hz']:.2f} Hz\n"
                f"- 최대 1초 램프 **{ff['max_ramp_pu_per_s']:.1%}/s**\n"
                f"- GPU 간 대역 동기화(상관) **{corr}** — 높을수록 시설 규모에서 변동이 더해짐")
        else:
            st.caption("20초 이상 데이터가 쌓이면 표시합니다.")
    with right:
        grid = st.session_state.get("grid") or NoGridSource()
        rows_all = [r for rs in snap["rows"].values() for r in rs]
        g = grid.snapshot(max((r["t_epoch"] for r in rows_all), default=None))
        if g:  # a point-of-connection feed (PQ meter / PMU export) replaces the simulated values
            example = "pcc_measurement_example" in grid.label
            st.markdown(f"**계통 상태** · {grid.label}" + (" (형식 시연용 합성 예시)" if example else ""))
            osc = (f"진동 주파수 {g['osc_freq_hz']:.2f} Hz" if g.get("osc_freq_hz") is not None
                   else f"진동 주파수 계산 불가(샘플 {g['sample_hz']:.1f}/s, 4/s 이상 필요)")
            st.markdown(
                f"- 주파수 변동 폭 **{g['swing_mhz']:.1f} mHz** (최근 60초, 현재 {g['freq_hz']:.3f} Hz)\n"
                f"- RoCoF **{g['rocof_max_hz_s'] or 0:.3f} Hz/s** · {osc}\n"
                + (f"- 수전점 전압 {g['voltage_v']:,.0f} V\n" if "voltage_v" in g else "")
                + "- 계통 고유 모드 겹침: 계통 운영 측 모드 정보 연계 시 표시")
            return
        st.markdown("**계통 상태** · 시연: 공개 시험 계통(WECC/Kundur)으로 대체")
        done = [p for p in snap["physics"].values() if p.get("status") == "done" and p.get("converged")]
        if done:
            ph = done[-1]
            nominal = 60.0  # both ANDES cases are 60 Hz systems
            mo = ph.get("mode_overlap")
            st.markdown(
                f"- 주파수 변동 폭 **{(ph.get('osc_ptp') or 0) * nominal * 1000:.1f} mHz** (최근 경보 구간)\n"
                f"- RoCoF **{ph.get('rocof_hz_s') or 0:.3f} Hz/s** · 진동 주파수 {ph.get('dominant_freq_hz') or 0:.2f} Hz\n"
                + (f"- 계통 고유 모드 {mo['mode_hz']:.2f} Hz(감쇠비 {mo['damping_ratio']:.1%})와 부하 변동이 겹침"
                   if mo else "- 계통 고유 모드와 겹치지 않음"))
        else:
            st.caption("경보가 나면 해당 구간의 부하를 계통 모델에 연결해 계통 상태를 계산합니다.")
        st.caption("실제 운영에서는 이 자리에 시설 접속점(UPS·PCC)의 주파수·RoCoF·전압이 표시됩니다. "
                   "지금 값은 경보 구간 부하를 공개 시험 계통에 연결해 계산한 시연 값입니다.")


def physics_block(ph: dict | None) -> None:
    if not ph:
        return
    if ph.get("status") == "running":
        st.caption("물리 재생 중 (테스트 계통)…")
        return
    s = ph.get("settings") or {}
    name = {"wecc": "WECC 179-bus", "kundur_ieeest": "Kundur 2-area"}.get(ph.get("test_system"), ph.get("test_system"))
    if not ph.get("converged"):
        st.caption(f"물리 재생 실패 ({name}): {ph.get('failure_reason')}")
        return
    sc = ph.get("scaling") or {}
    mo = ph.get("mode_overlap")
    overlap = (f"계통 고유 모드 {mo['mode_hz']:.2f} Hz(감쇠비 {mo['damping_ratio']:.1%})와 "
               f"{'기본파' if mo['harmonic'] == 1 else str(mo['harmonic']) + '차 고조파'}가 겹침"
               if mo else "계통 고유 모드(0.27–1.88 Hz)와 겹치지 않음")
    st.markdown(
        f"**물리 응답** · {name} · 참여율 {s.get('participation')} · 시설 비중 {s.get('facility_share', 0):.1%}  \n"
        f"주입 진폭 계통 부하의 {sc.get('injected_frac_of_system_load', 0):.2%}"
        f"{' (모선 한도로 잘림)' if sc.get('clipped') else ''} · 회전속도 진동 {ph.get('osc_std', 0):.2e} p.u. · "
        f"RoCoF {ph.get('rocof_hz_s') or 0:.3f} Hz/s · 응답 주파수 {ph.get('dominant_freq_hz') or 0:.2f} Hz  \n"
        f"부하 변동 {ph.get('forcing_freq_hz') or 0:.2f} Hz → {overlap}")
    st.caption("공개 테스트 계통의 민감도 응답이며 현재 실제 계통 상태나 피해 규모가 아닙니다.")


ICON_DIR = ROOT / "assets" / "llm_icons"  # LobeHub icons (MIT), identify the model only


def llm_icon(model: str, px: int = 22) -> str:
    name = "gemma-color" if model.startswith("gemma") else "deepseek-color" if model.startswith("deepseek") else "ollama"
    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        return "🤖"
    svg = path.read_text().replace('height="1em"', f'height="{px}"').replace('width="1em"', f'width="{px}"')
    return f"<img src='data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}' width='{px}' height='{px}' " \
           "style='vertical-align:middle'>"


def agent_panel(snap: dict, model: str) -> None:
    """The LLM analyst's replies, newest first, under an agent header with the model's icon."""
    busy = bool(snap["pending_alarms"])
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:10px'>{llm_icon(model)}"
        f"<span style='font-size:1.15em;font-weight:700'>Agent</span>"
        f"<span style='color:#777;font-size:0.85em'>{model} · 위협 지식 검색 → 판정</span>"
        f"<span style='margin-left:auto;font-size:0.85em;color:{'#d62728' if busy else '#2ca02c'}'>"
        f"{'● 분석 중' if busy else '● 대기'}</span></div>", unsafe_allow_html=True)
    box = st.container(height=520, border=True)
    with box:
        for gpu in sorted(set(snap["pending_alarms"].values())):
            st.markdown(f"{llm_icon(model, 18)} &nbsp;**GPU{gpu}** 경보 구간을 분석하고 있습니다… "
                        "(관측 증거 → 위협 지식 검색 → 판정)", unsafe_allow_html=True)
        if not snap["analyses"] and not busy:
            st.caption(f"아직 경보가 없습니다. 의심 창이 {snap['trigger_windows']}개 연속(데이터 기준 약 "
                       f"{15 * snap['trigger_windows'] + 15}초)이면 Agent가 자동으로 분석합니다.")
        latest, count = {}, {}
        for res in snap["analyses"]:  # one reply per alarm: the newest judgment, with how often it was re-checked
            latest[res.get("alarm_id")] = res
            count[res.get("alarm_id")] = count.get(res.get("alarm_id"), 0) + 1
        for aid, res in list(latest.items())[::-1][:8]:
            j = res["judgment"]
            verdict = j.get("verdict", "?")
            color = VERDICT_COLOR.get(verdict, "#333")
            title = VERDICT_TEXT.get(verdict, verdict) + (f": {j['threat_card']}" if verdict == "attack" and j.get("threat_card") else "")
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:4px 10px;margin:8px 0 2px 0'>"
                f"{llm_icon(j.get('model') or model, 18)} &nbsp;<b style='color:{color}'>{title}</b>"
                f"<span style='color:#777;font-size:0.82em'> · GPU{res['gpu_id']} · 의심 창 {res['n_windows']}개 · "
                f"{j.get('latency_s', '')}s{f' · 재분석 {count[aid] - 1}회' if count[aid] > 1 else ''}</span><br>{j.get('explanation') or '; '.join(j.get('errors', []))}</div>",
                unsafe_allow_html=True)
            physics_block(snap["physics"].get(res.get("alarm_id")))
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


@st.fragment(run_every=1.0)
def dashboard() -> None:
    engine: LiveEngine | None = st.session_state.get("engine")
    if engine is None:
        st.info("왼쪽에서 입력을 고르고 '시작'을 누르세요.")
        return
    snap = engine.snapshot()
    if snap["error"]:
        st.error(snap["error"])
    rows0 = next(iter(snap["rows"].values()), [])
    obs = observability(rows0)
    if obs:
        st.markdown(
            f"**관측 경계** · 폴링 {obs['poll_ms']:.0f} ms · 실효 갱신 {obs['refresh_ms']:.0f} ms · "
            f"값 변경률 {obs['change_ratio']:.0%} · 판정 가능 ≤ {obs['nyquist_hz']:.1f} Hz "
            f"(그보다 빠른 변조는 '판정 불가, 외부 계측 필요')")
    st.caption(f"{snap['source']} · Stage1 {snap['stage1']} · 진행 중 판정 {snap['pending']}건 · "
               f"{'실행 중' if engine.running else '정지'}")
    truth = st.session_state.get("truth")
    if truth:
        st.caption("평가용 정답 (파이프라인 입력 아님): " + ", ".join(
            f"GPU{g}={v.get('gt_variant') or v.get('gt_label')}" for g, v in truth.items()))

    notify(snap)
    windows = pd.DataFrame(snap["windows"])
    gpus = sorted(snap["rows"])
    left, right = st.columns([11, 9], gap="medium")
    with left:
        st.subheader("GPU 전력")
        for gpu in gpus:
            st.plotly_chart(power_figure(snap["rows"][gpu], windows, gpu), width="stretch", key=f"p{gpu}")
        grid_panel(snap)
    with right:
        st.subheader("창 분석")
        for gpu in gpus:
            st.markdown(gpu_status(gpu, snap), unsafe_allow_html=True)
        if not windows.empty:
            view = windows.tail(8).iloc[::-1][["gpu_id", "p_value", "is_candidate", "mean_w", "active_evidence"]]
            view["active_evidence"] = view["active_evidence"].map(lambda v: ", ".join(v or []))
            st.dataframe(view.round(3), hide_index=True, width="stretch", height=200)
        agent_panel(snap, engine.settings.llm_model)


st.title("GridTriage Console")
dashboard()
