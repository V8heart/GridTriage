"""GridPulse console: real-time view of the v4 pipeline.

    .venv/bin/streamlit run app.py

Live mode polls NVML on this server; Replay mode streams a recorded session.
The page only renders what pipeline/live.py computes (no analysis logic here)
and has no workload-control buttons: output is triage priority, not enforcement.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset.declared_context import DECLARED_POOL
from pipeline.kb.retriever import MODES, HybridRetriever, default_device
from pipeline.live import EngineSettings, LiveEngine, NvmlSource, ReplaySource, observability

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "dataset" / "real" / "index" / "labels.csv"
VERDICT_COLOR = {"attack": "#d62728", "normal": "#2ca02c", "unobservable": "#7f7f7f",
                 "unknown": "#ff7f0e", "invalid": "#9467bd"}

st.set_page_config(page_title="GridPulse Console", layout="wide")


@st.cache_resource(show_spinner="검색 인덱스와 모델을 불러오는 중")
def get_retriever(device: str) -> HybridRetriever:
    return HybridRetriever(device=device)


@st.cache_data
def session_table() -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    lab = lab[lab["valid"].astype(str).str.lower().isin({"true", "1"}) & lab["gpu_role"].eq("target")]
    lab = lab[[(ROOT / "dataset/real/sessions" / s / "telemetry.csv").exists() for s in lab["session_id"]]]
    lab["label"] = lab["gt_variant"].fillna(lab["gt_label"]).astype(str)
    return lab.drop_duplicates("session_id")[["session_id", "label", "declared_job_type"]].sort_values("label")


# -- sidebar: source and settings ------------------------------------------
st.sidebar.title("GridPulse")
mode = st.sidebar.radio("입력", ["Replay (저장 세션)", "Live (NVML)"])
llm = st.sidebar.selectbox("판정 LLM", ["gemma3:12b", "deepseek-r1:14b"])
retrieval = st.sidebar.selectbox("검색 방식", list(MODES), index=list(MODES).index("hybrid_rerank"))
k = st.sidebar.slider("LLM에 넘길 문서 수 k", 3, 12, 8)
host = st.sidebar.text_input("Ollama 주소", "http://127.0.0.1:11434")
devices = [default_device()] + [d for d in ("cuda:1", "cuda:0", "cpu") if d != default_device()]
pipe_device = st.sidebar.selectbox("파이프라인 장치 (검색 모델)", devices)
pipe_gpu = int(pipe_device.split(":")[1]) if pipe_device.startswith("cuda") else None

if mode.startswith("Replay"):
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

c1, c2 = st.sidebar.columns(2)
if c1.button("시작", type="primary") and (source_key[0] == "replay" or source_key[1]):
    old = st.session_state.get("engine")
    if old:
        old.stop()
    src = ReplaySource(source_key[1], speed=source_key[2]) if source_key[0] == "replay" else NvmlSource(dict(source_key[1]))
    engine = LiveEngine(src, get_retriever(pipe_device), EngineSettings(llm_model=llm, retrieval_mode=retrieval, k=k, ollama_host=host))
    engine.start()
    st.session_state["engine"] = engine
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

    windows = pd.DataFrame(snap["windows"])
    cols = st.columns(max(len(snap["rows"]), 1))
    for col, (gpu, rows) in zip(cols, sorted(snap["rows"].items())):
        col.plotly_chart(power_figure(rows, windows, gpu), width="stretch", key=f"p{gpu}")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Stage1 창 (최근)")
        if not windows.empty:
            view = windows.tail(16).iloc[::-1][["gpu_id", "p_value", "is_candidate", "idle_status", "mean_w",
                                                 "inst_step_mad_w", "active_evidence"]]
            view["active_evidence"] = view["active_evidence"].map(lambda v: ", ".join(v or []))
            st.dataframe(view.round(3), hide_index=True, width="stretch")
    with right:
        st.subheader("Stage2: 검색 → LLM 판정")
        for res in snap["analyses"][::-1][:6]:
            j = res["judgment"]
            verdict = j.get("verdict", "?")
            color = VERDICT_COLOR.get(verdict, "#333")
            head = (f"<span style='color:{color};font-weight:700'>{verdict.upper()}</span> "
                    f"{j.get('threat_card') or ''} · GPU{res['gpu_id']} · 창 {res['n_windows']}개 · "
                    f"{j.get('model', '')} {j.get('latency_s', '')}s")
            st.markdown(head, unsafe_allow_html=True)
            st.write(j.get("explanation") or "; ".join(j.get("errors", [])))
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


st.title("GridPulse Console")
dashboard()
