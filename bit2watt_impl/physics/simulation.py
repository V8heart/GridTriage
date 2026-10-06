"""Robust ANDES time-domain execution for public test systems."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dataset.attack_profiles import AttackProfile
from bit2watt_impl.physics.waveforms import build_event_schedule


@dataclass
class StepRecord:
    attempt: int
    target_s: float
    reached_s: float
    ok: bool
    busted: bool
    chunk_s: float
    exit_code: int | None = None
    message: str = ""


@dataclass
class SimulationResult:
    row: dict[str, Any]
    convergence: list[dict[str, Any]]


def _import_andes():
    try:
        import andes
    except ImportError as exc:
        raise RuntimeError(
            "ANDES is unavailable. Activate .venv-physics or run setup_physics_env.sh."
        ) from exc
    return andes


def _configure_constant_power(ss: Any) -> None:
    ss.config.warn_abnormal = 0
    ss.PQ.config.pq2z = 0
    ss.PQ.config.p2p, ss.PQ.config.p2i, ss.PQ.config.p2z = 1.0, 0.0, 0.0
    ss.TDS.config.no_tqdm = 1


def build_kundur_system() -> Any:
    andes = _import_andes()
    ss = andes.load(
        andes.get_case("kundur/kundur_ieeest.xlsx"),
        setup=False,
        no_output=True,
    )
    # This workbook contains a line-trip Toggle at t=2 s. It belongs to the
    # case's original disturbance study and must be disabled so the measured
    # response is attributable only to our PQ-load waveform.
    disabled_events: list[str] = []
    if len(ss.Toggle):
        ss.Toggle.u.v[:] = [0] * len(ss.Toggle)
        disabled_events.append("Toggle")
    ss.setup()
    ss._gridpulse_disabled_events = disabled_events
    _configure_constant_power(ss)
    if not ss.PFlow.run():
        raise RuntimeError("Kundur power flow did not converge")
    return ss


def build_wecc_system() -> Any:
    return build_wecc_system_at_penetration(0.0)


def build_wecc_system_at_penetration(
    penetration: float,
    *,
    require_pflow: bool = True,
    ppf_events: list[tuple[float, float]] | None = None,
) -> Any:
    """Load public WECC 179 GENCLS and optionally replace SGs with REGCA1.

    ``penetration`` is capacity fraction (Sn). Replacement happens before setup.
    Attaches ``_gridpulse_ibr_meta`` on the System for downstream logging.
    """
    from bit2watt_impl.physics.ibr_penetration import (
        finalize_inertia_meta,
        replace_sg_with_ibr,
    )

    andes = _import_andes()
    raw = andes.get_case("wecc/wecc.raw")
    dyr = andes.get_case("wecc/wecc_gencls.dyr")
    ss = andes.load(raw, addfile=dyr, setup=False, no_output=True)
    meta = replace_sg_with_ibr(ss, float(penetration))
    timeseries_path: str | None = None
    if ppf_events:
        # ANDES TimeSeries must be added before setup. This avoids repeatedly
        # stopping/restarting TDS for the ~20-step/cycle sine approximation,
        # which destabilizes high-IBR cases even at tiny amplitudes.
        import csv
        import os
        import tempfile

        powers = np.asarray(ss.PQ.p0.v, dtype=float)
        status = np.asarray(ss.PQ.u.v, dtype=float)
        eligible = np.where(
            (status > 0) & np.isfinite(powers) & (powers > 0),
            powers,
            np.nan,
        )
        pq_pos = int(np.nanargmax(eligible))
        pq_idx = ss.PQ.idx.v[pq_pos]
        p0 = float(powers[pq_pos])
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="gridpulse_ppf_",
            delete=False,
            newline="",
        )
        timeseries_path = handle.name
        with handle:
            writer = csv.DictWriter(handle, fieldnames=["t", "Ppf"])
            writer.writeheader()
            for time_s, relative_delta in ppf_events:
                writer.writerow(
                    {
                        "t": float(time_s),
                        "Ppf": p0 * (1.0 + float(relative_delta)),
                    }
                )
        ss.add(
            "TimeSeries",
            mode=1,
            path=timeseries_path,
            sheet="unused",
            fields="Ppf",
            tkey="t",
            model="PQ",
            dev=pq_idx,
            dests="Ppf",
        )
        ss._gridpulse_timeseries_injection = True
        ss._gridpulse_timeseries_pq_idx = pq_idx
    ss.setup()
    if timeseries_path:
        os.unlink(timeseries_path)
    meta = finalize_inertia_meta(ss, meta)
    ss._gridpulse_ibr_meta = meta
    _configure_constant_power(ss)
    pflow_ok = bool(ss.PFlow.run())
    ss._gridpulse_pflow_ok = pflow_ok
    if require_pflow and not pflow_ok:
        raise RuntimeError(
            f"WECC 179 power flow did not converge at IBR penetration={penetration}"
        )
    return ss


def current_time(ss: Any) -> float:
    """Return simulation clock in seconds.

    Fresh ANDES systems report ``dae.t == -1`` before the first successful TDS
    step. Treat non-positive clocks as 0 so ``run_to`` does not emit negative
    segment targets (e.g. -0.5) that waste a retry.
    """
    raw = getattr(ss.dae, "t", 0.0)
    try:
        value = float(np.asarray(raw).reshape(-1)[0])
    except Exception:
        value = 0.0
    if not np.isfinite(value) or value < 0.0:
        return 0.0
    return value


def select_pq_idx(
    ss: Any,
    requested: Any | None = None,
    *,
    strategy: str = "first",
) -> Any:
    indices = list(ss.PQ.idx.v)
    if not indices:
        raise RuntimeError("test system has no PQ loads")
    if requested is not None:
        requested_text = str(requested)
        for idx in indices:
            if str(idx) == requested_text:
                return idx
        raise ValueError(f"unknown PQ idx: {requested}")
    if strategy == "first":
        return indices[0]
    if strategy == "largest_p0":
        powers = np.asarray(ss.PQ.p0.v, dtype=float)
        status = np.asarray(ss.PQ.u.v, dtype=float)
        eligible = np.where((status > 0) & np.isfinite(powers) & (powers > 0), powers, np.nan)
        if not np.isfinite(eligible).any():
            raise RuntimeError("test system has no online positive-P PQ load")
        return indices[int(np.nanargmax(eligible))]
    raise ValueError(f"unknown PQ selection strategy: {strategy}")


def run_to(
    ss: Any,
    target_s: float,
    *,
    max_chunk_s: float = 0.5,
    min_chunk_s: float = 0.01,
    max_retries: int = 3,
) -> tuple[bool, list[StepRecord], str]:
    """Advance to target; never retry a busted System in place."""
    records: list[StepRecord] = []
    chunk = max_chunk_s
    retries = 0
    tolerance = 1e-7
    target_s = float(target_s)
    if target_s <= 0:
        return True, records, ""

    while current_time(ss) < target_s - tolerance:
        before = current_time(ss)
        segment_target = min(before + chunk, target_s)
        if segment_target <= before + tolerance:
            # Clock already at/above target after clamping; nothing to do.
            break
        ss.TDS.config.tf = segment_target
        try:
            returned = ss.TDS.run()
            ok = bool(returned)
            message = "" if ok else "TDS.run returned false"
        except Exception as exc:  # ANDES can raise on singular Jacobians
            ok = False
            message = f"{type(exc).__name__}: {exc}"
        reached = current_time(ss)
        busted = bool(getattr(ss.TDS, "busted", False))
        progressed = reached > before + tolerance
        reached_target = reached >= segment_target - tolerance
        records.append(
            StepRecord(
                attempt=len(records) + 1,
                target_s=float(segment_target),
                reached_s=float(reached),
                ok=bool(ok and reached_target),
                busted=busted,
                chunk_s=float(chunk),
                exit_code=(
                    int(ss.exit_code)
                    if getattr(ss, "exit_code", None) is not None
                    else None
                ),
                message=message,
            )
        )

        if busted:
            return False, records, "TDS entered busted state; fresh replay required"
        if ok and reached_target:
            retries = 0
            chunk = max_chunk_s
            continue

        retries += 1
        chunk /= 2.0
        if retries >= max_retries or chunk < min_chunk_s:
            reason = "TDS failed to reach target after adaptive retries"
            if not progressed:
                reason += " (no time progress)"
            return False, records, reason

    return True, records, ""


def _extract_omega(
    ss: Any,
    *,
    model_name: str,
    t_start_s: float,
    t_end_s: float,
    forcing_frequency_hz: float | None,
    toggle_times_s: list[float] | None = None,
    transition_mask_s: float = 0.15,
) -> dict[str, Any]:
    model = getattr(ss, model_name)
    frame = ss.TDS.get_timeseries(model.omega)
    times = np.asarray(frame.index, dtype=float)
    omega = np.asarray(frame.values, dtype=float)
    if omega.ndim == 1:
        omega = omega[:, None]
    mask = np.isfinite(times) & (times >= t_start_s) & (times <= t_end_s)
    times, omega = times[mask], omega[mask]
    if len(times) < 8:
        raise RuntimeError("insufficient omega samples in analysis interval")

    unique = np.concatenate(([True], np.diff(times) > 1e-10))
    times, omega = times[unique], omega[unique]
    if len(times) < 8:
        raise RuntimeError("insufficient unique omega samples")

    std_by_generator = np.nanstd(omega, axis=0)
    response_col = int(np.nanargmax(std_by_generator))
    selected = omega[:, response_col]
    nominal_hz = float(getattr(ss.config, "freq", 60.0))
    gradient = np.gradient(omega, times, axis=0)
    selected_grad = gradient[:, response_col]
    # Legacy max metric: sensitive to square-wave edges; keep for diagnostics.
    rocof_max = float(np.nanmax(np.abs(selected_grad)) * nominal_hz)

    toggles = list(toggle_times_s or [])
    rocof_rms: float | None
    rocof_p95: float | None
    rocof_status: str
    if not toggles:
        # Ops / non-periodic path: no schedule to mask.
        rocof_rms = None
        rocof_p95 = None
        rocof_status = "no_toggle_mask"
    else:
        keep = np.ones(len(times), dtype=bool)
        for t_toggle in toggles:
            keep &= np.abs(times - float(t_toggle)) > float(transition_mask_s)
        masked_grad = selected_grad[keep]
        masked_grad = masked_grad[np.isfinite(masked_grad)]
        if masked_grad.size < 30:
            rocof_rms = None
            rocof_p95 = None
            rocof_status = "insufficient_samples_after_transition_mask"
        else:
            rocof_rms = float(np.sqrt(np.mean(masked_grad**2)) * nominal_hz)
            rocof_p95 = float(np.nanpercentile(np.abs(masked_grad), 95) * nominal_hz)
            rocof_status = "ok"

    dt = float(np.median(np.diff(times)))
    uniform_times = np.arange(times[0], times[-1], dt)
    uniform_signal = np.interp(uniform_times, times, selected)
    centered = uniform_signal - np.mean(uniform_signal)
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    spectrum = np.abs(np.fft.rfft(centered))
    dominant = float(freqs[1 + np.argmax(spectrum[1:])]) if len(freqs) > 2 else 0.0
    forcing_amplitude = None
    if forcing_frequency_hz is not None and len(freqs):
        forcing_bin = int(np.argmin(np.abs(freqs - forcing_frequency_hz)))
        forcing_amplitude = float(2.0 * spectrum[forcing_bin] / len(centered))
    generator_indices = list(model.idx.v)

    return {
        # Alias kept one cycle for older CSV/summary consumers.
        "rocof_hz_s": rocof_max,
        "rocof_max_hz_s": rocof_max,
        "rocof_rms_hz_s": rocof_rms,
        "rocof_p95_hz_s": rocof_p95,
        "rocof_status": rocof_status,
        "osc_std": float(np.nanmax(std_by_generator)),
        "osc_ptp": float(np.nanmax(np.ptp(omega, axis=0))),
        "dominant_freq_hz": dominant,
        "forcing_response_amp": forcing_amplitude,
        "response_model": model_name,
        "response_device": str(generator_indices[response_col]),
        "omega_samples": int(len(times)),
    }


def _run_once(
    builder: Callable[[], Any],
    profile: AttackProfile,
    *,
    test_system: str,
    model_name: str,
    pq_idx: Any | None,
    max_chunk_s: float,
    criteria: int,
    pq_selection: str,
    tstep_s: float,
) -> SimulationResult:
    ss = builder()
    ss.TDS.config.criteria = criteria
    ss.TDS.config.tstep = tstep_s
    ss.TDS.config.shrinkt = 1
    ss.TDS.config.max_iter = max(30, int(ss.TDS.config.max_iter))
    # Initialize TDS after config so dae.t starts at 0 (not -1) and REGCA
    # algebraic states are consistent before the first chunked run_to.
    if hasattr(ss.TDS, "init"):
        ss.TDS.init()
    selected_pq = select_pq_idx(ss, pq_idx, strategy=pq_selection)
    p0 = float(ss.PQ.get(src="p0", idx=selected_pq, attr="v"))
    pq_bus = str(ss.PQ.get(src="bus", idx=selected_pq, attr="v"))
    all_records: list[dict[str, Any]] = []
    schedule = build_event_schedule(profile)
    toggle_times = [float(event.time_s) for event in schedule]

    if not getattr(ss, "_gridpulse_timeseries_injection", False):
        for segment_index, event in enumerate(schedule):
            ok, records, reason = run_to(ss, event.time_s, max_chunk_s=max_chunk_s)
            for record in records:
                item = asdict(record)
                item.update({"segment_index": segment_index, "attack_id": profile.attack_id})
                all_records.append(item)
            if not ok:
                return SimulationResult(
                    row={
                        **profile.to_dict(),
                        "test_system": test_system,
                        "case_version": "ANDES-2.0.0",
                        "pq_idx": str(selected_pq),
                        "pq_bus": pq_bus,
                        "pq_base_pu": p0,
                        "pq_selection": pq_selection,
                        "criteria": criteria,
                        "tstep_s": tstep_s,
                        "disabled_builtin_events": ",".join(
                            getattr(ss, "_gridpulse_disabled_events", [])
                        ),
                        "converged": False,
                        "failure_reason": reason,
                        "t_reached_s": current_time(ss),
                    },
                    convergence=all_records,
                )
            # p0(attr='v') and Ppf are both system-base values.
            ss.PQ.set(
                src="Ppf",
                idx=selected_pq,
                value=p0 * (1.0 + event.relative_delta),
            )

    final_target = profile.t_end_s + 2.0
    ok, records, reason = run_to(ss, final_target, max_chunk_s=max_chunk_s)
    for record in records:
        item = asdict(record)
        item.update({"segment_index": len(schedule), "attack_id": profile.attack_id})
        all_records.append(item)
    base = {
        **profile.to_dict(),
        "test_system": test_system,
        "case_version": "ANDES-2.0.0",
        "pq_idx": str(selected_pq),
        "pq_bus": pq_bus,
        "pq_base_pu": p0,
        "pq_selection": pq_selection,
        "criteria": criteria,
        "tstep_s": tstep_s,
        "disabled_builtin_events": ",".join(
            getattr(ss, "_gridpulse_disabled_events", [])
        ),
        "converged": bool(ok),
        "failure_reason": reason,
        "t_reached_s": current_time(ss),
    }
    if ok:
        features = _extract_omega(
            ss,
            model_name=model_name,
            t_start_s=max(profile.t_start_s + 1.0, 2.0),
            t_end_s=final_target,
            forcing_frequency_hz=profile.frequency_hz,
            toggle_times_s=toggle_times,
        )
        base.update(features)
        if features.get("rocof_status") == "insufficient_samples_after_transition_mask":
            # TDS succeeded; keep converged=True but surface RoCoF gap.
            if not base.get("failure_reason"):
                base["failure_reason"] = "insufficient_samples_after_transition_mask"
    return SimulationResult(row=base, convergence=all_records)


def simulate_profile(
    builder: Callable[[], Any],
    profile: AttackProfile,
    *,
    test_system: str,
    model_name: str,
    pq_idx: Any | None = None,
    max_chunk_s: float = 0.5,
    max_rebuilds: int = 2,
    criteria: int = 1,
    pq_selection: str = "first",
    base_tstep_s: float = 1.0 / 30.0,
) -> SimulationResult:
    """Run a profile, rebuilding and replaying from t=0 after busted failures.

    Rebuild 0 keeps the requested step/chunk. Later rebuilds may shrink them.
    Shrinking on the first attempt was observed to make WECC+IBR cases fail
    *before* the attack start time after an earlier post-toggle bust.
    """
    combined_logs: list[dict[str, Any]] = []
    last: SimulationResult | None = None
    for rebuild in range(max_rebuilds + 1):
        shrink = 2 ** max(rebuild - 1, 0) if rebuild > 0 else 1
        result = _run_once(
            builder,
            profile,
            test_system=test_system,
            model_name=model_name,
            pq_idx=pq_idx,
            max_chunk_s=max_chunk_s / shrink,
            criteria=criteria,
            pq_selection=pq_selection,
            tstep_s=base_tstep_s / shrink,
        )
        for item in result.convergence:
            item["rebuild"] = rebuild
        combined_logs.extend(result.convergence)
        result.row["rebuild_count"] = rebuild
        last = result
        if result.row["converged"]:
            result.convergence = combined_logs
            return result
        if not any(bool(item.get("busted")) for item in result.convergence):
            break
    assert last is not None
    last.convergence = combined_logs
    return last


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def power_series_to_relative_events(
    power_series: np.ndarray,
    sample_hz: float,
    *,
    t_start_s: float = 1.0,
    max_amp_frac: float = 0.15,
    min_interval_s: float = 0.1,
    change_eps: float = 0.01,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Convert observed GPU watts into sparse PQ relative-delta events.

    Mapping is shape-preserving relative deviation around the window mean,
    not a conversion from watts to grid MW. The relative deviation is clipped
    to ``max_amp_frac`` so public test-system TDS remains numerically stable.
    """
    power = np.asarray(power_series, dtype=float)
    power = power[np.isfinite(power)]
    if len(power) < 8:
        raise ValueError("power_series needs at least 8 finite samples")
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")

    mean = float(np.mean(power))
    if abs(mean) < 1e-9:
        raise ValueError("power_series mean is near zero; cannot normalize")
    raw_delta = (power - mean) / mean
    peak_raw = float(np.max(np.abs(raw_delta)))
    scale = 1.0
    if peak_raw > max_amp_frac and peak_raw > 0:
        scale = max_amp_frac / peak_raw
    clipped = np.clip(raw_delta * scale, -max_amp_frac, max_amp_frac)

    events: list[tuple[float, float]] = []
    last_t = -1e9
    last_delta: float | None = None
    for i, delta in enumerate(clipped):
        t = t_start_s + i / sample_hz
        delta_f = float(delta)
        if last_delta is None:
            events.append((t, delta_f))
            last_t, last_delta = t, delta_f
            continue
        if (t - last_t) < min_interval_s and abs(delta_f - last_delta) < change_eps:
            continue
        if abs(delta_f - last_delta) < change_eps and (t - last_t) < max(min_interval_s * 5, 0.5):
            continue
        events.append((t, delta_f))
        last_t, last_delta = t, delta_f

    t_end = t_start_s + (len(clipped) - 1) / sample_hz
    if not events or abs(events[-1][0] - t_end) > 1e-9:
        events.append((t_end, float(clipped[-1])))
    events.append((t_end + 2.0, 0.0))

    meta = {
        "mean_w": mean,
        "peak_raw_delta": peak_raw,
        "amp_scale": float(scale),
        "max_amp_frac": float(max_amp_frac),
        "n_events": float(len(events)),
        "t_start_s": float(t_start_s),
        "t_end_s": float(t_end),
        "duration_s": float(t_end - t_start_s),
    }
    return events, meta


FACILITY_BUS_CAP = 0.9  # PQ load may not swing below 10% of its base (TDS stability)


def facility_bus_fraction(power_series: np.ndarray, *, p0_bus: float, p_system: float, tdp_w: float,
                          participation: float, facility_share: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-sample relative change of one PQ load for a facility of ``facility_share`` of system load."""
    power = np.asarray(power_series, dtype=float)
    power = power[np.isfinite(power)]
    pu = (power - power.mean()) / float(tdp_w)
    system_frac = pu * float(participation) * float(facility_share)
    bus = system_frac * float(p_system) / float(p0_bus)
    clipped = bool(np.max(np.abs(bus)) > FACILITY_BUS_CAP)
    return np.clip(bus, -FACILITY_BUS_CAP, FACILITY_BUS_CAP), {
        "scale": "facility",
        "gpu_pu_peak": float(np.max(np.abs(pu))),
        "injected_frac_of_system_load": float(np.max(np.abs(system_frac))),
        "bus_frac_peak": float(min(np.max(np.abs(bus)), FACILITY_BUS_CAP)),
        "participation": float(participation),
        "facility_share": float(facility_share),
        "clipped": clipped,
    }


def inject_observed_waveform(
    power_series: np.ndarray,
    sample_hz: float,
    *,
    test_system: str = "kundur_ieeest",
    max_amp_frac: float = 0.15,
    min_interval_s: float = 0.1,
    max_chunk_s: float = 0.5,
    criteria: int = 1,
    facility: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay an observed power(t) window onto PQ.Ppf of a public test system.

    Unlike fixed AttackProfile square/ramp generators, this path does not invent
    a synthetic frequency/amplitude scenario. It replays the candidate window
    shape as a relative load modulation on Kundur/WECC.

    ``facility`` = {"tdp_w", "participation", "facility_share"} scales the shape by
    ΔP/P_system = (ΔP_GPU/TDP) × participation × (P_facility/P_system), the rule of
    the project brief §4.1.5, then converts it to the selected PQ bus. Without it the
    window is normalised to its own mean and capped at ``max_amp_frac`` (legacy).
    """
    if test_system == "kundur_ieeest":
        builder, model_name, pq_selection = build_kundur_system, "GENROU", "first"
    elif test_system in {"wecc_179_gencls", "wecc"}:
        builder, model_name, pq_selection = build_wecc_system, "GENCLS", "largest_p0"
    else:
        raise ValueError(f"unsupported test_system: {test_system}")

    ss = builder()
    ss.TDS.config.criteria = criteria
    ss.TDS.config.tstep = 1.0 / 30.0
    ss.TDS.config.shrinkt = 1
    ss.TDS.config.max_iter = max(30, int(ss.TDS.config.max_iter))
    if hasattr(ss.TDS, "init"):
        ss.TDS.init()
    selected_pq = select_pq_idx(ss, strategy=pq_selection)
    p0 = float(ss.PQ.get(src="p0", idx=selected_pq, attr="v"))
    pq_bus = str(ss.PQ.get(src="bus", idx=selected_pq, attr="v"))

    scaling: dict[str, Any] = {"scale": "window"}
    series, cap = power_series, max_amp_frac
    if facility:
        total = float(np.sum(np.asarray(ss.PQ.p0.v, dtype=float)))
        bus_frac, scaling = facility_bus_fraction(power_series, p0_bus=p0, p_system=total, **facility)
        series, cap = 1.0 + bus_frac, FACILITY_BUS_CAP
    events, meta = power_series_to_relative_events(
        series,
        sample_hz,
        max_amp_frac=cap,
        min_interval_s=min_interval_s,
    )
    meta.update({k: v for k, v in scaling.items() if isinstance(v, (int, float))})

    for t_s, relative_delta in events:
        ok, _, reason = run_to(ss, t_s, max_chunk_s=max_chunk_s)
        if not ok:
            return {
                "test_system": test_system,
                "case_version": "ANDES-2.0.0",
                "mode": "observed_waveform_replay",
                "pq_idx": str(selected_pq),
                "pq_bus": pq_bus,
                "pq_base_pu": p0,
                "converged": False,
                "failure_reason": reason,
                "t_reached_s": current_time(ss),
                "osc_std": None,
                "rocof_hz_s": None,
                "rocof_max_hz_s": None,
                "rocof_rms_hz_s": None,
                "rocof_p95_hz_s": None,
                "rocof_status": "tds_failed",
                "dominant_freq_hz": None,
                "osc_ptp": None,
                **{f"waveform_{k}": v for k, v in meta.items()},
            }
        ss.PQ.set(src="Ppf", idx=selected_pq, value=p0 * (1.0 + relative_delta))

    features = _extract_omega(
        ss,
        model_name=model_name,
        t_start_s=meta["t_start_s"],
        t_end_s=meta["t_end_s"] + 2.0,
        forcing_frequency_hz=None,
        toggle_times_s=[],
    )
    return {
        "test_system": test_system,
        "case_version": "ANDES-2.0.0",
        "mode": "observed_waveform_replay",
        "pq_idx": str(selected_pq),
        "pq_bus": pq_bus,
        "pq_base_pu": p0,
        "converged": True,
        "failure_reason": "",
        "t_reached_s": current_time(ss),
        "scaling": scaling,
        **features,
        **{f"waveform_{k}": v for k, v in meta.items()},
    }


def inject_observed_waveform_with_timeout(
    power_series: np.ndarray,
    sample_hz: float,
    *,
    timeout_s: float = 60.0,
    test_system: str = "kundur_ieeest",
    max_amp_frac: float = 0.15,
    facility: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run observed-waveform injection with a wall-clock timeout.

    Uses SIGALRM so a hung TDS does not block the cyber pipeline forever.
    """
    import signal

    class _PhysicsTimeout(Exception):
        pass

    def _handle(_signum, _frame) -> None:
        raise _PhysicsTimeout(f"physics validation exceeded {timeout_s:.1f}s")

    previous = signal.signal(signal.SIGALRM, _handle)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return inject_observed_waveform(
            power_series,
            sample_hz,
            test_system=test_system,
            max_amp_frac=max_amp_frac,
            facility=facility,
        )
    except _PhysicsTimeout as exc:
        return {
            "test_system": test_system,
            "mode": "observed_waveform_replay",
            "converged": False,
            "failure_reason": str(exc),
            "osc_std": None,
            "rocof_hz_s": None,
            "rocof_max_hz_s": None,
            "rocof_rms_hz_s": None,
            "rocof_p95_hz_s": None,
            "rocof_status": "timeout",
            "dominant_freq_hz": None,
            "osc_ptp": None,
        }
    except Exception as exc:
        return {
            "test_system": test_system,
            "mode": "observed_waveform_replay",
            "converged": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "osc_std": None,
            "rocof_hz_s": None,
            "rocof_max_hz_s": None,
            "rocof_rms_hz_s": None,
            "rocof_p95_hz_s": None,
            "rocof_status": "error",
            "dominant_freq_hz": None,
            "osc_ptp": None,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
