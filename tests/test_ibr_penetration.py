"""Unit checks for WECC IBR capacity replacement (requires ANDES / .venv-physics)."""
from __future__ import annotations

import pytest

pytest.importorskip("andes")

from bit2watt_impl.physics.ibr_penetration import (
    finalize_inertia_meta,
    plan_replacements,
    replace_sg_with_ibr,
)
from bit2watt_impl.physics.simulation import build_wecc_system_at_penetration


def test_plan_replacements_deterministic_counts():
    import andes

    ss = andes.load(
        andes.get_case("wecc/wecc.raw"),
        addfile=andes.get_case("wecc/wecc_gencls.dyr"),
        setup=False,
        no_output=True,
    )
    assert plan_replacements(ss, 0.0) == []
    assert len(plan_replacements(ss, 0.3)) == 3
    assert len(plan_replacements(ss, 0.5)) == 6
    assert len(plan_replacements(ss, 0.7)) == 12


def test_replace_zero_keeps_full_inertia():
    ss = build_wecc_system_at_penetration(0.0)
    meta = ss._gridtriage_ibr_meta
    assert meta["replaced_count"] == 0
    assert meta["achieved_penetration"] == 0.0
    assert meta["inertia_before"] == pytest.approx(8375.75, rel=1e-4)
    assert meta["inertia_after"] == pytest.approx(8375.75, rel=1e-4)


def test_replace_30pct_reduces_inertia():
    ss = build_wecc_system_at_penetration(0.3)
    meta = ss._gridtriage_ibr_meta
    assert meta["replaced_count"] == 3
    assert meta["achieved_penetration"] == pytest.approx(0.3199, rel=1e-3)
    assert meta["inertia_after"] == pytest.approx(5697.39, rel=1e-3)
    assert meta["inertia_before"] > meta["inertia_after"]
    assert len(ss.REGCA1) == 3
    assert len(ss.REECA1) == 3
