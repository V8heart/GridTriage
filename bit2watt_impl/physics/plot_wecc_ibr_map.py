"""Topology map of public WECC 179 with SG→IBR replacement overlays.

The bundled WECC case has no geographic coordinates. Layout is a graph spring
layout from Line connectivity — a schematic topology map, not a regional map.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gridtriage-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from bit2watt_impl.physics.ibr_penetration import plan_replacements
from bit2watt_impl.physics.simulation import (
    _configure_constant_power,
    _import_andes,
    ensure_parent,
    select_pq_idx,
)


def _load_wecc_raw():
    andes = _import_andes()
    ss = andes.load(
        andes.get_case("wecc/wecc.raw"),
        addfile=andes.get_case("wecc/wecc_gencls.dyr"),
        setup=True,
        no_output=True,
    )
    _configure_constant_power(ss)
    return ss


def _build_graph(ss) -> nx.Graph:
    graph = nx.Graph()
    for bus in ss.Bus.idx.v:
        graph.add_node(int(bus) if str(bus).isdigit() else bus)
    for b1, b2 in zip(ss.Line.bus1.v, ss.Line.bus2.v):
        u = int(b1) if str(b1).isdigit() else b1
        v = int(b2) if str(b2).isdigit() else b2
        if graph.has_node(u) and graph.has_node(v):
            graph.add_edge(u, v)
    return graph


def _layout(graph: nx.Graph, seed: int = 42) -> dict:
    # Fixed seed for reproducible schematic maps.
    return nx.spring_layout(graph, seed=seed, k=1.8 / np.sqrt(max(len(graph), 1)), iterations=200)


def _gen_table(ss) -> list[dict]:
    rows = []
    sn = np.asarray(ss.GENCLS.Sn.v, dtype=float)
    order = np.argsort(-sn)
    cum = 0.0
    total = float(sn.sum())
    for rank, i in enumerate(order, start=1):
        cum += float(sn[i])
        rows.append(
            {
                "rank": rank,
                "i": int(i),
                "idx": str(ss.GENCLS.idx.v[i]),
                "bus": int(ss.GENCLS.bus.v[i]),
                "sn": float(sn[i]),
                "cum_pen": cum / total,
            }
        )
    return rows


def plot_ibr_maps(
    *,
    output: Path,
    targets: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7),
) -> Path:
    ss = _load_wecc_raw()
    graph = _build_graph(ss)
    pos = _layout(graph)
    gens = _gen_table(ss)
    pq = select_pq_idx(ss, strategy="largest_p0")
    pq_bus = int(ss.PQ.get(src="bus", idx=pq, attr="v"))
    pq_p0 = float(ss.PQ.get(src="p0", idx=pq, attr="v"))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        replaced_idx = set(plan_replacements(ss, target))
        replaced_buses = {int(ss.GENCLS.bus.v[i]) for i in replaced_idx}
        remaining_buses = {
            int(ss.GENCLS.bus.v[i])
            for i in range(len(ss.GENCLS))
            if i not in replaced_idx
        }

        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#c5c9d0", width=0.4, alpha=0.55)
        # non-generator buses
        other = [n for n in graph.nodes if n not in remaining_buses and n not in replaced_buses]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=other,
            node_size=12,
            node_color="#d0d4da",
            ax=ax,
            linewidths=0,
        )

        # remaining SG
        rem = list(remaining_buses)
        rem_sizes = []
        for bus in rem:
            matches = [g for g in gens if g["bus"] == bus]
            rem_sizes.append(40 + 0.004 * max((g["sn"] for g in matches), default=0))
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=rem,
            node_size=rem_sizes,
            node_color="#2f6fed",
            ax=ax,
            edgecolors="white",
            linewidths=0.4,
            label="remaining GENCLS",
        )

        # replaced → IBR
        rep = list(replaced_buses)
        rep_sizes = []
        for bus in rep:
            matches = [g for g in gens if g["bus"] == bus]
            rep_sizes.append(50 + 0.004 * max((g["sn"] for g in matches), default=0))
        if rep:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=rep,
                node_size=rep_sizes,
                node_color="#e4572e",
                ax=ax,
                edgecolors="white",
                linewidths=0.5,
                label="replaced → REGCA1",
            )
            labels = {bus: str(bus) for bus in rep}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)

        # PQ injection bus
        if pq_bus in pos:
            ax.scatter(
                [pos[pq_bus][0]],
                [pos[pq_bus][1]],
                s=220,
                marker="*",
                c="#f0c808",
                edgecolors="#333",
                linewidths=0.6,
                zorder=5,
                label=f"PQ inject bus {pq_bus}",
            )
            ax.annotate(
                f"PQ {pq_p0:.1f} pu\nbus {pq_bus}",
                xy=pos[pq_bus],
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7,
                color="#7a5b00",
            )

        chosen = plan_replacements(ss, target)
        sn_rep = float(sum(float(ss.GENCLS.Sn.v[i]) for i in chosen))
        sn_tot = float(np.sum(ss.GENCLS.Sn.v))
        ax.set_title(
            f"target {target:.0%} → replaced {len(chosen)} units "
            f"(achieved {sn_rep / sn_tot:.1%} Sn)",
            fontsize=10,
        )
        ax.set_axis_off()
        if target == 0.0:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Public WECC 179 GENCLS — schematic topology with IBR replacement\n"
        "(spring layout from Line connectivity; not a geographic map / not Korean grid)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_parent(output)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_replacement_ladder(*, output: Path) -> Path:
    ss = _load_wecc_raw()
    gens = _gen_table(ss)
    in_30 = set(plan_replacements(ss, 0.3))
    in_50 = set(plan_replacements(ss, 0.5))
    in_70 = set(plan_replacements(ss, 0.7))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    y = np.arange(len(gens))[::-1]
    sn = [g["sn"] for g in gens]
    colors = []
    for g in gens:
        i = g["i"]
        if i in in_30:
            colors.append("#e4572e")
        elif i in in_50:
            colors.append("#f2a541")
        elif i in in_70:
            colors.append("#f6d55c")
        else:
            colors.append("#2f6fed")
    ax.barh(y, sn, color=colors, edgecolor="white", linewidth=0.4)
    labels = [f"{g['idx']} @ bus {g['bus']}  ({g['cum_pen']:.0%} cum)" for g in gens]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Machine Sn (MVA)")
    ax.set_title(
        "Capacity-ranked GENCLS replacement order on public WECC 179\n"
        "red=in 30% set / orange=added by 50% / yellow=added by 70% / blue=kept as SG"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    ensure_parent(output)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot schematic WECC topology with IBR replacement overlays"
    )
    parser.add_argument(
        "--map-out",
        type=Path,
        default=Path("dataset/eval/wecc_ibr_topology_map.png"),
    )
    parser.add_argument(
        "--ladder-out",
        type=Path,
        default=Path("dataset/eval/wecc_ibr_replacement_order.png"),
    )
    args = parser.parse_args()
    map_path = plot_ibr_maps(output=args.map_out)
    ladder_path = plot_replacement_ladder(output=args.ladder_out)
    print(f"wrote {map_path}")
    print(f"wrote {ladder_path}")


if __name__ == "__main__":
    main()
