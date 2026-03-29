"""
phrase_network.py — 2-D network visualisation of phrase-report connections.

Parses a Markdown phrase report (produced by phrase_report.py) and renders an
interactive matplotlib network graph where:

  - Each node  = one comment document
  - Each edge  = N or more shared distinctive phrases (N adjustable via slider)
  - Edge width = proportional to the number of shared phrases
  - Node colour = community cluster (greedy modularity detection)
  - Hover      = shows full Document ID + connection stats

Usage
-----
    python phrase_network.py <phrase_report.md>
    python phrase_network.py <phrase_report.md> --min-weight 3
    python phrase_network.py <phrase_report.md> --title "HHS-ONC-2026-0001"

Controls
--------
    Slider          — drag to change the minimum shared-phrase threshold
    Scroll wheel    — zoom in / out (centred on cursor)
    R key           — reset to full-graph view
    Hover over node — tooltip with Document ID and connection stats
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
except ImportError:
    print(
        "Error: networkx is required.\n"
        "Install with:  pip install networkx",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Markdown parser ────────────────────────────────────────────────────────────

# Matches phrase section headers produced by phrase_check.stream_report.
# Group 1: the phrase text
# Group 2 (optional): the classification tag — SLOPICAL, TOPICAL, or UNKNOWN
_SECTION_HEADER_RE = re.compile(
    r'^##\s+\d+\.\s+"([^"]+)"\s+\(found in \d+ comments\)'
    r'(?:\s+\[(SLOPICAL|TOPICAL|UNKNOWN)\])?',
    re.MULTILINE,
)
_TABLE_ROW_RE = re.compile(
    r'^\|\s*\d+\s*\|\s*([\w-]+)\s*\|\s*([^|]+?)\s*\|',
    re.MULTILINE,
)


def parse_phrase_report(
    md_path: str,
) -> Tuple[
    Dict[str, Dict],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], int],
]:
    """Parse the Markdown phrase report.

    Returns
    -------
    node_info          : doc_id -> {"doc_id": str, "number": str}
    edge_weights_all   : (doc_id_a, doc_id_b) -> shared-phrase count (all types)
    edge_weights_slop  : counts from slopical phrases only
    edge_weights_topic : counts from topical phrases only
    """
    text = Path(md_path).read_text(encoding="utf-8", errors="replace")
    # With 2 capture groups the split interleaves: [preamble, phrase, tag, body, ...]
    parts = _SECTION_HEADER_RE.split(text)

    node_info: Dict[str, Dict] = {}
    pair_all:   Dict[Tuple[str, str], int] = defaultdict(int)
    pair_slop:  Dict[Tuple[str, str], int] = defaultdict(int)
    pair_topic: Dict[Tuple[str, str], int] = defaultdict(int)

    phrase_count = 0
    slop_count = 0
    topic_count = 0

    # parts layout (2 capture groups): [pre, phrase0, tag0, body0, phrase1, tag1, body1, …]
    for i in range(1, len(parts), 3):
        phrase = parts[i]                               # always present
        tag = (parts[i + 1] or "UNKNOWN").upper() if i + 1 < len(parts) else "UNKNOWN"
        body = parts[i + 2] if i + 2 < len(parts) else ""

        doc_ids_in_phrase: List[str] = []

        for doc_id_raw, submitter_raw in _TABLE_ROW_RE.findall(body):
            doc_id = doc_id_raw.strip()
            if not doc_id:
                continue
            doc_ids_in_phrase.append(doc_id)
            if doc_id not in node_info:
                tail = doc_id.rsplit("-", 1)
                node_info[doc_id] = {
                    "doc_id": doc_id,
                    "number": tail[-1] if len(tail) == 2 else doc_id,
                    "submitter": submitter_raw.strip(),
                }

        # Count phrase type once per phrase (not once per edge pair)
        if tag == "SLOPICAL":
            slop_count += 1
        elif tag == "TOPICAL":
            topic_count += 1

        for id_a, id_b in combinations(doc_ids_in_phrase, 2):
            key: Tuple[str, str] = (min(id_a, id_b), max(id_a, id_b))
            pair_all[key] += 1
            if tag == "SLOPICAL":
                pair_slop[key] += 1
            elif tag == "TOPICAL":
                pair_topic[key] += 1

        phrase_count += 1

    classified = slop_count + topic_count > 0
    type_summary = (
        f" · {slop_count} slopical · {topic_count} topical"
        if classified else " · (unclassified)"
    )
    print(
        f"[phrase-network] Parsed {phrase_count} phrases{type_summary} · "
        f"{len(node_info)} unique comments · "
        f"{len(pair_all)} unique pairs with ≥1 shared phrase"
    )
    return node_info, dict(pair_all), dict(pair_slop), dict(pair_topic)


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(
    node_info: Dict[str, Dict],
    edge_weights: Dict[Tuple[str, str], int],
    min_weight: int = 1,
) -> nx.Graph:
    G: nx.Graph = nx.Graph()
    for (a, b), w in edge_weights.items():
        if w >= min_weight:
            if a not in G:
                G.add_node(a, **(node_info.get(a) or {"doc_id": a, "number": a}))
            if b not in G:
                G.add_node(b, **(node_info.get(b) or {"doc_id": b, "number": b}))
            G.add_edge(a, b, weight=w)
    return G


# ── Community colours ──────────────────────────────────────────────────────────

def assign_community_colors(G: nx.Graph) -> Dict[str, str]:
    if len(G.nodes) == 0:
        return {}
    try:
        communities = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        communities = [{n} for n in G.nodes]

    cmap = plt.get_cmap("tab20")
    color_map: Dict[str, str] = {}
    for idx, community in enumerate(sorted(communities, key=len, reverse=True)):
        hex_color = mcolors.to_hex(cmap(idx % 20))
        for node in community:
            color_map[node] = hex_color
    return color_map


# ── Main visualisation ─────────────────────────────────────────────────────────

def visualise(
    md_path: str,
    initial_min_weight: int = 5,
    title: Optional[str] = None,
) -> None:
    print(f"[phrase-network] Parsing: {md_path}")
    node_info, edge_weights_all, edge_weights_slop, edge_weights_topic = (
        parse_phrase_report(md_path)
    )

    if not edge_weights_all:
        print("Error: no shared phrases found.", file=sys.stderr)
        sys.exit(1)

    # Determine whether the report was classified (has slop/topic split)
    classified = bool(edge_weights_slop or edge_weights_topic)

    # The slider max is always derived from the combined set so that the
    # scale stays stable when switching between filters.
    max_weight = max(edge_weights_all.values())
    print(
        f"[phrase-network] Max shared phrases between any pair: {max_weight}\n"
        f"[phrase-network] Computing spring layout …"
    )

    # Community detection uses the full combined graph for stable colours.
    # Spring layout is recomputed per draw_graph call (filtered graph).
    G_full = build_graph(node_info, edge_weights_all, min_weight=1)

    print("[phrase-network] Detecting communities …")
    color_map_full = assign_community_colors(G_full)

    BG = "#12122a"
    EDGE_COLOR = "#f6cf62"
    TEXT_COLOR = "#fafafb"
    SLIDER_BG = "#1e1e3e"
    edge_rgb = mcolors.to_rgb(EDGE_COLOR)

    # ── Figure & axis layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    fig.canvas.manager.set_window_title(
        title or f"Phrase Network – {Path(md_path).name}"
    )

    ax_main = fig.add_axes([0.04, 0.12, 0.92, 0.84], facecolor=BG)
    ax_main.set_axis_off()
    ax_main.set_facecolor(BG)

    # Slider — slightly narrower to leave room for radio buttons on the right
    ax_slider = fig.add_axes([0.15, 0.035, 0.60, 0.030], facecolor=SLIDER_BG)
    slider = Slider(
        ax=ax_slider,
        label="Min shared\nphrases",
        valmin=1,
        valmax=max_weight,
        valinit=min(initial_min_weight, max_weight),
        valstep=1,
        color=EDGE_COLOR,
    )
    slider.label.set_color(TEXT_COLOR)
    slider.label.set_fontsize(11)
    slider.valtext.set_color(TEXT_COLOR)
    slider.valtext.set_fontsize(11)

    # Radio buttons — right of slider, only shown when report is classified
    _RADIO_LABELS = ("All Shared Phrases", "Slop Phrases Only", "Topical Phrases Only")
    ax_radio = fig.add_axes([0.79, 0.005, 0.20, 0.085], facecolor=SLIDER_BG)
    radio = RadioButtons(
        ax_radio,
        _RADIO_LABELS,
        active=0,
        activecolor=EDGE_COLOR,
    )
    for lbl in radio.labels:
        lbl.set_color(TEXT_COLOR)
        lbl.set_fontsize(11)
    ax_radio.set_visible(classified)  # hide when report has no classification

    # Title
    ax_main.set_title(
        title or Path(md_path).stem, color=TEXT_COLOR, fontsize=12, pad=6,
    )

    # Stats text
    stats_text = ax_main.text(
        0.005, 0.005, "",
        transform=ax_main.transAxes,
        color="#8888aa", fontsize=9.0, va="bottom",
    )

    # Pan / zoom hint
    ax_main.text(
        0.999, 0.005,
        "Drag to pan  \u00b7  Scroll to zoom  \u00b7  R to reset view",
        transform=ax_main.transAxes,
        color="#666688", fontsize=10.0, va="bottom", ha="right",
    )

    # ── Mutable draw state ─────────────────────────────────────────────────────
    state: Dict = {
        "nodes": [],
        "node_xy": {},
        "xs": np.array([]),
        "ys": np.array([]),
        "annot": None,
        "dynamic_artists": [],
        "home_xlim": None,
        "home_ylim": None,
        # active_weights: the edge dict currently shown (swapped by radio)
        "active_weights": edge_weights_all,
        # panning state
        "pan_start_px": None,   # (x_px, y_px) recorded on mouse-down
        "is_panning": False,    # True once the user has dragged ≥1 px
    }

    def _remove_dynamic_artists() -> None:
        for artist in state["dynamic_artists"]:
            try:
                artist.remove()
            except Exception:
                pass
        state["dynamic_artists"] = []
        if state["annot"] is not None:
            try:
                state["annot"].remove()
            except Exception:
                pass
            state["annot"] = None

    # ── Draw function ──────────────────────────────────────────────────────────
    def draw_graph(min_w: int) -> None:
        _remove_dynamic_artists()
        active = state["active_weights"]

        visible_edges = [
            (a, b, w) for (a, b), w in active.items() if w >= min_w
        ]
        visible_node_set: set = set()
        for a, b, _ in visible_edges:
            visible_node_set.add(a)
            visible_node_set.add(b)

        stats_text.set_text("")

        filter_label = radio.value_selected if classified else "Both"

        if not visible_node_set:
            msg = ax_main.text(
                0.5, 0.5,
                f"No connections with ≥ {min_w} shared phrases "
                f"({filter_label}).\n"
                "Drag the slider left or switch the filter.",
                transform=ax_main.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=13,
            )
            state["dynamic_artists"].append(msg)
            state["nodes"] = []
            state["node_xy"] = {}
            state["xs"] = np.array([])
            state["ys"] = np.array([])
            fig.canvas.draw_idle()
            return

        visible_nodes = sorted(visible_node_set)

        # Recompute spring layout for the currently visible (filtered) graph
        G_filtered = build_graph(node_info, active, min_weight=min_w)
        k_val = 0.9 / max(len(G_filtered.nodes) ** 0.5, 1.0)
        pos_current: Dict[str, np.ndarray] = nx.spring_layout(
            G_filtered, weight="weight", seed=42, k=k_val, iterations=120,
        )
        node_xy = {n: pos_current[n] for n in visible_nodes if n in pos_current}

        if not node_xy:
            state["nodes"] = []
            state["node_xy"] = {}
            state["xs"] = np.array([])
            state["ys"] = np.array([])
            fig.canvas.draw_idle()
            return

        segments = []
        edge_colors_rgba = []
        edge_lwidths = []
        weight_range = max(max_weight - min_w, 1)

        for a, b, w in visible_edges:
            if a not in node_xy or b not in node_xy:
                continue
            xa, ya = node_xy[a]
            xb, yb = node_xy[b]
            segments.append([(xa, ya), (xb, yb)])
            alpha = 0.45 + (w / max_weight) * 0.55
            edge_colors_rgba.append((*edge_rgb, alpha))
            lw = 1.5 + (w - min_w) / weight_range * 6.5
            edge_lwidths.append(lw)

        if segments:
            lc = LineCollection(
                segments,
                colors=edge_colors_rgba,
                linewidths=edge_lwidths,
                zorder=1,
                capstyle="round",
            )
            ax_main.add_collection(lc)
            state["dynamic_artists"].append(lc)

        xs = np.array([node_xy[n][0] for n in visible_nodes])
        ys = np.array([node_xy[n][1] for n in visible_nodes])
        node_colors = [color_map_full.get(n, "#ffffff") for n in visible_nodes]

        sc = ax_main.scatter(
            xs, ys,
            s=28, c=node_colors, alpha=0.88, zorder=3,
            linewidths=0.5, edgecolors="#ffffff55",
        )
        state["dynamic_artists"].append(sc)

        filter_str = f"  [{filter_label}]" if classified else ""
        stats_text.set_text(
            f"{len(visible_nodes)} comments  ·  {len(segments)} connections  ·  "
            f"min shared phrases ≥ {min_w}{filter_str}"
        )

        state["annot"] = ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(18, 18),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#1e1e3e",
                edgecolor=EDGE_COLOR,
                alpha=0.96,
                linewidth=1.2,
            ),
            arrowprops=dict(arrowstyle="->", color=EDGE_COLOR, lw=1.0),
            color=TEXT_COLOR,
            fontsize=8.5,
            fontfamily="monospace",
            zorder=10,
            visible=False,
        )

        # Always reset the home view to fit the freshly computed layout
        margin = 0.08
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        xpad = max((xmax - xmin) * margin, 0.05)
        ypad = max((ymax - ymin) * margin, 0.05)
        state["home_xlim"] = (xmin - xpad, xmax + xpad)
        state["home_ylim"] = (ymin - ypad, ymax + ypad)
        ax_main.set_xlim(state["home_xlim"])
        ax_main.set_ylim(state["home_ylim"])

        state["nodes"] = visible_nodes
        state["node_xy"] = node_xy
        state["xs"] = xs
        state["ys"] = ys

        fig.canvas.draw_idle()

    # ── Slider callback ────────────────────────────────────────────────────────
    def on_slider_change(val: float) -> None:
        if state["annot"] is not None:
            state["annot"].set_visible(False)
        draw_graph(int(round(val)))

    slider.on_changed(on_slider_change)

    # ── Radio-button callback ──────────────────────────────────────────────────
    def on_radio_change(label: str) -> None:
        if label == "Slop Phrases Only":
            state["active_weights"] = edge_weights_slop
        elif label == "Topical Phrases Only":
            state["active_weights"] = edge_weights_topic
        else:
            state["active_weights"] = edge_weights_all
        if state["annot"] is not None:
            state["annot"].set_visible(False)
        draw_graph(int(round(slider.val)))

    radio.on_clicked(on_radio_change)

    # ── Scroll-wheel zoom ──────────────────────────────────────────────────────
    def on_scroll(event) -> None:
        if event.inaxes is not ax_main:
            return
        scale = 1.18 if event.button == "down" else 1.0 / 1.18
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()
        ax_main.set_xlim([xdata + (x - xdata) * scale for x in xlim])
        ax_main.set_ylim([ydata + (y - ydata) * scale for y in ylim])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # ── Key handler (R = reset view) ───────────────────────────────────────────
    def on_key(event) -> None:
        if event.key in ("r", "R") and state["home_xlim"] is not None:
            ax_main.set_xlim(state["home_xlim"])
            ax_main.set_ylim(state["home_ylim"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── Press / release handlers (enable click-drag panning) ──────────────────
    def on_press(event) -> None:
        if event.inaxes is not ax_main or event.button != 1:
            return
        state["pan_start_px"] = (event.x, event.y)
        state["is_panning"] = False

    def on_release(event) -> None:
        if event.button == 1:
            state["pan_start_px"] = None
            state["is_panning"] = False

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # ── Hover / pan handler ────────────────────────────────────────────────────
    def on_hover(event) -> None:
        # ── Pan while left button is held ─────────────────────────────────────
        pan_start = state["pan_start_px"]
        if pan_start is not None:
            if event.inaxes is ax_main and event.x is not None and event.y is not None:
                dx_px = event.x - pan_start[0]
                dy_px = event.y - pan_start[1]
                state["pan_start_px"] = (event.x, event.y)  # roll origin forward
                if abs(dx_px) > 0 or abs(dy_px) > 0:
                    state["is_panning"] = True
                    inv = ax_main.transData.inverted()
                    p0 = inv.transform((0, 0))
                    p1 = inv.transform((dx_px, dy_px))
                    dx_data = p0[0] - p1[0]
                    dy_data = p0[1] - p1[1]
                    xlim = ax_main.get_xlim()
                    ylim = ax_main.get_ylim()
                    ax_main.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
                    ax_main.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
                    annot = state.get("annot")
                    if annot and annot.get_visible():
                        annot.set_visible(False)
                    fig.canvas.draw_idle()
            return  # don't process tooltip while button is held

        # ── Tooltip on hover ───────────────────────────────────────────────────
        annot = state.get("annot")
        if annot is None:
            return

        if event.inaxes is not ax_main:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        xs = state["xs"]
        ys = state["ys"]
        nodes = state["nodes"]
        node_xy = state["node_xy"]

        if len(xs) == 0:
            return

        data_pts = np.column_stack([xs, ys])
        try:
            display_pts = ax_main.transData.transform(data_pts)
        except Exception:
            return
        cursor = np.array([event.x, event.y])
        dists = np.linalg.norm(display_pts - cursor, axis=1)

        min_idx = int(np.argmin(dists))
        changed = False
        active = state["active_weights"]

        if dists[min_idx] <= 14:
            node_id = nodes[min_idx]
            min_w = int(round(slider.val))

            connections_now = sum(
                1 for (a, b), w in active.items()
                if w >= min_w and (a == node_id or b == node_id)
            )
            total_phrase_pairs = sum(
                w for (a, b), w in active.items()
                if (a == node_id or b == node_id)
            )

            submitter = node_info.get(node_id, {}).get("submitter", "")
            submitter_line = f"  {submitter}\n" if submitter else ""
            tooltip = (
                f"  {node_id}\n"
                f"{submitter_line}"
                f"  {'─' * 36}\n"
                f"  Connections (≥ {min_w} phrases): {connections_now}\n"
                f"  Total shared phrase-pairs (active): {total_phrase_pairs}"
            )
            x_d, y_d = node_xy[node_id]
            annot.xy = (x_d, y_d)
            annot.set_text(tooltip)
            if not annot.get_visible():
                annot.set_visible(True)
                changed = True
            else:
                changed = True
        else:
            if annot.get_visible():
                annot.set_visible(False)
                changed = True

        if changed:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # ── Initial draw ───────────────────────────────────────────────────────────
    draw_graph(min(initial_min_weight, max_weight))

    filter_note = (
        "  • Radio buttons    — filter by slopical / topical / both\n"
        if classified else ""
    )
    print(
        "[phrase-network] Graph window open.\n"
        "  • Drag the slider  — adjust minimum shared-phrase threshold\n"
        f"{filter_note}"
        "  • Click and drag   — pan the view\n"
        "  • Scroll wheel     — zoom in / out (centred on cursor)\n"
        "  • Press R          — reset to full-graph view\n"
        "  • Hover over node  — see Document ID and connection stats\n"
        "  Close the window to exit."
    )
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phrase_network",
        description=(
            "Visualise comment-to-comment connections from a phrase report "
            "as an interactive 2-D network graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phrase_network.py report.md\n"
            "  python phrase_network.py report.md --min-weight 3\n"
            "  python phrase_network.py report.md --title 'HHS-ONC-2026-0001'\n"
        ),
    )
    parser.add_argument(
        "input_file",
        help="Path to a Markdown phrase report (.md) produced by phrase_report.py.",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Initial minimum shared phrases to show an edge (default: 5). "
            "Adjustable live via the slider."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title displayed above the graph.",
    )

    args = parser.parse_args()

    path = Path(args.input_file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    visualise(
        md_path=str(path),
        initial_min_weight=args.min_weight,
        title=args.title,
    )


if __name__ == "__main__":
    main()
