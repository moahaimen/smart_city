from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "results" / "logs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "results" / "logs" / "cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


FIG_DPI = 1000
OUTPUT_DIR = ROOT / "results" / "figures"
PNG_PATH = OUTPUT_DIR / "project_workflow_figure.png"
PDF_PATH = OUTPUT_DIR / "project_workflow_figure.pdf"


PALETTE = {
    "bg": "#f7f7f7",
    "text": "#1f1f1f",
    "muted": "#46515c",
    "blue_edge": "#5a9bc6",
    "blue_fill": "#deeff9",
    "blue_header": "#2f94c2",
    "orange_edge": "#cd8a34",
    "orange_fill": "#fff0dd",
    "orange_header": "#f28a00",
    "green": "#6bc35e",
    "red": "#d54c2e",
    "cyan": "#3ea0d1",
    "box_edge": "#8091a0",
    "shadow": "#c8c8c8",
    "network_edge": "#9a9a9a",
}


def rounded_panel(ax, x: float, y: float, w: float, h: float, face: str, edge: str, header: str, header_face: str) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.4,rounding_size=2.2",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(panel)
    header_h = 3.9
    header_box = FancyBboxPatch(
        (x + 1.8, y + h - header_h - 1.3),
        w - 3.6,
        header_h,
        boxstyle="round,pad=0.3,rounding_size=1.6",
        linewidth=0.0,
        facecolor=header_face,
    )
    ax.add_patch(header_box)
    ax.text(
        x + w / 2,
        y + h - header_h / 2 - 1.3,
        header,
        ha="center",
        va="center",
        fontsize=12.7,
        color="white",
        fontweight="bold",
    )


def info_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str = "#ffffff",
    edge: str = PALETTE["box_edge"],
    title_face: str = "#f3f3f3",
    title_size: float = 11.5,
    body_size: float = 9.0,
) -> None:
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=1.2",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(outer)
    title_h = 2.8
    ax.add_patch(Rectangle((x, y + h - title_h), w, title_h, linewidth=0.0, facecolor=title_face))
    ax.plot([x, x + w], [y + h - title_h, y + h - title_h], color=edge, linewidth=0.6)
    ax.text(x + w / 2, y + h - title_h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=PALETTE["text"])
    ax.text(x + 1.0, y + h - title_h - 0.9, body, ha="left", va="top", fontsize=body_size, color=PALETTE["text"], linespacing=1.35)


def arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = "#333333", lw: float = 1.5, ls: str = "-", shrink_a: float = 0.0, shrink_b: float = 0.0) -> None:
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12.0,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
    )
    ax.add_patch(patch)


def small_tag(ax, x: float, y: float, w: float, h: float, text: str, face: str, text_color: str = "white", edge: str | None = None) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.18,rounding_size=0.5",
        linewidth=0.0 if edge is None else 0.6,
        edgecolor=edge if edge is not None else face,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.0, color=text_color, fontweight="bold")


def node(ax, x: float, y: float, label: str, fill: str = "#1c91c0", outer: str = "#0f6d98", ring: bool = False, ring_color: str = "#f28a00") -> None:
    ax.add_patch(Circle((x, y), 2.1, facecolor=outer, edgecolor="white", linewidth=0.9, zorder=5))
    ax.add_patch(Circle((x, y + 0.15), 1.7, facecolor=fill, edgecolor="none", zorder=6))
    ax.text(x, y - 0.2, label, ha="center", va="center", fontsize=8.4, color="white", fontweight="bold", zorder=7)
    if ring:
        ax.add_patch(Circle((x, y), 2.55, fill=False, edgecolor=ring_color, linewidth=1.1, zorder=4))


def cloud(ax, x: float, y: float, w: float, h: float) -> None:
    parts = [
        (x + 9.5, y + 8.0, 16.0, 12.0),
        (x + 21.0, y + 10.4, 20.0, 15.0),
        (x + 35.0, y + 11.0, 22.0, 15.0),
        (x + 50.5, y + 10.1, 18.5, 13.5),
        (x + 61.0, y + 8.2, 14.5, 11.2),
        (x + 28.0, y + 5.2, 48.0, 13.8),
    ]
    for cx, cy, ew, eh in parts:
        ax.add_patch(Ellipse((cx, cy), ew, eh, facecolor="white", edgecolor=PALETTE["network_edge"], linewidth=1.0, zorder=1))


def legend_item(ax, x: float, y: float, color: str, text: str, dashed: bool = False) -> None:
    ax.add_patch(Rectangle((x, y), 2.3, 1.2, facecolor=color, edgecolor=color, linestyle="--" if dashed else "-", linewidth=0.8))
    ax.text(x + 3.0, y + 0.6, text, ha="left", va="center", fontsize=8.4, color=PALETTE["text"])


def build_figure() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 8.1))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    ax.text(25.0, 67.3, "Study Pipeline", ha="center", va="center", fontsize=11.8, fontweight="bold", color=PALETTE["text"])
    ax.text(73.0, 67.3, "Round-Level Algorithm", ha="center", va="center", fontsize=11.8, fontweight="bold", color=PALETTE["text"])

    rounded_panel(
        ax,
        x=4.0,
        y=24.8,
        w=41.0,
        h=39.5,
        face=PALETTE["blue_fill"],
        edge=PALETTE["blue_edge"],
        header="TCN-PPA-LEACH Workflow",
        header_face=PALETTE["blue_header"],
    )
    rounded_panel(
        ax,
        x=48.5,
        y=24.8,
        w=47.0,
        h=39.5,
        face=PALETTE["orange_fill"],
        edge=PALETTE["orange_edge"],
        header="Round-Level Protocol Logic",
        header_face=PALETTE["orange_header"],
    )

    info_box(
        ax,
        x=7.5,
        y=45.1,
        w=15.8,
        h=12.1,
        title="Input Sources",
        body=" - config and seeds\n - real CSV or synthetic mix\n - severity and network settings",
    )
    info_box(
        ax,
        x=24.7,
        y=45.1,
        w=17.0,
        h=12.1,
        title="Data Preparation",
        body=" - hotspot-aware metadata\n - multivariate windows\n - 12-step input, 1-step forecast\n - split = 70 / 15 / 15",
    )
    info_box(
        ax,
        x=7.5,
        y=33.4,
        w=15.8,
        h=9.8,
        title="TCN Training",
        body=" - lightweight PyTorch TCN\n - predict next PM2.5\n - save checkpoint + metrics",
        title_face="#eef6fb",
    )
    info_box(
        ax,
        x=24.7,
        y=33.4,
        w=17.0,
        h=9.8,
        title="Study Runner",
        body=" - main: 10 seeds, 4 scenarios,\n   3 protocols\n - ablation: 10 seeds,\n   3 scenarios, 5 variants\n - optional sensitivity sweep",
        face="#fff8ef",
        edge=PALETTE["orange_edge"],
        title_face="#ffe2ba",
        body_size=7.7,
    )
    info_box(
        ax,
        x=7.5,
        y=25.7,
        w=34.2,
        h=6.0,
        title="Outputs",
        body=" - tables and figures\n - fairness report\n - reviewer docs",
        body_size=7.8,
        title_face="#eef6fb",
    )

    arrow(ax, 23.5, 51.0, 24.4, 51.0)
    arrow(ax, 33.2, 44.7, 15.4, 43.0)
    arrow(ax, 23.5, 38.3, 24.4, 38.3)
    arrow(ax, 33.2, 33.0, 24.4, 28.9)

    info_box(
        ax,
        x=51.0,
        y=45.1,
        w=19.0,
        h=12.1,
        title="Prediction and Priority",
        body=" - 12-step input window\n - TCN next-step PM2.5 forecast",
        body_size=7.8,
    )
    small_tag(ax, 52.3, 49.0, 6.4, 2.1, "Current", PALETTE["red"])
    small_tag(ax, 59.7, 49.0, 6.9, 2.1, "Predicted", PALETTE["green"])
    small_tag(ax, 53.7, 46.2, 11.8, 1.9, "AoI + Trend + Cost", PALETTE["cyan"], text_color="white")

    info_box(
        ax,
        x=72.2,
        y=45.1,
        w=20.0,
        h=12.1,
        title="Clustering and Control",
        body=" - TCN-PPA CH score:\n   energy + priority + distance\n - choose CH or direct sink\n - advertise, join, aggregate",
        body_size=7.7,
    )
    info_box(
        ax,
        x=51.0,
        y=33.4,
        w=19.0,
        h=9.4,
        title="Transmission Policy",
        body=" - warning and hazardous always kept\n - suppress only low-value routine packets\n - order by severity -> predicted -> priority",
        body_size=7.4,
    )
    info_box(
        ax,
        x=72.2,
        y=33.4,
        w=20.0,
        h=9.4,
        title="Round Metrics",
        body=" - alive nodes and energy\n - generated / suppressed /\n   delivered packets\n - PDR, delay, AoI,\n   hazardous success",
        body_size=7.4,
    )
    info_box(
        ax,
        x=51.0,
        y=25.7,
        w=41.2,
        h=6.0,
        title="Protocol Family",
        body=" - Standard LEACH: probabilistic CH\n - EA-LEACH: energy + distance CH\n - TCN-PPA-LEACH: energy + priority + distance CH",
        body_size=7.3,
    )

    arrow(ax, 45.3, 49.8, 48.1, 49.8, lw=1.8)
    arrow(ax, 64.0, 45.7, 64.0, 43.9, color=PALETTE["orange_edge"], lw=1.2)
    arrow(ax, 82.2, 45.7, 82.2, 43.9, color=PALETTE["orange_edge"], lw=1.2)
    arrow(ax, 71.0, 39.0, 72.0, 39.0, color=PALETTE["orange_edge"], lw=1.2)

    cloud(ax, x=19.5, y=1.7, w=61.0, h=23.0)
    ax.text(50.0, 1.8, "LEACH-Family WSN", ha="center", va="center", fontsize=12.0, fontweight="bold", color=PALETTE["text"], zorder=8)

    node(ax, 41.0, 17.8, "Sink", fill="#1b87b6", outer="#0f6d98", ring=False)
    node(ax, 30.6, 11.0, "N1", ring=False)
    node(ax, 42.0, 6.0, "CH8", ring=True)
    node(ax, 53.2, 10.5, "CH3", ring=True)
    node(ax, 64.2, 10.4, "N5", ring=False)
    node(ax, 61.0, 5.0, "N11", ring=False)

    arrow(ax, 31.8, 12.2, 39.0, 16.4, color=PALETTE["red"], lw=1.9)
    arrow(ax, 43.6, 7.8, 51.2, 9.9, color=PALETTE["red"], lw=1.9)
    arrow(ax, 54.9, 11.8, 42.8, 17.0, color=PALETTE["red"], lw=1.9)
    arrow(ax, 62.2, 10.4, 55.2, 10.4, color=PALETTE["cyan"], lw=1.4, ls="--")
    arrow(ax, 59.0, 5.8, 44.5, 6.0, color=PALETTE["cyan"], lw=1.4, ls="--")
    arrow(ax, 31.8, 9.2, 40.0, 6.6, color=PALETTE["cyan"], lw=1.4, ls="--")

    ax.text(32.0, 14.6, "Hazardous packet", fontsize=9.2, color=PALETTE["red"], fontweight="bold", zorder=8)
    ax.text(53.8, 13.3, "Prioritized forwarding", fontsize=8.5, color=PALETTE["red"], zorder=8)
    ax.text(54.0, 4.4, "Routine packet", fontsize=8.5, color=PALETTE["cyan"], zorder=8)
    ax.text(23.8, 4.6, "Cluster heads use orange rings", fontsize=8.5, color=PALETTE["text"], zorder=8)

    arrow(ax, 69.0, 26.0, 61.5, 18.6, color=PALETTE["orange_edge"], lw=1.8)
    arrow(ax, 24.5, 24.6, 31.0, 17.5, color=PALETTE["blue_edge"], lw=1.7)

    ax.text(7.0, 18.0, "Scenario bundle generation\nand per-seed simulation", ha="center", va="center", fontsize=10.0, color=PALETTE["muted"])
    arrow(ax, 11.3, 20.2, 20.0, 14.6, color=PALETTE["muted"], lw=1.2)

    ax.text(88.0, 14.1, "Main outputs", ha="center", va="center", fontsize=10.8, color=PALETTE["muted"], fontweight="bold")
    ax.text(88.0, 11.0, "PDR, delay, AoI,\nenergy, hazardous success", ha="center", va="center", fontsize=8.8, color=PALETTE["muted"])
    arrow(ax, 78.0, 9.4, 84.6, 10.3, color=PALETTE["muted"], lw=1.3)

    ax.text(82.0, 6.4, "LEGEND", fontsize=10.5, fontweight="bold", color=PALETTE["text"])
    legend_item(ax, 79.5, 4.4, PALETTE["red"], "Hazardous prioritized packet")
    legend_item(ax, 79.5, 2.6, PALETTE["cyan"], "Routine packet / direct traffic", dashed=True)
    legend_item(ax, 79.5, 0.8, PALETTE["green"], "Prediction and priority logic")

    fig.tight_layout(pad=0.2)
    fig.savefig(PNG_PATH, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return PNG_PATH, PDF_PATH


def main() -> None:
    png_path, pdf_path = build_figure()
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
