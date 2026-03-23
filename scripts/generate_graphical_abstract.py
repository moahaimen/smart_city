from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "results" / "logs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "results" / "logs" / "cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


FIG_DPI = 1000
OUTPUT_DIR = ROOT / "results" / "figures"
PNG_PATH = OUTPUT_DIR / "graphical_abstract.png"
PDF_PATH = OUTPUT_DIR / "graphical_abstract.pdf"


COLORS = {
    "bg": "#f6f3ee",
    "ink": "#1f1f1f",
    "muted": "#5d6670",
    "line": "#2c3e50",
    "left_fill": "#dff0f9",
    "left_edge": "#4c91b8",
    "center_fill": "#fff0df",
    "center_edge": "#db8b2f",
    "right_fill": "#eaf5e3",
    "right_edge": "#72a85a",
    "banner_fill": "#1f354a",
    "banner_text": "#ffffff",
    "hazard": "#c9252d",
    "routine": "#2c78b7",
    "priority": "#2a8c49",
    "accent": "#ef8f00",
    "soft_orange": "#ffd6a6",
    "soft_blue": "#c9e8fa",
    "soft_green": "#d9f0cc",
}


def add_shadowed_panel(ax, x: float, y: float, w: float, h: float, face: str, edge: str, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.5, y - 0.5),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=2.2",
            linewidth=0.0,
            facecolor="#d7d1c8",
            alpha=0.55,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=2.2",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    ax.text(x + 1.6, y + h - 2.2, title, ha="left", va="center", fontsize=18, fontweight="bold", color=COLORS["ink"], zorder=3)


def add_chip(ax, x: float, y: float, w: float, h: float, text: str, face: str, text_color: str = "white") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=1.0",
            linewidth=0.0,
            facecolor=face,
            zorder=6,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_color, zorder=7)


def add_metric_badge(ax, x: float, y: float, w: float, h: float, title: str, value: str, note: str, face: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.28,rounding_size=1.2",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
            zorder=4,
        )
    )
    ax.text(x + 1.0, y + h - 1.1, title, ha="left", va="center", fontsize=10.5, fontweight="bold", color=COLORS["ink"], zorder=5)
    ax.text(x + 1.0, y + h / 2 - 0.1, value, ha="left", va="center", fontsize=18, fontweight="bold", color=edge, zorder=5)
    ax.text(x + 1.0, y + 0.9, note, ha="left", va="bottom", fontsize=8.4, color=COLORS["muted"], zorder=5)


def arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = COLORS["line"], lw: float = 2.2, rad: float = 0.0, style: str = "-|>") -> None:
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=18,
        linewidth=lw,
        color=color,
        zorder=8,
    )
    ax.add_patch(patch)


def draw_city(ax, x: float, y: float) -> None:
    widths = [2.0, 1.6, 2.4, 1.8, 2.8, 1.7]
    heights = [6.0, 8.0, 10.2, 7.4, 9.0, 5.8]
    cursor = x
    for width, height in zip(widths, heights):
        ax.add_patch(Rectangle((cursor, y), width, height, facecolor="#6f8796", edgecolor="none", zorder=4))
        for i in range(int(width * 2)):
            for j in range(int(height // 1.7)):
                wx = cursor + 0.25 + i * 0.45
                wy = y + 0.5 + j * 1.2
                if wx < cursor + width - 0.25 and wy < y + height - 0.45:
                    ax.add_patch(Rectangle((wx, wy), 0.18, 0.28, facecolor="#f7f0b2", edgecolor="none", zorder=5))
        cursor += width + 0.45
    ax.add_patch(Ellipse((x + 7.5, y + 12.8), 12.0, 5.2, facecolor="#e5e0d4", edgecolor="#c3bcb3", linewidth=1.0, zorder=3))
    ax.add_patch(Ellipse((x + 10.8, y + 12.4), 6.8, 3.4, facecolor="#d4cdc3", edgecolor="none", zorder=3))


def draw_timeseries(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.8", linewidth=0.8, edgecolor="#8aa7bb", facecolor="white", zorder=4))
    ax.plot([x + 0.8, x + 0.8], [y + 0.8, y + h - 0.8], color="#9aa8b3", linewidth=0.8, zorder=5)
    ax.plot([x + 0.8, x + w - 0.6], [y + 0.8, y + 0.8], color="#9aa8b3", linewidth=0.8, zorder=5)
    xs = [x + 1.0, x + 2.0, x + 3.0, x + 4.1, x + 5.0, x + 6.0, x + 7.1]
    ys = [y + 1.2, y + 1.5, y + 1.1, y + 2.3, y + 2.9, y + 4.0, y + 4.7]
    ax.plot(xs, ys, color=COLORS["hazard"], linewidth=2.0, zorder=6)
    ax.fill_between(xs, ys, y + 0.8, color="#f9d3d6", alpha=0.55, zorder=5)
    ax.text(x + 0.8, y + h + 0.45, "PM2.5 dynamics", fontsize=8.8, color=COLORS["muted"], ha="left", va="bottom", zorder=6)


def draw_center_network(ax, x: float, y: float) -> None:
    centers = {
        "sink": (x + 2.5, y + 4.9),
        "ch1": (x + 9.2, y + 5.0),
        "n1": (x + 15.3, y + 7.6),
        "n2": (x + 18.7, y + 3.8),
        "ch2": (x + 25.0, y + 4.5),
        "n3": (x + 28.2, y + 7.4),
        "n4": (x + 28.2, y + 2.1),
    }

    for key, (cx, cy) in centers.items():
        ring = key.startswith("ch")
        outer = COLORS["accent"] if ring else "#284d74"
        inner = "#ffd39a" if ring else COLORS["soft_blue"]
        ax.add_patch(Circle((cx, cy), 1.25 if ring else 0.9, fill=False if ring else True, edgecolor=outer, linewidth=2.0 if ring else 1.0, facecolor="none" if ring else inner, zorder=5))
        if ring:
            ax.add_patch(Circle((cx, cy), 0.72, facecolor=inner, edgecolor="black", linewidth=0.8, zorder=6))
        else:
            ax.add_patch(Circle((cx, cy), 0.72, facecolor=inner, edgecolor="black", linewidth=0.8, zorder=6))

    ax.text(centers["sink"][0] - 0.2, centers["sink"][1] - 1.5, "Sink", fontsize=8.6, color=COLORS["line"], ha="center", zorder=7)

    # Hazardous forwarding
    arrow(ax, centers["sink"][0] + 1.0, centers["sink"][1] + 0.1, centers["ch1"][0] - 1.1, centers["ch1"][1] + 0.1, color=COLORS["hazard"], lw=2.2)
    arrow(ax, centers["ch1"][0] + 1.0, centers["ch1"][1] + 0.6, centers["n1"][0] - 1.0, centers["n1"][1] + 0.1, color=COLORS["hazard"], lw=2.2)
    arrow(ax, centers["n1"][0] + 0.8, centers["n1"][1], centers["ch2"][0] - 1.0, centers["ch2"][1] + 0.2, color=COLORS["hazard"], lw=2.2)

    # Routine traffic
    arrow(ax, centers["ch1"][0] + 1.0, centers["ch1"][1] - 0.2, centers["n2"][0] - 0.8, centers["n2"][1] + 0.2, color=COLORS["routine"], lw=1.7)
    arrow(ax, centers["n2"][0] + 0.8, centers["n2"][1], centers["ch2"][0] - 1.0, centers["ch2"][1] - 0.1, color=COLORS["routine"], lw=1.7)
    arrow(ax, centers["ch2"][0] + 1.0, centers["ch2"][1] + 0.1, centers["n3"][0] - 0.9, centers["n3"][1], color=COLORS["routine"], lw=1.7)
    arrow(ax, centers["ch2"][0] + 0.9, centers["ch2"][1] - 0.3, centers["n4"][0] - 0.9, centers["n4"][1] + 0.1, color=COLORS["routine"], lw=1.7)

    ax.text(x + 7.8, y + 8.7, "Hazardous", fontsize=8.7, color=COLORS["hazard"], ha="center", fontweight="bold", zorder=7)
    ax.text(x + 22.7, y + 8.7, "Routine", fontsize=8.7, color=COLORS["routine"], ha="center", fontweight="bold", zorder=7)


def build_graphical_abstract() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7.8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    ax.text(
        50,
        67.5,
        "TCN-PPA-LEACH for Smart-City Air-Pollution Monitoring",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
    )

    add_shadowed_panel(ax, 3, 12, 26, 51, COLORS["left_fill"], COLORS["left_edge"], "Monitoring Challenge")
    add_shadowed_panel(ax, 36, 12, 31, 51, COLORS["center_fill"], COLORS["center_edge"], "Proposed Method")
    add_shadowed_panel(ax, 71, 12, 26, 51, COLORS["right_fill"], COLORS["right_edge"], "Key Outcomes")

    # Left panel
    draw_city(ax, 6.0, 42.5)
    ax.text(16.0, 56.2, "Dynamic urban PM2.5 hotspots", ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["ink"])
    draw_timeseries(ax, 6.4, 33.7, 9.0, 6.2)
    add_chip(ax, 18.0, 36.0, 4.7, 1.8, "Normal", "#7ebf6a")
    add_chip(ax, 18.0, 33.7, 5.2, 1.8, "Warning", "#f2b445")
    add_chip(ax, 18.0, 31.4, 6.2, 1.8, "Hazardous", COLORS["hazard"])
    ax.text(6.2, 28.6, "Inputs", fontsize=12.2, fontweight="bold", color=COLORS["ink"])
    ax.text(
        6.2,
        26.6,
        "\n".join(
            [
                "- Real CSV or deterministic synthetic fallback",
                "- Multivariate pollution windowing",
                "- Severity-aware smart-city WSN scenarios",
                "- Baselines: LEACH and EA-LEACH",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.1,
        color=COLORS["ink"],
        linespacing=1.45,
    )

    # Center panel
    ax.add_patch(FancyBboxPatch((39.2, 46.2), 24.6, 10.5, boxstyle="round,pad=0.25,rounding_size=1.6", linewidth=1.0, edgecolor="#c99663", facecolor="#fff8f0", zorder=3))
    ax.text(51.5, 54.6, "1. Forecast next-step PM2.5", ha="center", va="center", fontsize=12.2, fontweight="bold", color=COLORS["ink"])
    ax.text(51.5, 50.8, "Lightweight TCN over 12-step multivariate windows", ha="center", va="center", fontsize=9.8, color=COLORS["muted"])

    ax.add_patch(FancyBboxPatch((39.2, 33.2), 24.6, 10.5, boxstyle="round,pad=0.25,rounding_size=1.6", linewidth=1.0, edgecolor="#c99663", facecolor="#fff8f0", zorder=3))
    ax.text(51.5, 41.6, "2. Compute node priority", ha="center", va="center", fontsize=12.2, fontweight="bold", color=COLORS["ink"])
    add_chip(ax, 41.0, 36.0, 5.2, 2.0, "Current", COLORS["hazard"])
    add_chip(ax, 47.0, 36.0, 6.0, 2.0, "Predicted", "#69b95a")
    add_chip(ax, 53.8, 36.0, 7.8, 2.0, "AoI + Trend", COLORS["routine"])
    ax.text(51.5, 34.0, "+ hotspot - communication cost", ha="center", va="center", fontsize=9.0, color=COLORS["muted"])

    ax.add_patch(FancyBboxPatch((39.2, 18.2), 24.6, 11.8, boxstyle="round,pad=0.25,rounding_size=1.6", linewidth=1.0, edgecolor="#c99663", facecolor="#fff8f0", zorder=3))
    ax.text(51.5, 27.9, "3. Priority-aware LEACH routing", ha="center", va="center", fontsize=12.2, fontweight="bold", color=COLORS["ink"])
    ax.text(51.5, 24.7, "Energy-aware cluster-head election + priority routing", ha="center", va="center", fontsize=9.6, color=COLORS["muted"])
    draw_center_network(ax, 40.2, 12.4)

    arrow(ax, 51.5, 46.2, 51.5, 43.9, color=COLORS["center_edge"], lw=2.0)
    arrow(ax, 51.5, 33.2, 51.5, 30.6, color=COLORS["center_edge"], lw=2.0)

    # Right panel
    ax.text(74.0, 56.0, "Main evaluation", fontsize=12.3, fontweight="bold", color=COLORS["ink"], ha="left")
    ax.text(74.0, 53.9, "10 seeds x 4 scenarios x 3 protocols", fontsize=10.6, color=COLORS["muted"], ha="left")
    ax.text(74.0, 51.9, "Representative numbers from Hazardous Spike scenario", fontsize=9.0, color=COLORS["muted"], ha="left")

    add_metric_badge(
        ax, 74.0, 42.2, 20.0, 7.6,
        "Hazardous success", "1.00",
        "vs LEACH 0.91, EA-LEACH 0.83",
        "#ffffff", COLORS["priority"]
    )
    add_metric_badge(
        ax, 74.0, 33.4, 20.0, 7.6,
        "Average AoI (rounds)", "0.37",
        "vs LEACH 17.51, EA-LEACH 4.46",
        "#ffffff", COLORS["routine"]
    )
    add_metric_badge(
        ax, 74.0, 24.6, 9.4, 7.6,
        "Delay", "1.16",
        "hops",
        "#ffffff", COLORS["center_edge"]
    )
    add_metric_badge(
        ax, 84.6, 24.6, 9.4, 7.6,
        "FND", "259",
        "rounds",
        "#ffffff", COLORS["hazard"]
    )
    ax.text(84.0, 18.0, "Validated with ablations and fairness logging", ha="center", va="center", fontsize=9.1, color=COLORS["muted"])
    ax.text(84.0, 15.3, "Fresher critical updates, longer stability", ha="center", va="center", fontsize=10.0, color=COLORS["ink"], fontweight="bold")

    # Flow arrows between panels
    arrow(ax, 29.8, 37.5, 35.0, 37.5, color=COLORS["line"], lw=2.6)
    arrow(ax, 67.8, 37.5, 70.0, 37.5, color=COLORS["line"], lw=2.6)

    # Bottom banner
    ax.add_patch(
        FancyBboxPatch(
            (7.0, 3.2),
            86.0,
            5.4,
            boxstyle="round,pad=0.35,rounding_size=1.6",
            linewidth=0.0,
            facecolor=COLORS["banner_fill"],
            zorder=2,
        )
    )
    ax.text(
        50.0,
        5.9,
        "Prediction-guided routing improves hazardous-event freshness and delivery reliability in smart-city IoT pollution monitoring.",
        ha="center",
        va="center",
        fontsize=11.6,
        color=COLORS["banner_text"],
        fontweight="bold",
        zorder=3,
    )

    fig.tight_layout(pad=0.15)
    fig.savefig(PNG_PATH, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return PNG_PATH, PDF_PATH


def main() -> None:
    png_path, pdf_path = build_graphical_abstract()
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
