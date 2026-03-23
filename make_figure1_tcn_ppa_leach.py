from __future__ import annotations

import argparse
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


PNG_NAME = "figure1_tcn_ppa_leach_architecture_final.png"
PDF_NAME = "figure1_tcn_ppa_leach_architecture_final.pdf"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def arrow(ax, start, end, color="#2B2B2B", lw=1.8, rad=0.0, z=6, scale=18):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
    )


def panel(ax, x, y, w, h, header_h, header_color, title, title_color="white", edge="#5A8BB7", body="#FFFFFF", roundness=1.6, title_size=16):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={roundness}",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=body,
        zorder=2,
    )
    ax.add_patch(box)
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - header_h),
            w,
            header_h,
            boxstyle=f"round,pad=0.0,rounding_size={roundness}",
            linewidth=0,
            facecolor=header_color,
            zorder=3,
        )
    )
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h - 0.9, facecolor=header_color, edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h - header_h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=4)
    return {"x": x, "y": y, "w": w, "h": h, "left": x + 1.8, "right": x + w - 1.8, "top": y + h - header_h - 1.5, "bottom": y + 1.8, "cx": x + w / 2, "cy": y + h / 2}


def chip(ax, x, y, w, h, text, face, color="white", size=11, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.5", facecolor=face, edgecolor="none", zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color, zorder=5)


def person(ax, x, y, scale=1.0, shirt="#D3382C", outline="#5B5B5B", z=6):
    ax.add_patch(Circle((x, y + 2.0 * scale), 0.75 * scale, facecolor="#FFFFFF", edgecolor=outline, linewidth=1.2, zorder=z))
    ax.add_patch(Ellipse((x, y + 0.2 * scale), 2.2 * scale, 2.8 * scale, facecolor=shirt, edgecolor=outline, linewidth=1.2, zorder=z))


def tree(ax, x, y, scale=1.0, z=5):
    ax.add_patch(Rectangle((x - 0.20 * scale, y - 0.8 * scale), 0.4 * scale, 1.1 * scale, facecolor="#8B5A2B", edgecolor="none", zorder=z))
    for dx, dy, r in [(-0.55, 0.15, 0.55), (0.0, 0.45, 0.68), (0.55, 0.10, 0.55)]:
        ax.add_patch(Circle((x + dx * scale, y + dy * scale), r * scale, facecolor="#6FB04A", edgecolor="none", zorder=z))


def cluster_disk(ax, x, y, w, h, fill, edge, z=3):
    ax.add_patch(Ellipse((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=z))
    ax.add_patch(Ellipse((x, y - 0.25), w, h * 0.92, facecolor="none", edgecolor=edge, linewidth=1.0, alpha=0.35, zorder=z))


def draw_sliding_window(ax, x, y, w, h):
    chip(ax, x + 0.4, y + h + 1.5, 12.5, 3.3, "Sliding Window", "#FFFFFF", color="#2B2B2B", size=12, weight="bold")
    body = Rectangle((x, y), w - 2.0, h, facecolor="#E9F5FE", edgecolor="#3A82B1", linewidth=1.6, zorder=4)
    tip = Polygon([[x + w - 2.0, y], [x + w, y + h / 2], [x + w - 2.0, y + h]], closed=True, facecolor="#79B6E0", edgecolor="#3A82B1", linewidth=1.6, zorder=4)
    ax.add_patch(body)
    ax.add_patch(tip)
    cell_w = (w - 2.2) / 7.0
    labels = [r"$t-5$", r"$t-4$", r"$t-3$", r"$t-2$", r"$t-1$", r"$t$", ""]
    dot_colors = ["#D6372A", "#2D6EA8", "#4A7899", "#4A7899", "#4A7899", "#D6372A", "#D6372A"]
    for i in range(7):
        sx = x + i * cell_w
        if i > 0:
            ax.plot([sx, sx], [y, y + h], color="#4B89B7", linewidth=1.1, zorder=5)
        ax.plot([sx + 0.9, sx + cell_w - 0.5], [y + 0.2, y + h - 0.2], color="#88B8DA", linewidth=1.0, zorder=5)
        if labels[i]:
            ax.text(sx + cell_w / 2, y + h - 1.9, labels[i], ha="center", va="center", fontsize=10, color="#2C4053", zorder=6)
        ax.add_patch(Circle((sx + cell_w / 2, y + 1.6), 0.33, facecolor=dot_colors[i], edgecolor="none", zorder=6))
    return [x + cell_w * 1.45, x + cell_w * 2.2, x + cell_w * 3.1, x + cell_w * 4.15, x + cell_w * 5.25]


def draw_monitor(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y + 2.0), w, h - 3.0, facecolor="#FAFAFA", edgecolor="#323232", linewidth=2.2, zorder=5))
    ax.add_patch(Rectangle((x, y), w, 2.0, facecolor="#B4B4B4", edgecolor="#323232", linewidth=2.0, zorder=5))
    ax.add_patch(Rectangle((x + w * 0.46, y - 0.9), w * 0.08, 0.9, facecolor="#444444", edgecolor="#444444", zorder=5))
    ax.add_patch(Rectangle((x + w * 0.34, y - 1.6), w * 0.32, 0.6, facecolor="#444444", edgecolor="#444444", zorder=5))
    ax.add_patch(Polygon([[x + 1.2, y + h - 4.8], [x + 2.3, y + h - 2.3], [x + 3.1, y + h - 2.3], [x + 3.1, y + h - 5.5], [x + 1.2, y + h - 5.5]], closed=True, facecolor="#4A92CF", edgecolor="none", zorder=6))
    ax.text(x + w * 0.68, y + h * 0.60, "AoI", ha="center", va="center", fontsize=28, color="#377FC0", fontweight="bold", zorder=6)


def draw_tower(ax, x, y, scale=1.0):
    ax.add_patch(Polygon([[x - 1.8 * scale, y - 8.0 * scale], [x, y + 4.0 * scale], [x + 1.8 * scale, y - 8.0 * scale]], closed=False, fill=False, edgecolor="#2F3A46", linewidth=4.0, zorder=5))
    for yy in [y - 5.6 * scale, y - 3.0 * scale, y - 0.6 * scale]:
        ax.plot([x - 1.2 * scale, x + 1.2 * scale], [yy, yy], color="#2F3A46", linewidth=2.4, zorder=5)
    ax.plot([x - 1.5 * scale, x + 1.5 * scale], [y - 7.2 * scale, y + 2.8 * scale], color="#2F3A46", linewidth=2.0, zorder=5)
    ax.plot([x + 1.5 * scale, x - 1.5 * scale], [y - 7.2 * scale, y + 2.8 * scale], color="#2F3A46", linewidth=2.0, zorder=5)
    ax.add_patch(Circle((x, y + 4.0 * scale), 0.9 * scale, facecolor="#2F3A46", edgecolor="none", zorder=5))
    for radius in [3.1, 4.8, 6.5]:
        ax.add_patch(Arc((x - 0.8 * scale, y + 4.0 * scale), radius * scale, radius * scale, angle=0, theta1=120, theta2=240, edgecolor="#4E667F", linewidth=2.2, zorder=4))
        ax.add_patch(Arc((x + 0.8 * scale, y + 4.0 * scale), radius * scale, radius * scale, angle=0, theta1=-60, theta2=60, edgecolor="#4E667F", linewidth=2.2, zorder=4))


def create_figure():
    configure_style()
    fig = plt.figure(figsize=(14.5, 9.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor="white", edgecolor="none", zorder=0))

    ax.text(50, 96.4, "System Architecture and Operational Workflow of the Proposed TCN-PPA-LEACH Framework", ha="center", va="center", fontsize=18, fontweight="bold", color="#2A2A2A", zorder=5)
    ax.plot([10, 90], [93, 93], color="#2F2F2F", linewidth=1.2, zorder=4)
    ax.plot([3, 97], [6, 6], color="#2F2F2F", linewidth=1.0, zorder=4)

    left = panel(ax, 5.5, 46.5, 25.0, 46.0, 5.8, "#2F6FA7", "Sensing and Input Processing", edge="#4A8ABC", title_size=15)
    center = panel(ax, 39.8, 60.0, 30.5, 31.5, 5.8, "#F28B00", "Dynamic Priority Assessment", edge="#E08B2F", body="#FFF7E9", title_size=15.5)
    right_top = panel(ax, 71.8, 67.8, 26.0, 24.0, 5.8, "#7D7D7D", "Packet Scheduling & Suppression", edge="#838383", title_size=13.2)
    right_bottom = panel(ax, 71.8, 50.5, 26.0, 18.5, 0.0, "#FFFFFF", "", title_color="#2A2A2A", edge="#707070")
    ch_box = panel(ax, 41.5, 62.0, 26.6, 13.0, 4.8, "#4F9E43", "CH Selection & Cluster Formation", edge="#4F9E43", body="#FFF9E9", title_size=13.2)

    ax.text(left["left"], 82.0, "Sensed Data:", ha="left", va="center", fontsize=15, fontweight="bold", color="#1D1D1D", zorder=5)
    bullets = [
        r"PM$_{2.5}$, PM$_{10}$, CO, NO$_2$, Temp, Humidity,",
        r"Hour_sin, Hour_cos, Hotspot Relevance",
    ]
    for i, line in enumerate(bullets):
        yy = 77.5 - i * 4.8
        ax.text(left["left"] + 0.7, yy, u"\u2022", fontsize=20, color="#2D2D2D", va="center", zorder=5)
        ax.text(left["left"] + 2.0, yy, line, fontsize=13.5, color="#1F1F1F", va="center", zorder=5)

    ax.plot([left["left"], left["right"]], [67.0, 67.0], color="#AAB7C5", linewidth=0.9, linestyle=(0, (1.5, 2.5)), zorder=4)
    dot_xs = draw_sliding_window(ax, 7.0, 59.0, 28.5, 4.0)
    for dx in dot_xs:
        arrow(ax, (dx, 59.2), (dx, 52.2), color="#2E3D4E", lw=1.7, scale=16)

    chip(ax, 7.6, 47.6, 15.2, 4.6, "TCN-Based Prediction", "#D53D2F", size=13)
    ax.add_patch(FancyBboxPatch((9.2, 41.2), 20.0, 6.0, boxstyle="round,pad=0.02,rounding_size=0.4", facecolor="#FFF4E0", edgecolor="#B69165", linewidth=1.1, zorder=4))
    ax.text(19.2, 44.2, r"Next-Step PM$_{2.5}$ Forecast", ha="center", va="center", fontsize=13.5, color="#222222", zorder=5)
    chip(ax, 8.0, 35.4, 8.5, 4.0, "Hazardous", "#D53D2F", size=10.8)
    chip(ax, 16.9, 35.4, 7.7, 4.0, "Warning", "#F59A00", size=10.8)
    chip(ax, 24.9, 35.4, 7.8, 4.0, "Routine", "#747B82", size=10.8)

    ax.add_patch(FancyBboxPatch((43.0, 75.5), 22.2, 8.6, boxstyle="round,pad=0.02,rounding_size=1.2", facecolor="#FFFDF7", edgecolor="#8D7A59", linewidth=1.1, zorder=4))
    ax.text(54.1, 80.1, "Priority Score:", ha="right", va="center", fontsize=16, fontweight="bold", color="#202020", zorder=5)
    ax.text(54.4, 80.1, r"$Sev,\,PredSev,$", ha="left", va="center", fontsize=16, color="#202020", zorder=5)
    ax.text(54.1, 76.7, r"$AoI, \Delta\,Rate, Hotspot, Cost$", ha="center", va="center", fontsize=15, color="#202020", zorder=5)

    ax.text(55.0, 69.6, u"\u2022", fontsize=16, color="#2F2F2F", va="center", zorder=5)
    ax.text(57.4, 69.6, "CH Election", fontsize=15, color="#1D1D1D", va="center", zorder=5)
    ax.text(55.0, 65.6, u"\u2022", fontsize=16, color="#2F2F2F", va="center", zorder=5)
    ax.text(57.4, 65.6, "Cluster Joining", fontsize=15, color="#1D1D1D", va="center", fontstyle="italic", zorder=5)
    ax.plot([46.0, 54.0], [69.6, 69.6], color="#8B8B8B", linewidth=1.0, zorder=4)
    ax.plot([63.5, 69.0], [69.6, 69.6], color="#8B8B8B", linewidth=1.0, zorder=4)
    ax.plot([46.0, 54.0], [65.6, 65.6], color="#8B8B8B", linewidth=1.0, zorder=4)
    ax.plot([63.5, 69.0], [65.6, 65.6], color="#8B8B8B", linewidth=1.0, zorder=4)

    bullet_x = right_top["x"] + 2.6
    text_x = bullet_x + 1.8
    ax.text(bullet_x, 84.3, u"\u2022", fontsize=20, color="#D32424", va="center", zorder=5)
    ax.text(text_x, 84.3, "Hazardous", fontsize=13.5, color="#D32424", fontweight="bold", va="center", zorder=5)
    ax.text(text_x + 7.5, 84.3, "Packets Priority Delivery", fontsize=13.2, color="#1F1F1F", va="center", zorder=5)
    ax.text(bullet_x, 79.2, u"\u2022", fontsize=20, color="#4D4D4D", va="center", zorder=5)
    ax.text(text_x, 79.2, "Routine Packets Suppressed", fontsize=13.2, color="#1F1F1F", va="center", zorder=5)

    ax.text(right_bottom["cx"], 66.3, "Sink-Side AoI Tracking", ha="center", va="center", fontsize=16, fontweight="bold", color="#1D1D1D", zorder=5)
    ax.plot([right_bottom["x"] + 0.1, right_bottom["right"]], [63.0, 63.0], color="#B0B0B0", linewidth=1.0, zorder=4)
    draw_monitor(ax, right_bottom["cx"] - 5.9, 45.2, 11.8, 11.8)
    ax.plot([right_bottom["x"] + 1.4, right_bottom["right"] - 1.1], [42.2, 42.2], color="#B0B0B0", linewidth=1.0, zorder=4)
    ax.plot([right_bottom["x"] + 1.4, right_bottom["right"] - 1.1], [36.0, 36.0], color="#B0B0B0", linewidth=1.0, zorder=4)
    ax.add_patch(Rectangle((right_bottom["x"] + 2.2, 39.3), 1.4, 1.8, facecolor="#F53B2C", edgecolor="#F53B2C", zorder=5))
    ax.text(right_bottom["x"] + 4.2, 40.2, r"No Update $\rightarrow$ AoI++", ha="left", va="center", fontsize=12.0, color="#1F1F1F", zorder=5)
    ax.add_patch(Rectangle((right_bottom["x"] + 2.2, 34.3), 1.4, 1.8, facecolor="#777777", edgecolor="#777777", zorder=5))
    ax.text(right_bottom["x"] + 4.2, 35.2, r"Fresh Packet $\rightarrow$ AoI = 0", ha="left", va="center", fontsize=12.0, color="#1F1F1F", zorder=5)

    arrow(ax, (30.5, 76.5), (40.3, 76.5), color="#222222", lw=1.8, scale=20)
    arrow(ax, (34.7, 61.0), (41.6, 61.0), color="#222222", lw=1.8, scale=20)
    arrow(ax, (54.8, 75.4), (54.8, 72.0), color="#333333", lw=1.8, scale=18)
    arrow(ax, (68.3, 80.0), (72.5, 80.0), color="#222222", lw=1.8, scale=20)
    arrow(ax, (84.8, 67.8), (84.8, 63.5), color="#333333", lw=1.8, scale=18)
    arrow(ax, (68.3, 60.5), (71.8, 60.5), color="#222222", lw=1.8, scale=20)
    arrow(ax, (54.8, 62.0), (54.8, 53.6), color="#333333", lw=1.8, scale=18)

    draw_tower(ax, 54.8, 42.0, scale=0.9)
    chip(ax, 47.1, 30.4, 14.0, 4.8, "Base Station (BS)", "#2F6FA7", size=14)
    ax.text(54.1, 27.3, "Sink", ha="center", va="center", fontsize=15, fontweight="bold", color="#2B2B2B", zorder=5)

    ax.add_patch(Ellipse((54.8, 22.0), 64.0, 22.5, facecolor="none", edgecolor="#2F2F2F", linewidth=1.2, linestyle=(0, (3.0, 2.5)), zorder=2))

    cluster_disk(ax, 37.5, 22.8, 16.0, 10.5, "#E7F4CB", "#6A7553")
    person(ax, 33.8, 32.2, 0.95, shirt="#D4372C")
    person(ax, 36.8, 31.2, 0.90, shirt="#FFFFFF")
    person(ax, 39.5, 32.9, 1.00, shirt="#D4372C")
    person(ax, 40.2, 28.7, 0.95, shirt="#FFFFFF")
    person(ax, 34.2, 29.3, 0.92, shirt="#D4372C")
    tree(ax, 31.0, 22.8, 1.1)
    tree(ax, 43.2, 23.0, 1.2)
    tree(ax, 35.0, 18.5, 0.9)

    cluster_disk(ax, 74.0, 22.8, 16.0, 10.5, "#EAF0F7", "#6A7480")
    person(ax, 69.8, 31.3, 0.92, shirt="#FFFFFF")
    person(ax, 72.5, 32.5, 0.88, shirt="#FFFFFF")
    person(ax, 76.8, 30.1, 0.95, shirt="#D4372C")
    person(ax, 79.5, 32.7, 0.92, shirt="#FFFFFF")
    person(ax, 74.2, 27.8, 0.95, shirt="#D4372C")
    tree(ax, 70.8, 17.8, 0.9)
    tree(ax, 79.2, 17.8, 0.9)

    cluster_disk(ax, 54.8, 13.2, 23.5, 10.0, "#FFE5A8", "#D79947")
    person(ax, 49.8, 18.0, 0.85, shirt="#D4372C")
    person(ax, 54.8, 16.6, 1.18, shirt="#D4372C")
    person(ax, 60.7, 18.0, 0.85, shirt="#D4372C")
    person(ax, 51.6, 12.5, 1.00, shirt="#FFFFFF")
    person(ax, 57.6, 12.8, 1.05, shirt="#FFFFFF")
    tree(ax, 47.6, 11.4, 0.95)
    tree(ax, 60.7, 12.0, 0.95)
    ax.add_patch(Ellipse((54.8, 12.1), 6.0, 3.0, facecolor="#2C75B8", edgecolor="#215482", linewidth=1.2, zorder=5))
    ax.add_patch(Rectangle((51.8, 12.1), 6.0, 3.0, facecolor="#2C75B8", edgecolor="#215482", linewidth=1.2, zorder=5))
    ax.add_patch(Ellipse((54.8, 15.1), 6.0, 3.0, facecolor="#468DCA", edgecolor="#215482", linewidth=1.2, zorder=5))
    person(ax, 54.8, 14.6, 1.0, shirt="#D4372C", z=6)

    arrow(ax, (43.3, 21.8), (48.5, 17.0), color="#2A2A2A", lw=2.0, scale=18)
    arrow(ax, (48.7, 17.2), (68.5, 22.0), color="#2A2A2A", lw=2.0, scale=18)
    arrow(ax, (68.5, 17.0), (60.5, 12.8), color="#2A2A2A", lw=2.0, scale=18)
    ax.text(57.2, 21.4, "Data Transmission", ha="center", va="center", fontsize=13, fontweight="bold", color="#1F1F1F", rotation=-17, zorder=5)
    arrow(ax, (54.8, 8.8), (54.8, 6.7), color="#2A2A2A", lw=2.0, scale=18)
    ax.text(54.8, 11.5, "Direct CH-to-BS Transmission", ha="center", va="center", fontsize=15, fontweight="bold", color="#1F1F1F", zorder=5)

    ax.text(16.8, 1.8, "Figure 1.", ha="left", va="center", fontsize=16, fontstyle="italic", fontweight="bold", color="#1F1F1F", zorder=5)
    ax.text(24.8, 1.8, "System architecture and operational workflow of the proposed TCN-PPA-LEACH framework.", ha="left", va="center", fontsize=15, fontstyle="italic", color="#1F1F1F", zorder=5)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the TCN-PPA-LEACH reference-style architecture figure.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Destination directory.")
    parser.add_argument("--dpi", type=int, default=1000, help="PNG export DPI.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

    fig = create_figure()
    png_path = args.output_dir / PNG_NAME
    pdf_path = args.output_dir / PDF_NAME
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
