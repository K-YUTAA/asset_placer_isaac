from __future__ import annotations

import argparse
import pathlib
from typing import Dict

from layout_tools import as_float, load_layout_contract, obb_corners_xy


def _category_color(category: str) -> str:
    # Stable, lightweight color mapping by category.
    palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    idx = abs(hash(category)) % len(palette)
    return palette[idx]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick plot for layout JSON")
    parser.add_argument("--layout", required=True, help="Path to layout JSON (extension or layout contract)")
    parser.add_argument("--out", default=None, help="Output image path (e.g. plot.png). If omitted, use interactive view.")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no_labels", action="store_true", help="Disable object id labels")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib が必要です。例: `uv sync --extra experiments`\n"
            f"import error: {exc}"
        )

    layout = load_layout_contract(pathlib.Path(args.layout))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
    ax.set_aspect("equal", adjustable="box")

    room_poly = layout["room"]["boundary_poly_xy"]
    rx = [as_float(p[0], 0.0) for p in room_poly]
    ry = [as_float(p[1], 0.0) for p in room_poly]
    if rx and ry:
        rx.append(rx[0])
        ry.append(ry[0])
        ax.plot(rx, ry, color="#111111", linewidth=2.0, label="room boundary")

    legend_once: Dict[str, bool] = {}

    for obj in layout.get("objects", []):
        obj_id = str(obj.get("id") or "")
        category = str(obj.get("category") or "object")
        color = _category_color(category)
        corners = obb_corners_xy(obj)
        if not corners:
            continue

        patch = Polygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor="#222222",
            alpha=0.45,
            linewidth=1.0,
            label=category if not legend_once.get(category) else None,
        )
        legend_once[category] = True
        ax.add_patch(patch)

        pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
        x = as_float(pose[0] if len(pose) > 0 else 0.0, 0.0)
        y = as_float(pose[1] if len(pose) > 1 else 0.0, 0.0)
        ax.plot([x], [y], marker="o", markersize=2.0, color="#111111")
        if not args.no_labels and obj_id:
            ax.text(x, y, obj_id, fontsize=7, ha="center", va="center")

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    if args.title:
        ax.set_title(args.title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        print(f"saved: {out_path}")
        return

    plt.show()


if __name__ == "__main__":
    main()
