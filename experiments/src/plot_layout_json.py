from __future__ import annotations

import argparse
import math
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layout_tools import as_float, load_layout_contract, obb_corners_xy, read_json, room_bbox


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


def _rotation_to_rad(value: Any) -> float:
    raw = as_float(value, 0.0)
    if abs(raw) > (2.0 * math.pi + 1e-6):
        return math.radians(raw)
    return raw


def _rotation_front_unit(rotation_value: Any) -> Tuple[float, float]:
    # rotationZ semantics: local +Y (functional front) points to world direction.
    # 0:+Y, 90:+X, 180:-Y, 270:-X
    theta = _rotation_to_rad(rotation_value)
    return math.sin(theta), math.cos(theta)


def _rotation_right_unit(rotation_value: Any) -> Tuple[float, float]:
    fx, fy = _rotation_front_unit(rotation_value)
    return fy, -fx


def _corners_local_semantics(
    cx: float, cy: float, length: float, width: float, rotation_value: Any
) -> List[Tuple[float, float]]:
    hx = 0.5 * max(0.0, length)
    hy = 0.5 * max(0.0, width)

    rx, ry = _rotation_right_unit(rotation_value)   # local +X
    fx, fy = _rotation_front_unit(rotation_value)   # local +Y

    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(cx + dx * rx + dy * fx, cy + dx * ry + dy * fy) for dx, dy in local]


def _corners_world_axis(cx: float, cy: float, length: float, width: float) -> List[Tuple[float, float]]:
    hx = 0.5 * max(0.0, length)
    hy = 0.5 * max(0.0, width)
    return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]


def _extract_rooms(raw_layout: Dict[str, Any]) -> List[Tuple[str, List[Tuple[float, float]], List[Dict[str, Any]]]]:
    rooms = []
    raw_rooms = raw_layout.get("rooms")
    if not isinstance(raw_rooms, list):
        return rooms

    for room in raw_rooms:
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("room_id") or room.get("room_name") or "room")
        poly = room.get("room_polygon")
        if not isinstance(poly, list) or len(poly) < 3:
            continue

        points: List[Tuple[float, float]] = []
        for p in poly:
            if isinstance(p, dict):
                points.append((as_float(p.get("X"), 0.0), as_float(p.get("Y"), 0.0)))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((as_float(p[0], 0.0), as_float(p[1], 0.0)))
        if len(points) < 3:
            continue

        openings = room.get("openings")
        openings_list = openings if isinstance(openings, list) else []
        rooms.append((room_id, points, openings_list))
    return rooms


def _extract_outer_polygon(
    raw_layout: Dict[str, Any], normalized_layout: Dict[str, Any]
) -> List[Tuple[float, float]]:
    outer = raw_layout.get("outer_polygon")
    points: List[Tuple[float, float]] = []
    if isinstance(outer, list) and outer:
        for p in outer:
            if isinstance(p, dict):
                points.append((as_float(p.get("X"), 0.0), as_float(p.get("Y"), 0.0)))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((as_float(p[0], 0.0), as_float(p[1], 0.0)))
    if len(points) >= 3:
        return points

    room_poly = (normalized_layout.get("room") or {}).get("boundary_poly_xy") or []
    return [(as_float(p[0], 0.0), as_float(p[1], 0.0)) for p in room_poly if len(p) >= 2]


def _background_mask(arr: Any, crop_mode: str) -> Any:
    if crop_mode == "none":
        return None
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    if crop_mode == "beige":
        # Tuned for your floor-plan screenshots (light-beige background area).
        return (r > 235) & (g > 215) & (g < 245) & (b > 205) & (b < 240)
    return (r < 245) | (g < 245) | (b < 245)


def _load_background_image(path: pathlib.Path, crop_mode: str, flip_ud: bool):
    try:
        import numpy as np
        import cv2
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Background image support requires pillow + numpy + opencv-python. "
            "Run `uv sync --extra experiments`.\n"
            f"import error: {exc}"
        )

    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    mask = _background_mask(arr, crop_mode)
    if mask is not None:
        # Use largest connected mask area to avoid loose pixels at the border.
        labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if labels_n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + int(np.argmax(areas))
            x_min = int(stats[best, cv2.CC_STAT_LEFT])
            y_min = int(stats[best, cv2.CC_STAT_TOP])
            w = int(stats[best, cv2.CC_STAT_WIDTH])
            h = int(stats[best, cv2.CC_STAT_HEIGHT])
            x_max = x_min + w - 1
            y_max = y_min + h - 1
            arr = arr[y_min : y_max + 1, x_min : x_max + 1, :]
        else:
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                arr = arr[y_min : y_max + 1, x_min : x_max + 1, :]
    if flip_ud:
        arr = np.flipud(arr)
    return arr


def _parse_bg_extent(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("--bg_extent must be: min_x,max_x,min_y,max_y")
    min_x = as_float(parts[0], 0.0)
    max_x = as_float(parts[1], 0.0)
    min_y = as_float(parts[2], 0.0)
    max_y = as_float(parts[3], 0.0)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("--bg_extent requires max_x>min_x and max_y>min_y")
    return min_x, max_x, min_y, max_y


def _find_object_front_tip(
    raw_layout: Dict[str, Any],
    object_name: str,
    arrow_scale: float,
) -> Tuple[float, float] | None:
    objs = raw_layout.get("area_objects_list")
    if not isinstance(objs, list):
        return None
    needle = object_name.strip().lower()
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("object_name") or obj.get("id") or "").strip().lower()
        if name != needle and needle not in name:
            continue
        x = as_float(obj.get("X"), 0.0)
        y = as_float(obj.get("Y"), 0.0)
        length = as_float(obj.get("Length"), 1.0)
        width = as_float(obj.get("Width"), 1.0)
        fx, fy = _rotation_front_unit(obj.get("rotationZ", 0.0))
        arrow_len = max(0.1, min(length, width) * max(0.05, arrow_scale))
        return x + fx * arrow_len, y + fy * arrow_len
    return None


def _load_json_optional(path_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_text:
        return None
    path = pathlib.Path(path_text)
    if not path.exists():
        return None
    data = read_json(path)
    if isinstance(data, dict):
        return data
    return None


def _draw_metrics_overlay(ax, metrics: Dict[str, Any]) -> None:
    ordered_keys = [
        "C_vis",
        "C_vis_start",
        "R_reach",
        "clr_min",
        "Delta_layout",
        "OOE_enabled",
        "OOE_tau_p",
        "OOE_tau_v",
        "OOE_C_obj_entry_hit",
        "OOE_R_rec_entry_hit",
        "OOE_C_obj_entry_surf",
        "OOE_R_rec_entry_surf",
        "validity",
        "Adopt",
    ]
    lines = ["metrics"]
    for key in ordered_keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        try:
            value_f = float(value)
            lines.append(f"{key}: {value_f:.4f}")
        except Exception:
            lines.append(f"{key}: {value}")

    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "#222222", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
        zorder=20,
    )


def _extract_anchor_xy(anchor_entry: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(anchor_entry, dict):
        return None
    xy = anchor_entry.get("xy")
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    return (as_float(xy[0], 0.0), as_float(xy[1], 0.0))


def _extract_path_xy(path_json: Optional[Dict[str, Any]]) -> List[Tuple[float, float]]:
    if not isinstance(path_json, dict):
        return []

    raw_xy = path_json.get("path_xy")
    if isinstance(raw_xy, list):
        out_xy: List[Tuple[float, float]] = []
        for p in raw_xy:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out_xy.append((as_float(p[0], 0.0), as_float(p[1], 0.0)))
        if len(out_xy) >= 2:
            return out_xy

    path_cells = path_json.get("path_cells")
    bounds = path_json.get("bounds")
    resolution = as_float(path_json.get("resolution"), 0.0)
    if isinstance(path_cells, list) and isinstance(bounds, (list, tuple)) and len(bounds) >= 4 and resolution > 1e-9:
        min_x = as_float(bounds[0], 0.0)
        min_y = as_float(bounds[1], 0.0)
        out_xy: List[Tuple[float, float]] = []
        for cell in path_cells:
            if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                continue
            ix = int(as_float(cell[0], 0.0))
            iy = int(as_float(cell[1], 0.0))
            out_xy.append((min_x + (ix + 0.5) * resolution, min_y + (iy + 0.5) * resolution))
        if len(out_xy) >= 2:
            return out_xy

    return []


def _draw_task_points_overlay(ax, task_points: Dict[str, Any], path_xy: Optional[List[Tuple[float, float]]] = None) -> None:
    anchors = task_points.get("anchors")
    if not isinstance(anchors, dict):
        anchors = {}

    anchor_styles = {
        "c": {"color": "#7e57c2", "marker": "X"},
        "t": {"color": "#fb8c00", "marker": "D"},
        "s0": {"color": "#00897b", "marker": "P"},
        "s": {"color": "#2e7d32", "marker": "o"},
        "g_bed": {"color": "#c62828", "marker": "^"},
    }

    for key, style in anchor_styles.items():
        xy = _extract_anchor_xy(anchors.get(key))
        if xy is None:
            continue
        ax.scatter([xy[0]], [xy[1]], s=42, c=style["color"], marker=style["marker"], zorder=30)
        ax.text(
            xy[0],
            xy[1],
            f" {key}",
            color=style["color"],
            fontsize=8,
            ha="left",
            va="center",
            zorder=31,
        )

    # Draw snapped start/goal.
    start = task_points.get("start") if isinstance(task_points.get("start"), dict) else {}
    goal = task_points.get("goal") if isinstance(task_points.get("goal"), dict) else {}
    start_xy = _extract_anchor_xy({"xy": start.get("xy")}) if start else None
    goal_xy = _extract_anchor_xy({"xy": goal.get("xy")}) if goal else None

    if start_xy is not None:
        ax.scatter([start_xy[0]], [start_xy[1]], s=60, c="#00aa00", marker="o", edgecolors="#ffffff", linewidths=0.8, zorder=32)
        ax.text(start_xy[0], start_xy[1], " start", color="#006400", fontsize=8, ha="left", va="bottom", zorder=33)
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=60, c="#d32f2f", marker="*", edgecolors="#ffffff", linewidths=0.8, zorder=32)
        ax.text(goal_xy[0], goal_xy[1], " goal", color="#8b0000", fontsize=8, ha="left", va="bottom", zorder=33)
    if isinstance(path_xy, list) and len(path_xy) >= 2:
        ax.plot(
            [p[0] for p in path_xy],
            [p[1] for p in path_xy],
            linestyle="-",
            linewidth=1.3,
            color="#1565c0",
            alpha=0.85,
            zorder=29,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick plot for layout JSON")
    parser.add_argument("--layout", required=True, help="Path to layout JSON (extension or layout contract)")
    parser.add_argument("--out", default=None, help="Output image path (e.g. plot.png). If omitted, use interactive view.")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no_labels", action="store_true", help="Disable object labels")
    parser.add_argument("--include_floor", action="store_true", help="Include floor object in object plotting")
    parser.add_argument("--arrow_scale", type=float, default=0.35, help="Front-arrow length scale vs min(Length, Width)")
    parser.add_argument("--grid_step", type=float, default=1.0, help="Grid step in meters")

    parser.add_argument("--bg_image", default=None, help="Optional floor-plan image path")
    parser.add_argument(
        "--bg_crop_mode",
        choices=["none", "beige", "nonwhite"],
        default="none",
        help="Background auto-crop mode",
    )
    parser.add_argument("--bg_no_flip", action="store_true", help="Do not vertically flip background image")
    parser.add_argument("--bg_alpha", type=float, default=0.85, help="Background alpha [0..1]")
    parser.add_argument("--bg_extent", default=None, help="Manual background extent: min_x,max_x,min_y,max_y")
    parser.add_argument(
        "--bg_top_from_object",
        default=None,
        help="Use front-arrow tip Y of the object (object_name/id) as background max_y",
    )
    parser.add_argument("--metrics_json", default=None, help="Optional metrics.json path for in-figure metric overlay")
    parser.add_argument("--task_points_json", default=None, help="Optional task_points.json path for anchor overlay (s0,s,t,c,g_bed)")
    parser.add_argument("--path_json", default=None, help="Optional path_cells.json path for A* path overlay")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required. Run `uv sync --extra experiments`.\n"
            f"import error: {exc}"
        )

    layout_path = pathlib.Path(args.layout)
    raw_layout = read_json(layout_path)
    layout = load_layout_contract(layout_path)
    metrics_json = _load_json_optional(args.metrics_json)
    task_points_json = _load_json_optional(args.task_points_json)
    path_json_path = args.path_json
    if not path_json_path and args.task_points_json:
        task_points_path = pathlib.Path(args.task_points_json)
        sibling = task_points_path.with_name("path_cells.json")
        if sibling.exists():
            path_json_path = str(sibling)
    path_json = _load_json_optional(path_json_path)
    path_xy = _extract_path_xy(path_json)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
    ax.set_aspect("equal", adjustable="box")

    outer_poly = _extract_outer_polygon(raw_layout, layout)
    room_polys = _extract_rooms(raw_layout)

    if args.bg_image:
        bg = _load_background_image(
            pathlib.Path(args.bg_image),
            crop_mode=args.bg_crop_mode,
            flip_ud=(not args.bg_no_flip),
        )
        area_x = as_float(raw_layout.get("area_size_X"), 0.0)
        area_y = as_float(raw_layout.get("area_size_Y"), 0.0)
        if area_x > 0.0 and area_y > 0.0:
            min_x, min_y, max_x, max_y = 0.0, 0.0, area_x, area_y
        elif len(outer_poly) >= 3:
            min_x, min_y, max_x, max_y = room_bbox(outer_poly)
        else:
            base_poly = (layout.get("room") or {}).get("boundary_poly_xy") or []
            min_x, min_y, max_x, max_y = room_bbox(base_poly)

        if args.bg_extent:
            ex_min_x, ex_max_x, ex_min_y, ex_max_y = _parse_bg_extent(args.bg_extent)
            min_x, max_x, min_y, max_y = ex_min_x, ex_max_x, ex_min_y, ex_max_y
        elif args.bg_top_from_object:
            tip = _find_object_front_tip(raw_layout, args.bg_top_from_object, args.arrow_scale)
            if tip is not None:
                max_y = tip[1]

        ax.imshow(bg, extent=[min_x, max_x, min_y, max_y], origin="lower", alpha=max(0.0, min(1.0, args.bg_alpha)), zorder=0)

    if len(outer_poly) >= 3:
        ox = [p[0] for p in outer_poly] + [outer_poly[0][0]]
        oy = [p[1] for p in outer_poly] + [outer_poly[0][1]]
        ax.plot(ox, oy, color="#111111", linewidth=2.2, label="outer boundary", zorder=3)

    for room_id, poly, openings in room_polys:
        rx = [p[0] for p in poly] + [poly[0][0]]
        ry = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(rx, ry, color="#222222", linestyle="--", linewidth=1.5, zorder=3)
        cx = sum(p[0] for p in poly) / max(1, len(poly))
        cy = sum(p[1] for p in poly) / max(1, len(poly))
        ax.text(cx, cy, room_id, fontsize=7, ha="center", va="center", color="#222222", zorder=4)

        for opening in openings:
            if not isinstance(opening, dict):
                continue
            ox = as_float(opening.get("X"), 0.0)
            oy = as_float(opening.get("Y"), 0.0)
            typ = str(opening.get("type") or "opening")
            ax.plot([ox], [oy], marker="o", markersize=3.0, color="#444444", zorder=5)
            ax.text(ox, oy, typ, fontsize=6, ha="left", va="bottom", color="#444444", zorder=5)

    legend_once: Dict[str, bool] = {}
    size_mode = str(raw_layout.get("size_mode") or "world").strip().lower()
    ext_objs = raw_layout.get("area_objects_list")
    use_extension_objects = isinstance(ext_objs, list) and len(ext_objs) > 0

    if use_extension_objects:
        for idx, raw_obj in enumerate(ext_objs):
            if not isinstance(raw_obj, dict):
                continue
            category = str(raw_obj.get("category") or "object")
            if not args.include_floor and category.lower() == "floor":
                continue
            color = _category_color(category)
            label = str(raw_obj.get("object_name") or raw_obj.get("id") or f"obj_{idx:02d}")

            x = as_float(raw_obj.get("X"), 0.0)
            y = as_float(raw_obj.get("Y"), 0.0)
            length = as_float(raw_obj.get("Length"), 1.0)
            width = as_float(raw_obj.get("Width"), 1.0)
            rotation = raw_obj.get("rotationZ", 0.0)

            if size_mode == "local":
                corners = _corners_local_semantics(x, y, length, width, rotation)
            else:
                # world mode keeps Length/Width as world-axis dimensions.
                corners = _corners_world_axis(x, y, length, width)

            patch = Polygon(
                corners,
                closed=True,
                facecolor=color,
                edgecolor="#222222",
                alpha=0.42,
                linewidth=1.0,
                label=category if not legend_once.get(category) else None,
                zorder=2,
            )
            legend_once[category] = True
            ax.add_patch(patch)

            fx, fy = _rotation_front_unit(rotation)
            arrow_len = max(0.1, min(length, width) * max(0.05, args.arrow_scale))
            ax.annotate(
                "",
                xy=(x + fx * arrow_len, y + fy * arrow_len),
                xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 1.2},
                zorder=6,
            )
            ax.plot([x], [y], marker="o", markersize=2.5, color="#111111", zorder=6)

            if not args.no_labels:
                ax.text(x, y, label, fontsize=6.5, ha="center", va="center", color="#111111", zorder=6)
    else:
        for obj in layout.get("objects", []):
            obj_id = str(obj.get("id") or "")
            category = str(obj.get("category") or "object")
            if not args.include_floor and category.lower() == "floor":
                continue
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
                zorder=2,
            )
            legend_once[category] = True
            ax.add_patch(patch)

            pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
            size = obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
            x = as_float(pose[0] if len(pose) > 0 else 0.0, 0.0)
            y = as_float(pose[1] if len(pose) > 1 else 0.0, 0.0)
            yaw = _rotation_to_rad(pose[3] if len(pose) > 3 else 0.0)
            length = as_float(size[0] if len(size) > 0 else 1.0, 1.0)
            width = as_float(size[1] if len(size) > 1 else 1.0, 1.0)

            # Contract yaw assumes local +X orientation; local +Y (front) is 90 deg CCW from +X.
            fx, fy = -math.sin(yaw), math.cos(yaw)
            arrow_len = max(0.1, min(length, width) * max(0.05, args.arrow_scale))
            ax.annotate(
                "",
                xy=(x + fx * arrow_len, y + fy * arrow_len),
                xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 1.2},
                zorder=6,
            )
            ax.plot([x], [y], marker="o", markersize=2.0, color="#111111", zorder=6)
            if not args.no_labels and obj_id:
                ax.text(x, y, obj_id, fontsize=7, ha="center", va="center", zorder=6)

    if len(outer_poly) >= 3:
        min_x, min_y, max_x, max_y = room_bbox(outer_poly)
    else:
        base_poly = (layout.get("room") or {}).get("boundary_poly_xy") or []
        min_x, min_y, max_x, max_y = room_bbox(base_poly)
    pad = 0.05 * max(max_x - min_x, max_y - min_y, 1.0)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    if args.grid_step > 0.0:
        import numpy as np

        xs = np.arange(math.floor(min_x), math.ceil(max_x) + args.grid_step, args.grid_step)
        ys = np.arange(math.floor(min_y), math.ceil(max_y) + args.grid_step, args.grid_step)
        ax.set_xticks(xs)
        ax.set_yticks(ys)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    title = args.title or str(raw_layout.get("area_name") or layout_path.stem)
    ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    if task_points_json:
        _draw_task_points_overlay(ax, task_points_json, path_xy=path_xy)
    if metrics_json:
        _draw_metrics_overlay(ax, metrics_json)

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
