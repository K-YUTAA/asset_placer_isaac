from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layout_tools import as_float

Grid = List[List[bool]]
Cell = Tuple[int, int]


def _grid_dims(bounds: Tuple[float, float, float, float], resolution: float) -> Tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    nx = max(1, int(math.ceil((max_x - min_x) / resolution)))
    ny = max(1, int(math.ceil((max_y - min_y) / resolution)))
    return nx, ny


def _cell_center(ix: int, iy: int, bounds: Tuple[float, float, float, float], resolution: float) -> Tuple[float, float]:
    min_x, min_y, _, _ = bounds
    return min_x + (ix + 0.5) * resolution, min_y + (iy + 0.5) * resolution


def _xy_to_cell(x: float, y: float, bounds: Tuple[float, float, float, float], resolution: float) -> Cell:
    min_x, min_y, max_x, max_y = bounds
    nx, ny = _grid_dims(bounds, resolution)

    clamped_x = min(max(x, min_x), max_x - 1e-9)
    clamped_y = min(max(y, min_y), max_y - 1e-9)
    ix = int((clamped_x - min_x) / resolution)
    iy = int((clamped_y - min_y) / resolution)
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    return ix, iy


def _polygon_centroid_xy(poly: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not poly:
        return 0.0, 0.0

    # Shoelace centroid. Falls back to vertex average if degenerate.
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)
    for i in range(n):
        x0 = as_float(poly[i][0], 0.0)
        y0 = as_float(poly[i][1], 0.0)
        x1 = as_float(poly[(i + 1) % n][0], 0.0)
        y1 = as_float(poly[(i + 1) % n][1], 0.0)
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(a) < 1e-12:
        xs = [as_float(p[0], 0.0) for p in poly]
        ys = [as_float(p[1], 0.0) for p in poly]
        return (sum(xs) / len(xs)) if xs else 0.0, (sum(ys) / len(ys)) if ys else 0.0

    a *= 0.5
    cx /= 6.0 * a
    cy /= 6.0 * a
    return cx, cy


def _obj_text(obj: Dict[str, Any]) -> str:
    return f"{obj.get('id','')} {obj.get('category','')} {obj.get('asset_query','')}".lower()


def _select_object(
    candidates: List[Dict[str, Any]],
    strategy: str,
    room_centroid: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    strategy = (strategy or "").strip().lower()

    if strategy == "first":
        return sorted(candidates, key=lambda o: str(o.get("id") or ""))[0]

    if strategy in {"largest_opening", "largest"}:

        def key(o: Dict[str, Any]) -> Tuple[float, str]:
            size = o.get("size_lwh_m") or [1.0, 1.0, 1.0]
            l = as_float(size[0] if len(size) > 0 else 1.0, 1.0)
            w = as_float(size[1] if len(size) > 1 else 1.0, 1.0)
            opening = max(l, w)
            return (opening, str(o.get("id") or ""))

        return max(candidates, key=key)

    if strategy == "closest_to_room_centroid":
        cx, cy = room_centroid

        def key(o: Dict[str, Any]) -> Tuple[float, str]:
            pose = o.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
            x = as_float(pose[0] if len(pose) > 0 else 0.0, 0.0)
            y = as_float(pose[1] if len(pose) > 1 else 0.0, 0.0)
            return (math.hypot(x - cx, y - cy), str(o.get("id") or ""))

        return min(candidates, key=key)

    # Default: deterministic.
    return sorted(candidates, key=lambda o: str(o.get("id") or ""))[0]


def _snap_cell_to_free(cell: Cell, free_mask: Grid, max_radius_cells: int) -> Tuple[Cell, bool]:
    ny = len(free_mask)
    nx = len(free_mask[0]) if ny > 0 else 0
    if nx <= 0 or ny <= 0:
        return cell, False

    ix0, iy0 = cell
    if 0 <= ix0 < nx and 0 <= iy0 < ny and free_mask[iy0][ix0]:
        return cell, False

    r = max(0, int(max_radius_cells))
    best: Optional[Cell] = None
    best_d2 = float("inf")

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 == 0 or d2 > r * r:
                continue
            ix = ix0 + dx
            iy = iy0 + dy
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                continue
            if not free_mask[iy][ix]:
                continue
            if d2 < best_d2 or (d2 == best_d2 and best is not None and (iy, ix) < (best[1], best[0])):
                best = (ix, iy)
                best_d2 = d2

    if best is None:
        return cell, False
    return best, True


def _resolve_door(layout: Dict[str, Any], start_spec: Dict[str, Any], room_centroid: Tuple[float, float]) -> Optional[Dict[str, Any]]:
    objects = [o for o in layout.get("objects", []) if isinstance(o, dict)]
    doors = [o for o in objects if "door" in _obj_text(o)]

    # Prefer sliding door if available.
    sliding = [o for o in doors if "sliding" in _obj_text(o)]
    if sliding:
        doors = sliding

    selector = start_spec.get("door_selector")
    strategy = ""
    if isinstance(selector, dict):
        strategy = str(selector.get("strategy") or "")

    return _select_object(doors, strategy or "largest_opening", room_centroid)


def _resolve_bed(layout: Dict[str, Any], goal_spec: Dict[str, Any], room_centroid: Tuple[float, float]) -> Optional[Dict[str, Any]]:
    objects = [o for o in layout.get("objects", []) if isinstance(o, dict)]
    beds = [o for o in objects if "bed" in _obj_text(o)]

    selector = goal_spec.get("bed_selector")
    strategy = ""
    if isinstance(selector, dict):
        strategy = str(selector.get("strategy") or "")

    return _select_object(beds, strategy or "first", room_centroid)


def _bedside_candidates_xy(bed_obj: Dict[str, Any], offset_m: float) -> List[Tuple[float, float]]:
    size = bed_obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
    pose = bed_obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]

    x = as_float(pose[0] if len(pose) > 0 else 0.0, 0.0)
    y = as_float(pose[1] if len(pose) > 1 else 0.0, 0.0)
    yaw = as_float(pose[3] if len(pose) > 3 else 0.0, 0.0)

    length = as_float(size[0] if len(size) > 0 else 1.0, 1.0)
    width = as_float(size[1] if len(size) > 1 else 1.0, 1.0)

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    # Local axes in world.
    x_axis = (cos_y, sin_y)      # local +X
    y_axis = (-sin_y, cos_y)     # local +Y

    # g_bed candidates are generated from the two long-side edges.
    # That means offsetting along the short-side normal.
    if length >= width:
        short_axis = y_axis
        short_half = 0.5 * width
    else:
        short_axis = x_axis
        short_half = 0.5 * length

    d = short_half + max(0.0, float(offset_m))
    sx, sy = short_axis

    return [
        (x + sx * d, y + sy * d),
        (x - sx * d, y - sy * d),
    ]


def resolve_task_points(
    layout: Dict[str, Any],
    config: Dict[str, Any],
    bounds: Tuple[float, float, float, float],
    resolution: float,
    free_mask: Grid,
) -> Dict[str, Any]:
    task = config.get("task")
    task = task if isinstance(task, dict) else {}

    room_poly = (layout.get("room") or {}).get("boundary_poly_xy") or []
    room_centroid = _polygon_centroid_xy(room_poly)

    snap_spec = task.get("snap")
    snap_spec = snap_spec if isinstance(snap_spec, dict) else {}
    max_radius_cells = int(snap_spec.get("max_radius_cells", 30))

    default_start_xy = config.get("start_xy") or [0.8, 0.8]
    default_goal_xy = config.get("goal_xy") or [5.0, 5.0]

    selectors: Dict[str, Any] = {"door_id": None, "bed_id": None}
    anchors: Dict[str, Any] = {
        "c": {"xy": [float(room_centroid[0]), float(room_centroid[1])], "label": "c"},
        "t": None,
        "s0": None,
        "s": None,
        "g_bed": None,
    }

    # Start
    start_spec = task.get("start") if isinstance(task.get("start"), dict) else {}
    start_mode = str(start_spec.get("mode") or "")
    start_xy = [as_float(default_start_xy[0], 0.8), as_float(default_start_xy[1], 0.8)]

    if start_mode == "entrance_slidingdoor_center":
        door = _resolve_door(layout, start_spec, room_centroid)
        if door is not None:
            selectors["door_id"] = door.get("id")
            pose = door.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
            anchors["t"] = {
                "xy": [
                    as_float(pose[0] if len(pose) > 0 else 0.0, 0.0),
                    as_float(pose[1] if len(pose) > 1 else 0.0, 0.0),
                ],
                "label": "t",
            }
            dx = room_centroid[0] - as_float(pose[0] if len(pose) > 0 else 0.0, 0.0)
            dy = room_centroid[1] - as_float(pose[1] if len(pose) > 1 else 0.0, 0.0)
            norm = math.hypot(dx, dy)
            if norm < 1e-9:
                dx, dy, norm = 1.0, 0.0, 1.0
            ux, uy = dx / norm, dy / norm
            in_offset_m = as_float(start_spec.get("in_offset_m"), 0.40)
            start_xy = [as_float(pose[0], 0.0) + ux * in_offset_m, as_float(pose[1], 0.0) + uy * in_offset_m]
    anchors["s0"] = {"xy": [float(start_xy[0]), float(start_xy[1])], "label": "s0"}

    start_cell = _xy_to_cell(float(start_xy[0]), float(start_xy[1]), bounds, resolution)
    snapped_start_cell, moved_start = _snap_cell_to_free(start_cell, free_mask, max_radius_cells)
    snapped_start_xy = list(_cell_center(snapped_start_cell[0], snapped_start_cell[1], bounds, resolution))
    anchors["s"] = {"xy": [float(snapped_start_xy[0]), float(snapped_start_xy[1])], "label": "s"}

    # Goal
    goal_spec = task.get("goal") if isinstance(task.get("goal"), dict) else {}
    goal_mode = str(goal_spec.get("mode") or "")
    goal_xy = [as_float(default_goal_xy[0], 5.0), as_float(default_goal_xy[1], 5.0)]

    if goal_mode == "bedside":
        bed = _resolve_bed(layout, goal_spec, room_centroid)
        if bed is not None:
            selectors["bed_id"] = bed.get("id")
            offset_m = as_float(goal_spec.get("offset_m"), 0.60)
            candidates = _bedside_candidates_xy(bed, offset_m)
            choose = str(goal_spec.get("choose") or "closest_to_room_centroid").strip().lower()

            if choose == "closest_to_room_centroid":
                cx, cy = room_centroid
                goal_xy = list(min(candidates, key=lambda p: (math.hypot(p[0] - cx, p[1] - cy), p[0], p[1])))
            else:
                goal_xy = list(candidates[0])
            anchors["g_bed"] = {"xy": [float(goal_xy[0]), float(goal_xy[1])], "label": "g_bed"}

    goal_cell = _xy_to_cell(float(goal_xy[0]), float(goal_xy[1]), bounds, resolution)
    snapped_goal_cell, moved_goal = _snap_cell_to_free(goal_cell, free_mask, max_radius_cells)
    snapped_goal_xy = list(_cell_center(snapped_goal_cell[0], snapped_goal_cell[1], bounds, resolution))

    return {
        "start": {"mode": start_mode or "manual", "xy": snapped_start_xy, "cell": [snapped_start_cell[0], snapped_start_cell[1]]},
        "goal": {"mode": goal_mode or "manual", "xy": snapped_goal_xy, "cell": [snapped_goal_cell[0], snapped_goal_cell[1]]},
        "selectors": selectors,
        "snap": {"max_radius_cells": max_radius_cells, "moved_start": moved_start, "moved_goal": moved_goal},
        "anchors": anchors,
    }
