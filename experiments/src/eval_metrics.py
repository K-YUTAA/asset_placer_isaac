from __future__ import annotations

import argparse
import heapq
import json
import math
import pathlib
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layout_tools import (
    as_float,
    load_layout_contract,
    point_in_obb,
    point_in_polygon,
    room_bbox,
    wrap_angle,
    write_json,
)

import task_points

Grid = List[List[bool]]
Cell = Tuple[int, int]
WALL_HIT_ID = -1
NONE_HIT_ID = 0


def default_eval_config() -> Dict[str, Any]:
    return {
        "grid_resolution_m": 0.10,
        "robot_radius_m": 0.30,
        "start_xy": [0.8, 0.8],
        "goal_xy": [5.0, 5.0],
        "sample_step_m": 0.50,
        "max_sensor_samples": 10,
        "tau_R": 0.90,
        "tau_clr": 0.20,
        "tau_V": 0.40,
        "tau_Delta": 0.15,
        "lambda_rot": 0.50,
        "entry_observability": {
            "enabled": False,
            "mode": "both",
            "exclude_categories": ["floor", "door", "window"],
            "target_categories": [],
            "height_by_category_m": {},
            "sensor_height_m": 0.60,
            "num_rays": 720,
            "max_range_m": 10.0,
            "tau_p": 0.02,
            "tau_v": 0.30,
        },
    }


def merge_eval_config(base: Dict[str, Any], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if update:
        merged.update(update)
        if isinstance(base.get("entry_observability"), dict):
            eo = dict(base.get("entry_observability") or {})
            if isinstance(update.get("entry_observability"), dict):
                eo.update(update.get("entry_observability") or {})
            merged["entry_observability"] = eo
    return merged


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


def _make_grid(nx: int, ny: int, value: bool = False) -> Grid:
    return [[value for _ in range(nx)] for _ in range(ny)]


def _build_room_and_occupancy(layout: Dict[str, Any], resolution: float) -> Tuple[Grid, Grid, Tuple[float, float, float, float]]:
    room_poly = layout["room"]["boundary_poly_xy"]
    bounds = room_bbox(room_poly)
    nx, ny = _grid_dims(bounds, resolution)

    room_mask = _make_grid(nx, ny, False)
    occ = _make_grid(nx, ny, False)

    for iy in range(ny):
        for ix in range(nx):
            x, y = _cell_center(ix, iy, bounds, resolution)
            inside_room = point_in_polygon(x, y, room_poly)
            room_mask[iy][ix] = inside_room
            if not inside_room:
                continue
            for obj in layout.get("objects", []):
                if point_in_obb(x, y, obj):
                    occ[iy][ix] = True
                    break

    return room_mask, occ, bounds


def _inflate_occupancy(occ: Grid, radius_cells: int) -> Grid:
    ny = len(occ)
    nx = len(occ[0]) if ny > 0 else 0
    inflated = _make_grid(nx, ny, False)

    offsets: List[Tuple[int, int]] = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                offsets.append((dx, dy))

    for iy in range(ny):
        for ix in range(nx):
            if not occ[iy][ix]:
                continue
            for dx, dy in offsets:
                tx, ty = ix + dx, iy + dy
                if 0 <= tx < nx and 0 <= ty < ny:
                    inflated[ty][tx] = True

    return inflated


def _build_free_mask(room_mask: Grid, inflated_occ: Grid) -> Grid:
    ny = len(room_mask)
    nx = len(room_mask[0]) if ny > 0 else 0
    free = _make_grid(nx, ny, False)
    for iy in range(ny):
        for ix in range(nx):
            free[iy][ix] = room_mask[iy][ix] and (not inflated_occ[iy][ix])
    return free


def _neighbors4(ix: int, iy: int, nx: int, ny: int) -> List[Cell]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        tx, ty = ix + dx, iy + dy
        if 0 <= tx < nx and 0 <= ty < ny:
            out.append((tx, ty))
    return out


def _neighbors8_with_cost(ix: int, iy: int, nx: int, ny: int, resolution: float) -> List[Tuple[Cell, float]]:
    out: List[Tuple[Cell, float]] = []
    for dx, dy in (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    ):
        tx, ty = ix + dx, iy + dy
        if 0 <= tx < nx and 0 <= ty < ny:
            step = resolution * (math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0)
            out.append(((tx, ty), step))
    return out


def _bfs_reachable(free_mask: Grid, start: Cell) -> Tuple[set[Cell], Dict[Cell, Cell]]:
    ny = len(free_mask)
    nx = len(free_mask[0]) if ny > 0 else 0
    sx, sy = start
    if sy < 0 or sy >= ny or sx < 0 or sx >= nx or not free_mask[sy][sx]:
        return set(), {}

    q: deque[Cell] = deque([start])
    visited: set[Cell] = {start}
    parent: Dict[Cell, Cell] = {}

    while q:
        cx, cy = q.popleft()
        for nx_cell, ny_cell in _neighbors4(cx, cy, nx, ny):
            if (nx_cell, ny_cell) in visited:
                continue
            if not free_mask[ny_cell][nx_cell]:
                continue
            visited.add((nx_cell, ny_cell))
            parent[(nx_cell, ny_cell)] = (cx, cy)
            q.append((nx_cell, ny_cell))

    return visited, parent


def _astar_path(free_mask: Grid, start: Cell, goal: Cell) -> List[Cell]:
    ny = len(free_mask)
    nx = len(free_mask[0]) if ny > 0 else 0

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < nx and 0 <= sy < ny and free_mask[sy][sx]):
        return []
    if not (0 <= gx < nx and 0 <= gy < ny and free_mask[gy][gx]):
        return []

    def heuristic(a: Cell, b: Cell) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap: List[Tuple[float, Cell]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for nx_cell, ny_cell in _neighbors4(cx, cy, nx, ny):
            if not free_mask[ny_cell][nx_cell]:
                continue
            neighbor = (nx_cell, ny_cell)
            tentative = g_score[current] + 1.0
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return []


def _distance_transform(occ: Grid, resolution: float) -> List[List[float]]:
    ny = len(occ)
    nx = len(occ[0]) if ny > 0 else 0
    dist = [[float("inf") for _ in range(nx)] for _ in range(ny)]

    heap: List[Tuple[float, Cell]] = []
    for iy in range(ny):
        for ix in range(nx):
            if occ[iy][ix]:
                dist[iy][ix] = 0.0
                heapq.heappush(heap, (0.0, (ix, iy)))

    while heap:
        cur_dist, (cx, cy) = heapq.heappop(heap)
        if cur_dist > dist[cy][cx]:
            continue
        for (nx_cell, ny_cell), step in _neighbors8_with_cost(cx, cy, nx, ny, resolution):
            nxt = cur_dist + step
            if nxt < dist[ny_cell][nx_cell]:
                dist[ny_cell][nx_cell] = nxt
                heapq.heappush(heap, (nxt, (nx_cell, ny_cell)))

    return dist


def _bresenham_line(a: Cell, b: Cell) -> List[Cell]:
    x0, y0 = a
    x1, y1 = b
    points: List[Cell] = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return points


def _line_of_sight_clear(a: Cell, b: Cell, occ: Grid, room_mask: Grid) -> bool:
    ny = len(occ)
    nx = len(occ[0]) if ny > 0 else 0

    for x, y in _bresenham_line(a, b):
        if x < 0 or y < 0 or x >= nx or y >= ny:
            return False
        if not room_mask[y][x]:
            return False
        if occ[y][x]:
            return False
    return True


def _sample_path_cells(path: List[Cell], sample_step_m: float, resolution: float, max_samples: int) -> List[Cell]:
    if not path:
        return []

    step_cells = max(1, int(round(sample_step_m / resolution)))
    sampled = [path[0]]
    idx = step_cells
    while idx < len(path) and len(sampled) < max_samples:
        sampled.append(path[idx])
        idx += step_cells

    if sampled[-1] != path[-1] and len(sampled) < max_samples:
        sampled.append(path[-1])

    return sampled


def _to_lower_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    out: set[str] = set()
    for v in values:
        text = str(v or "").strip().lower()
        if text:
            out.add(text)
    return out


def _resolve_object_height_m(obj: Dict[str, Any], height_by_category_m: Dict[str, float]) -> float:
    size = obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
    raw_h = as_float(size[2] if len(size) > 2 else 0.0, 0.0)
    if raw_h > 1e-9:
        return raw_h

    category = str(obj.get("category") or "").strip().lower()
    fallback_h = as_float(height_by_category_m.get(category), 0.0)
    if fallback_h > 1e-9:
        return fallback_h
    return 1.0


def _collect_entry_target_objects(
    layout: Dict[str, Any],
    entry_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    exclude_categories = _to_lower_set(entry_cfg.get("exclude_categories"))
    target_categories = _to_lower_set(entry_cfg.get("target_categories"))
    has_target_filter = len(target_categories) > 0

    raw_height_map = entry_cfg.get("height_by_category_m")
    raw_height_map = raw_height_map if isinstance(raw_height_map, dict) else {}
    height_by_category_m = {str(k).strip().lower(): as_float(v, 0.0) for k, v in raw_height_map.items()}

    targets: List[Dict[str, Any]] = []
    for idx, obj in enumerate(layout.get("objects", [])):
        if not isinstance(obj, dict):
            continue

        category = str(obj.get("category") or "").strip().lower()
        if category in exclude_categories:
            continue
        if has_target_filter and category not in target_categories:
            continue

        obj_id = str(obj.get("id") or f"obj_{idx:02d}")
        size = obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
        length = max(1e-6, as_float(size[0] if len(size) > 0 else 1.0, 1.0))
        width = max(1e-6, as_float(size[1] if len(size) > 1 else 1.0, 1.0))
        height = _resolve_object_height_m(obj, height_by_category_m)
        side_area = 2.0 * (length + width) * height
        weight = max(1e-6, length * width)

        targets.append(
            {
                "obj": obj,
                "id": obj_id,
                "category": category or "unknown",
                "height_m": height,
                "length_m": length,
                "width_m": width,
                "side_area_m2": max(1e-6, side_area),
                "weight": weight,
            }
        )

    targets.sort(key=lambda x: x["id"])
    return targets


def _compute_visible_free_from_start(
    start_cell: Cell,
    free_mask_raw: Grid,
    occ: Grid,
    room_mask: Grid,
) -> Tuple[Grid, int, int]:
    ny = len(free_mask_raw)
    nx = len(free_mask_raw[0]) if ny > 0 else 0
    visible_free = _make_grid(nx, ny, False)

    free_total = 0
    visible_total = 0
    for iy in range(ny):
        for ix in range(nx):
            if not free_mask_raw[iy][ix]:
                continue
            free_total += 1
            if _line_of_sight_clear(start_cell, (ix, iy), occ, room_mask):
                visible_free[iy][ix] = True
                visible_total += 1

    return visible_free, free_total, visible_total


def _build_occ_sense_and_obj_id_grid(
    target_objects: List[Dict[str, Any]],
    bounds: Tuple[float, float, float, float],
    resolution: float,
    room_mask: Grid,
    sensor_height_m: float,
) -> Tuple[Grid, List[List[int]], Dict[int, Dict[str, Any]]]:
    ny = len(room_mask)
    nx = len(room_mask[0]) if ny > 0 else 0
    occ_sense = _make_grid(nx, ny, False)
    obj_id_grid: List[List[int]] = [[0 for _ in range(nx)] for _ in range(ny)]

    sensed_objects = [o for o in target_objects if as_float(o.get("height_m"), 0.0) + 1e-9 >= sensor_height_m]

    id_to_meta: Dict[int, Dict[str, Any]] = {}
    for grid_id, target in enumerate(sensed_objects, start=1):
        id_to_meta[grid_id] = {
            "id": target["id"],
            "category": target["category"],
            "height_m": as_float(target.get("height_m"), 1.0),
            "weight": as_float(target.get("weight"), 1.0),
        }

    for iy in range(ny):
        for ix in range(nx):
            if not room_mask[iy][ix]:
                continue
            x, y = _cell_center(ix, iy, bounds, resolution)
            for grid_id, target in enumerate(sensed_objects, start=1):
                if point_in_obb(x, y, target["obj"]):
                    occ_sense[iy][ix] = True
                    obj_id_grid[iy][ix] = grid_id
                    break

    return occ_sense, obj_id_grid, id_to_meta


def _raycast_first_hit(
    sensor_cell: Cell,
    end_cell: Cell,
    occ_sense: Grid,
    obj_id_grid: List[List[int]],
    room_mask: Grid,
) -> int:
    ny = len(room_mask)
    nx = len(room_mask[0]) if ny > 0 else 0
    for idx, (x, y) in enumerate(_bresenham_line(sensor_cell, end_cell)):
        if idx == 0:
            continue
        if x < 0 or y < 0 or x >= nx or y >= ny:
            return WALL_HIT_ID
        if not room_mask[y][x]:
            return WALL_HIT_ID
        if occ_sense[y][x]:
            obj_id = obj_id_grid[y][x]
            return obj_id if obj_id > 0 else WALL_HIT_ID
    return NONE_HIT_ID


def _compute_entry_observability_first_hit(
    start_cell: Cell,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    occ_sense: Grid,
    obj_id_grid: List[List[int]],
    room_mask: Grid,
    id_to_meta: Dict[int, Dict[str, Any]],
    entry_cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    num_rays = max(1, int(entry_cfg.get("num_rays", 720)))
    tau_p = as_float(entry_cfg.get("tau_p"), 0.02)
    max_range_default = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
    max_range_m = as_float(entry_cfg.get("max_range_m"), max_range_default)
    max_range_cells = max(1, int(math.ceil(max_range_m / max(1e-9, resolution))))

    hit_counts: Dict[int, int] = {grid_id: 0 for grid_id in id_to_meta.keys()}
    wall_hits = 0
    none_hits = 0
    sampled_rays: List[Dict[str, Any]] = []
    sample_stride = max(1, num_rays // 36)

    sx, sy = start_cell
    for k in range(num_rays):
        alpha = (2.0 * math.pi * float(k)) / float(num_rays)
        ex = sx + int(round(math.cos(alpha) * max_range_cells))
        ey = sy + int(round(math.sin(alpha) * max_range_cells))
        hit_id = _raycast_first_hit(start_cell, (ex, ey), occ_sense, obj_id_grid, room_mask)
        if hit_id == WALL_HIT_ID:
            wall_hits += 1
        elif hit_id == NONE_HIT_ID:
            none_hits += 1
        else:
            hit_counts[hit_id] = hit_counts.get(hit_id, 0) + 1

        if (k % sample_stride) == 0:
            sampled_rays.append({"index": k, "end_cell": [ex, ey], "hit_id": int(hit_id)})

    weighted_sum = 0.0
    weight_total = 0.0
    recognized = 0
    per_object: Dict[str, Dict[str, Any]] = {}
    p_hit_by_object: Dict[str, float] = {}
    hit_counts_by_object: Dict[str, int] = {}

    for grid_id, meta in id_to_meta.items():
        obj_id = str(meta.get("id") or f"obj_{grid_id:02d}")
        p_hit = hit_counts.get(grid_id, 0) / float(num_rays)
        weight = max(1e-6, as_float(meta.get("weight"), 1.0))
        weighted_sum += weight * p_hit
        weight_total += weight
        if p_hit >= tau_p:
            recognized += 1

        per_object[obj_id] = {"p_hit": p_hit}
        p_hit_by_object[obj_id] = p_hit
        hit_counts_by_object[obj_id] = int(hit_counts.get(grid_id, 0))

    obj_count = len(id_to_meta)
    metrics = {
        "OOE_C_obj_entry_hit": (weighted_sum / weight_total) if weight_total > 0.0 else 0.0,
        "OOE_R_rec_entry_hit": (recognized / float(obj_count)) if obj_count > 0 else 0.0,
    }
    debug = {
        "num_rays": num_rays,
        "tau_p": tau_p,
        "wall_hits": wall_hits,
        "none_hits": none_hits,
        "hit_counts": hit_counts_by_object,
        "p_hit": p_hit_by_object,
        "sampled_rays": sampled_rays,
    }
    return metrics, per_object, debug


def _compute_entry_observability_surface(
    target_objects: List[Dict[str, Any]],
    room_mask: Grid,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    visible_free: Grid,
    entry_cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    ny = len(room_mask)
    nx = len(room_mask[0]) if ny > 0 else 0
    tau_v = as_float(entry_cfg.get("tau_v"), 0.30)

    weighted_sum = 0.0
    weight_total = 0.0
    recognized = 0
    per_object: Dict[str, Dict[str, Any]] = {}
    surface_debug_objects: List[Dict[str, Any]] = []

    for target in target_objects:
        obj = target["obj"]
        occ_cells: set[Cell] = set()
        for iy in range(ny):
            for ix in range(nx):
                if not room_mask[iy][ix]:
                    continue
                x, y = _cell_center(ix, iy, bounds, resolution)
                if point_in_obb(x, y, obj):
                    occ_cells.add((ix, iy))

        boundary_cells: List[Cell] = []
        for ix, iy in occ_cells:
            for tx, ty in _neighbors4(ix, iy, nx, ny):
                if (tx, ty) not in occ_cells:
                    boundary_cells.append((ix, iy))
                    break

        visible_boundary = 0
        for ix, iy in boundary_cells:
            visible = False
            for tx, ty in _neighbors4(ix, iy, nx, ny):
                if visible_free[ty][tx]:
                    visible = True
                    break
            if visible:
                visible_boundary += 1

        boundary_count = len(boundary_cells)
        v_surf = (visible_boundary / float(boundary_count)) if boundary_count > 0 else 0.0

        side_area = max(1e-6, as_float(target.get("side_area_m2"), 1.0))
        visible_side_area = side_area * v_surf
        weight = max(1e-6, as_float(target.get("weight"), 1.0))

        weighted_sum += weight * v_surf
        weight_total += weight
        if v_surf >= tau_v:
            recognized += 1

        obj_id = str(target.get("id") or "")
        per_object[obj_id] = {"v_surf": v_surf, "visible_side_area_m2": visible_side_area}
        surface_debug_objects.append(
            {
                "id": obj_id,
                "category": target.get("category"),
                "boundary_cells": boundary_count,
                "visible_boundary_cells": visible_boundary,
                "v_surf": v_surf,
                "visible_side_area_m2": visible_side_area,
            }
        )

    obj_count = len(target_objects)
    metrics = {
        "OOE_C_obj_entry_surf": (weighted_sum / weight_total) if weight_total > 0.0 else 0.0,
        "OOE_R_rec_entry_surf": (recognized / float(obj_count)) if obj_count > 0 else 0.0,
    }
    debug = {"tau_v": tau_v, "objects": surface_debug_objects}
    return metrics, per_object, debug


def _compute_delta_layout(
    layout: Dict[str, Any],
    baseline_layout: Optional[Dict[str, Any]],
    lambda_rot: float,
) -> float:
    if not baseline_layout:
        return 0.0

    objects = {obj["id"]: obj for obj in layout.get("objects", [])}
    baseline = {obj["id"]: obj for obj in baseline_layout.get("objects", [])}

    common_ids = [obj_id for obj_id in objects.keys() if obj_id in baseline]
    movable_ids = [obj_id for obj_id in common_ids if bool(objects[obj_id].get("movable", True))]
    if not movable_ids:
        return 0.0

    boundary = layout["room"]["boundary_poly_xy"]
    min_x, min_y, max_x, max_y = room_bbox(boundary)
    diag = math.hypot(max_x - min_x, max_y - min_y)
    if diag < 1e-9:
        diag = 1.0

    areas = {}
    for obj_id in movable_ids:
        size = objects[obj_id].get("size_lwh_m", [1.0, 1.0, 1.0])
        areas[obj_id] = max(1e-6, as_float(size[0], 1.0) * as_float(size[1], 1.0))

    area_sum = sum(areas.values())
    delta = 0.0
    for obj_id in movable_ids:
        cur = objects[obj_id]["pose_xyz_yaw"]
        ref = baseline[obj_id]["pose_xyz_yaw"]

        dp = math.hypot(as_float(cur[0], 0.0) - as_float(ref[0], 0.0), as_float(cur[1], 0.0) - as_float(ref[1], 0.0))
        dtheta = abs(wrap_angle(as_float(cur[3], 0.0) - as_float(ref[3], 0.0)))

        w = areas[obj_id] / area_sum
        delta += w * ((dp / diag) + lambda_rot * (dtheta / math.pi))

    return delta


def _grid_to_pgm(path: pathlib.Path, values: List[List[int]]) -> None:
    ny = len(values)
    nx = len(values[0]) if ny > 0 else 0
    lines = ["P2", f"{nx} {ny}", "255"]
    for row in values:
        lines.append(" ".join(str(max(0, min(255, int(v)))) for v in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def evaluate_layout(
    layout: Dict[str, Any],
    baseline_layout: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()

    resolution = as_float(config.get("grid_resolution_m"), 0.10)
    robot_radius = as_float(config.get("robot_radius_m"), 0.30)
    start_xy = config.get("start_xy") or [0.8, 0.8]
    goal_xy = config.get("goal_xy") or [5.0, 5.0]

    room_mask, occ, bounds = _build_room_and_occupancy(layout, resolution)
    nx, ny = _grid_dims(bounds, resolution)

    radius_cells = max(0, int(math.ceil(robot_radius / resolution)))
    inflated = _inflate_occupancy(occ, radius_cells)
    free_mask = _build_free_mask(room_mask, inflated)

    task_points_debug = None
    task_spec = config.get("task")
    if isinstance(task_spec, dict) and task_spec:
        task_points_debug = task_points.resolve_task_points(layout, config, bounds, resolution, free_mask)
        start_xy = task_points_debug["start"]["xy"]
        goal_xy = task_points_debug["goal"]["xy"]
        start_cell = task_points_debug["start"]["cell"]
        goal_cell = task_points_debug["goal"]["cell"]
        start = (int(start_cell[0]), int(start_cell[1]))
        goal = (int(goal_cell[0]), int(goal_cell[1]))
    else:
        start = _xy_to_cell(as_float(start_xy[0], 0.8), as_float(start_xy[1], 0.8), bounds, resolution)
        goal = _xy_to_cell(as_float(goal_xy[0], 5.0), as_float(goal_xy[1], 5.0), bounds, resolution)

    reachable, _ = _bfs_reachable(free_mask, start)
    free_total = sum(1 for iy in range(ny) for ix in range(nx) if free_mask[iy][ix])
    reach_rate = (len(reachable) / free_total) if free_total > 0 else 0.0

    path_cells = _astar_path(free_mask, start, goal)

    dist_to_occ = _distance_transform(occ, resolution)
    clr_min = 0.0
    bottleneck_cell: Optional[Cell] = None
    if path_cells:
        clearance_values = []
        for ix, iy in path_cells:
            clearance_values.append((dist_to_occ[iy][ix] - robot_radius, (ix, iy)))
        clr_min, bottleneck_cell = min(clearance_values, key=lambda x: x[0])

    sensor_cells = []
    if start in reachable:
        sensor_cells.append(start)
    sensor_cells.extend(
        _sample_path_cells(
            path_cells,
            as_float(config.get("sample_step_m"), 0.50),
            resolution,
            int(config.get("max_sensor_samples", 10)),
        )
    )

    visible_count = 0
    for iy in range(ny):
        for ix in range(nx):
            if not free_mask[iy][ix]:
                continue
            visible = False
            for sensor_cell in sensor_cells:
                if _line_of_sight_clear(sensor_cell, (ix, iy), occ, room_mask):
                    visible = True
                    break
            if visible:
                visible_count += 1

    c_vis = (visible_count / free_total) if free_total > 0 else 0.0
    delta_layout = _compute_delta_layout(layout, baseline_layout, as_float(config.get("lambda_rot"), 0.50))

    entry_cfg = config.get("entry_observability")
    entry_cfg = entry_cfg if isinstance(entry_cfg, dict) else {}
    entry_enabled = bool(entry_cfg.get("enabled", False))
    entry_mode = str(entry_cfg.get("mode") or "both").strip().lower()
    if entry_mode not in {"first_hit", "surface", "both"}:
        entry_mode = "both"

    c_vis_start = 0.0
    ooe_c_hit = 0.0
    ooe_r_hit = 0.0
    ooe_c_surf = 0.0
    ooe_r_surf = 0.0
    ooe_per_object: List[Dict[str, Any]] = []
    entry_debug: Dict[str, Any] = {}
    if entry_enabled and 0 <= start[0] < nx and 0 <= start[1] < ny:
        free_mask_raw = _build_free_mask(room_mask, occ)
        visible_free, free_raw_total, visible_raw_total = _compute_visible_free_from_start(start, free_mask_raw, occ, room_mask)
        c_vis_start = (visible_raw_total / free_raw_total) if free_raw_total > 0 else 0.0

        target_objects = _collect_entry_target_objects(layout, entry_cfg)
        per_object_map: Dict[str, Dict[str, Any]] = {
            str(target["id"]): {
                "id": str(target["id"]),
                "category": target["category"],
                "p_hit": 0.0,
                "v_surf": 0.0,
                "visible_side_area_m2": 0.0,
                "height_m": as_float(target.get("height_m"), 1.0),
            }
            for target in target_objects
        }

        if entry_mode in {"first_hit", "both"}:
            sensor_height_m = as_float(entry_cfg.get("sensor_height_m"), 0.60)
            occ_sense, obj_id_grid, id_to_meta = _build_occ_sense_and_obj_id_grid(
                target_objects=target_objects,
                bounds=bounds,
                resolution=resolution,
                room_mask=room_mask,
                sensor_height_m=sensor_height_m,
            )
            first_hit_metrics, first_hit_per_obj, first_hit_debug = _compute_entry_observability_first_hit(
                start_cell=start,
                bounds=bounds,
                resolution=resolution,
                occ_sense=occ_sense,
                obj_id_grid=obj_id_grid,
                room_mask=room_mask,
                id_to_meta=id_to_meta,
                entry_cfg=entry_cfg,
            )
            ooe_c_hit = first_hit_metrics["OOE_C_obj_entry_hit"]
            ooe_r_hit = first_hit_metrics["OOE_R_rec_entry_hit"]
            for obj_id, item in first_hit_per_obj.items():
                if obj_id in per_object_map:
                    per_object_map[obj_id].update(item)
            entry_debug["occ_sense"] = occ_sense
            entry_debug["obj_id_meta"] = {str(k): v for k, v in id_to_meta.items()}
            entry_debug["first_hit"] = first_hit_debug

        if entry_mode in {"surface", "both"}:
            surface_metrics, surface_per_obj, surface_debug = _compute_entry_observability_surface(
                target_objects=target_objects,
                room_mask=room_mask,
                bounds=bounds,
                resolution=resolution,
                visible_free=visible_free,
                entry_cfg=entry_cfg,
            )
            ooe_c_surf = surface_metrics["OOE_C_obj_entry_surf"]
            ooe_r_surf = surface_metrics["OOE_R_rec_entry_surf"]
            for obj_id, item in surface_per_obj.items():
                if obj_id in per_object_map:
                    per_object_map[obj_id].update(item)
            entry_debug["surface"] = surface_debug

        ooe_per_object = [per_object_map[obj_id] for obj_id in sorted(per_object_map.keys())]
        entry_debug["enabled"] = True
        entry_debug["mode"] = entry_mode
        entry_debug["C_vis_start"] = c_vis_start
        entry_debug["free_total_raw"] = free_raw_total
        entry_debug["visible_total_raw"] = visible_raw_total

    tau_r = as_float(config.get("tau_R"), 0.90)
    tau_clr = as_float(config.get("tau_clr"), 0.20)
    tau_v = as_float(config.get("tau_V"), 0.40)
    tau_delta = as_float(config.get("tau_Delta"), 0.15)

    adopt = int(reach_rate >= tau_r and clr_min >= tau_clr and c_vis >= tau_v and delta_layout <= tau_delta)

    valid_start = free_mask[start[1]][start[0]] if 0 <= start[1] < ny and 0 <= start[0] < nx else False
    valid_goal = free_mask[goal[1]][goal[0]] if 0 <= goal[1] < ny and 0 <= goal[0] < nx else False
    validity = int(valid_start and valid_goal)

    runtime_sec = time.perf_counter() - t0

    metrics = {
        "C_vis": c_vis,
        "R_reach": reach_rate,
        "clr_min": max(0.0, clr_min),
        "Delta_layout": delta_layout,
        "Adopt": adopt,
        "validity": validity,
        "runtime_sec": runtime_sec,
        "C_vis_start": c_vis_start,
        "OOE_C_obj_entry_hit": ooe_c_hit,
        "OOE_R_rec_entry_hit": ooe_r_hit,
        "OOE_C_obj_entry_surf": ooe_c_surf,
        "OOE_R_rec_entry_surf": ooe_r_surf,
        "OOE_per_object": ooe_per_object,
    }

    debug = {
        "bounds": bounds,
        "resolution": resolution,
        "start_cell": start,
        "goal_cell": goal,
        "path_cells": path_cells,
        "bottleneck_cell": bottleneck_cell,
        "room_mask": room_mask,
        "occupancy": occ,
        "inflated": inflated,
        "free_mask": free_mask,
        "reachable": reachable,
        "distance_to_occ": dist_to_occ,
    }
    if task_points_debug is not None:
        debug["task_points"] = task_points_debug
    if entry_debug:
        debug["entry_observability"] = entry_debug

    return metrics, debug


def _save_debug_maps(debug: Dict[str, Any], out_dir: pathlib.Path) -> None:
    room_mask: Grid = debug["room_mask"]
    occ: Grid = debug["occupancy"]
    free_mask: Grid = debug["free_mask"]
    reachable: set[Cell] = debug["reachable"]
    path_cells: List[Cell] = debug["path_cells"]

    ny = len(room_mask)
    nx = len(room_mask[0]) if ny > 0 else 0

    occ_img = [[128 for _ in range(nx)] for _ in range(ny)]
    reach_img = [[128 for _ in range(nx)] for _ in range(ny)]

    for iy in range(ny):
        for ix in range(nx):
            if not room_mask[iy][ix]:
                continue
            occ_img[iy][ix] = 0 if occ[iy][ix] else 255

            if free_mask[iy][ix]:
                reach_img[iy][ix] = 80
            else:
                reach_img[iy][ix] = 0

            if (ix, iy) in reachable:
                reach_img[iy][ix] = 255

    for ix, iy in path_cells:
        if 0 <= ix < nx and 0 <= iy < ny:
            reach_img[iy][ix] = 200

    _grid_to_pgm(out_dir / "occupancy.pgm", occ_img)
    _grid_to_pgm(out_dir / "reachability.pgm", reach_img)
    if debug.get("task_points") is not None:
        write_json(out_dir / "task_points.json", debug["task_points"])
    entry_debug = debug.get("entry_observability")
    if isinstance(entry_debug, dict):
        occ_sense = entry_debug.get("occ_sense")
        if isinstance(occ_sense, list) and occ_sense:
            ooe_occ_img = [[128 for _ in range(nx)] for _ in range(ny)]
            for iy in range(ny):
                for ix in range(nx):
                    if not room_mask[iy][ix]:
                        continue
                    ooe_occ_img[iy][ix] = 0 if bool(occ_sense[iy][ix]) else 255
            _grid_to_pgm(out_dir / "ooe_occ_sense.pgm", ooe_occ_img)
        if entry_debug.get("obj_id_meta") is not None:
            write_json(out_dir / "ooe_obj_id.json", {"objects": entry_debug.get("obj_id_meta")})
        if entry_debug.get("first_hit") is not None:
            write_json(out_dir / "ooe_hits.json", entry_debug.get("first_hit"))
            first_hit = entry_debug.get("first_hit")
            if isinstance(first_hit, dict) and first_hit.get("sampled_rays") is not None:
                write_json(out_dir / "ooe_rays.json", {"sampled_rays": first_hit.get("sampled_rays")})
        if entry_debug.get("surface") is not None:
            write_json(out_dir / "ooe_surface.json", entry_debug.get("surface"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate layout metrics")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--baseline_layout", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug_dir", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = default_eval_config()
    if args.config:
        cfg = merge_eval_config(cfg, json.loads(pathlib.Path(args.config).read_text(encoding="utf-8-sig")))

    layout = load_layout_contract(pathlib.Path(args.layout))
    baseline_layout = load_layout_contract(pathlib.Path(args.baseline_layout)) if args.baseline_layout else None

    metrics, debug = evaluate_layout(layout, baseline_layout, cfg)

    if args.debug_dir:
        _save_debug_maps(debug, pathlib.Path(args.debug_dir))

    write_json(pathlib.Path(args.out), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

