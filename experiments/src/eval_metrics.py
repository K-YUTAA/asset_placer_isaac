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
        "occupancy_exclude_categories": ["floor"],
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


def _build_room_and_occupancy(
    layout: Dict[str, Any],
    resolution: float,
    occupancy_exclude_categories: Optional[set[str]] = None,
) -> Tuple[Grid, Grid, Tuple[float, float, float, float]]:
    room_poly = layout["room"]["boundary_poly_xy"]
    bounds = room_bbox(room_poly)
    nx, ny = _grid_dims(bounds, resolution)

    room_mask = _make_grid(nx, ny, False)
    occ = _make_grid(nx, ny, False)
    exclude_set = occupancy_exclude_categories or set()
    occupancy_objects: List[Dict[str, Any]] = []
    for obj in layout.get("objects", []):
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "").strip().lower()
        obj_id = str(obj.get("id") or "").strip().lower()
        if category in {"window", "opening"}:
            # Window/opening objects are handled by room-boundary logic.
            continue
        if category in exclude_set:
            continue
        # Backward compatibility for contracts with missing category.
        if "floor" in exclude_set and (obj_id.startswith("floor_") or obj_id == "floor"):
            continue
        occupancy_objects.append(obj)

    for iy in range(ny):
        for ix in range(nx):
            x, y = _cell_center(ix, iy, bounds, resolution)
            inside_room = point_in_polygon(x, y, room_poly)
            room_mask[iy][ix] = inside_room
            if not inside_room:
                continue
            for obj in occupancy_objects:
                if point_in_obb(x, y, obj):
                    occ[iy][ix] = True
                    break

    _apply_internal_room_walls(layout, room_mask, occ, bounds, resolution)

    return room_mask, occ, bounds


def _normalize_room_poly(room: Dict[str, Any]) -> List[Tuple[float, float]]:
    raw_poly = room.get("room_polygon")
    if not isinstance(raw_poly, list):
        raw_poly = room.get("boundary_poly_xy")
    points: List[Tuple[float, float]] = []
    if not isinstance(raw_poly, list):
        return points
    for p in raw_poly:
        if isinstance(p, dict):
            points.append((as_float(p.get("X", p.get("x", 0.0)), 0.0), as_float(p.get("Y", p.get("y", 0.0)), 0.0)))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            points.append((as_float(p[0], 0.0), as_float(p[1], 0.0)))
    return points


def _segment_overlap_collinear(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
    tol: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    ax = a1[0] - a0[0]
    ay = a1[1] - a0[1]
    alen = math.hypot(ax, ay)
    if alen <= tol:
        return None

    bx = b1[0] - b0[0]
    by = b1[1] - b0[1]
    cross_ab = ax * by - ay * bx
    if abs(cross_ab) > tol * max(1.0, alen):
        return None

    vx = b0[0] - a0[0]
    vy = b0[1] - a0[1]
    cross_av = ax * vy - ay * vx
    if abs(cross_av) > tol * max(1.0, alen):
        return None

    ux = ax / alen
    uy = ay / alen
    t_b0 = (b0[0] - a0[0]) * ux + (b0[1] - a0[1]) * uy
    t_b1 = (b1[0] - a0[0]) * ux + (b1[1] - a0[1]) * uy

    t0 = max(0.0, min(t_b0, t_b1))
    t1 = min(alen, max(t_b0, t_b1))
    if t1 - t0 <= tol:
        return None

    p0 = (a0[0] + ux * t0, a0[1] + uy * t0)
    p1 = (a0[0] + ux * t1, a0[1] + uy * t1)
    return p0, p1, (t1 - t0)


def _merge_intervals(intervals: List[Tuple[float, float]], lo: float, hi: float, tol: float) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for a, b in intervals:
        x0 = max(lo, min(a, b))
        x1 = min(hi, max(a, b))
        if x1 - x0 > tol:
            clipped.append((x0, x1))
    if not clipped:
        return []
    clipped.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[float, float]] = [clipped[0]]
    for a, b in clipped[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b + tol:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def _subtract_intervals(base_lo: float, base_hi: float, cuts: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    merged = _merge_intervals(cuts, base_lo, base_hi, tol)
    if not merged:
        return [(base_lo, base_hi)] if base_hi - base_lo > tol else []
    out: List[Tuple[float, float]] = []
    cur = base_lo
    for a, b in merged:
        if a - cur > tol:
            out.append((cur, a))
        cur = max(cur, b)
    if base_hi - cur > tol:
        out.append((cur, base_hi))
    return out


def _mark_wall_segment_cells(
    occ: Grid,
    room_mask: Grid,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
) -> None:
    ny = len(occ)
    nx = len(occ[0]) if ny > 0 else 0
    if nx <= 0 or ny <= 0:
        return

    c0 = _xy_to_cell(p0[0], p0[1], bounds, resolution)
    c1 = _xy_to_cell(p1[0], p1[1], bounds, resolution)
    for ix, iy in _bresenham_line(c0, c1):
        if 0 <= ix < nx and 0 <= iy < ny and room_mask[iy][ix]:
            occ[iy][ix] = True


def _collect_shared_room_segments(rooms: List[Dict[str, Any]], tol: float) -> List[Tuple[int, int, Tuple[float, float], Tuple[float, float], float]]:
    segments: List[Tuple[int, Tuple[float, float], Tuple[float, float]]] = []
    for room_idx, room in enumerate(rooms):
        poly = _normalize_room_poly(room)
        n = len(poly)
        if n < 3:
            continue
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i + 1) % n]
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= tol:
                continue
            segments.append((room_idx, p0, p1))

    shared: Dict[Tuple[int, int, int, int], Tuple[int, int, Tuple[float, float], Tuple[float, float], float]] = {}
    for i in range(len(segments)):
        ri, a0, a1 = segments[i]
        for j in range(i + 1, len(segments)):
            rj, b0, b1 = segments[j]
            if ri == rj:
                continue
            overlap = _segment_overlap_collinear(a0, a1, b0, b1, tol)
            if overlap is None:
                continue
            p0, p1, seg_len = overlap
            key = (
                int(round(min(p0[0], p1[0]) * 1000.0)),
                int(round(min(p0[1], p1[1]) * 1000.0)),
                int(round(max(p0[0], p1[0]) * 1000.0)),
                int(round(max(p0[1], p1[1]) * 1000.0)),
            )
            shared[key] = (ri, rj, p0, p1, seg_len)

    return list(shared.values())


def _opening_intervals_on_segment(
    seg_p0: Tuple[float, float],
    seg_p1: Tuple[float, float],
    seg_len: float,
    room_entries: List[Tuple[int, Dict[str, Any]]],
    resolution: float,
) -> List[Tuple[float, float]]:
    if seg_len <= 1e-9:
        return []
    ux = (seg_p1[0] - seg_p0[0]) / seg_len
    uy = (seg_p1[1] - seg_p0[1]) / seg_len
    line_tol = max(0.08, resolution * 0.75)
    intervals: List[Tuple[float, float]] = []
    # Treat only explicit generic openings as passable gaps.
    # Door/window are blocked unless modeled as true opening objects.
    opening_types = {"opening"}

    for _, room in room_entries:
        raw_openings = room.get("openings")
        if not isinstance(raw_openings, list):
            continue
        for opening in raw_openings:
            if not isinstance(opening, dict):
                continue
            typ = str(opening.get("type") or "").strip().lower()
            if typ and typ not in opening_types:
                continue
            width = as_float(opening.get("Width", opening.get("width", 0.0)), 0.0)
            if width <= 1e-6:
                continue
            ox = as_float(opening.get("X", opening.get("x", 0.0)), 0.0)
            oy = as_float(opening.get("Y", opening.get("y", 0.0)), 0.0)
            vx = ox - seg_p0[0]
            vy = oy - seg_p0[1]
            t = vx * ux + vy * uy
            dist_line = abs(vx * uy - vy * ux)
            if dist_line > line_tol:
                continue
            if t < -width or t > seg_len + width:
                continue
            intervals.append((t - 0.5 * width, t + 0.5 * width))
    return intervals


def _apply_internal_room_walls(
    layout: Dict[str, Any],
    room_mask: Grid,
    occ: Grid,
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> None:
    raw_rooms = layout.get("rooms")
    if not isinstance(raw_rooms, list) or len(raw_rooms) < 2:
        return

    rooms = [room for room in raw_rooms if isinstance(room, dict) and len(_normalize_room_poly(room)) >= 3]
    if len(rooms) < 2:
        return

    tol = max(1e-4, resolution * 0.25)
    shared_segments = _collect_shared_room_segments(rooms, tol)
    for room_i, room_j, seg_p0, seg_p1, seg_len in shared_segments:
        opening_intervals = _opening_intervals_on_segment(
            seg_p0=seg_p0,
            seg_p1=seg_p1,
            seg_len=seg_len,
            room_entries=[(room_i, rooms[room_i]), (room_j, rooms[room_j])],
            resolution=resolution,
        )
        wall_ranges = _subtract_intervals(0.0, seg_len, opening_intervals, tol)
        if not wall_ranges:
            continue

        ux = (seg_p1[0] - seg_p0[0]) / max(seg_len, 1e-9)
        uy = (seg_p1[1] - seg_p0[1]) / max(seg_len, 1e-9)
        for t0, t1 in wall_ranges:
            p0 = (seg_p0[0] + ux * t0, seg_p0[1] + uy * t0)
            p1 = (seg_p0[0] + ux * t1, seg_p0[1] + uy * t1)
            _mark_wall_segment_cells(occ, room_mask, bounds, resolution, p0, p1)


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


def _astar_path(free_mask: Grid, start: Cell, goal: Cell, resolution: float = 1.0) -> List[Cell]:
    ny = len(free_mask)
    nx = len(free_mask[0]) if ny > 0 else 0

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < nx and 0 <= sy < ny and free_mask[sy][sx]):
        return []
    if not (0 <= gx < nx and 0 <= gy < ny and free_mask[gy][gx]):
        return []

    d = max(1e-9, resolution)
    d_diag = math.sqrt(2.0) * d

    def heuristic(a: Cell, b: Cell) -> float:
        # Octile distance for 8-neighborhood.
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return d * (dx + dy) + (d_diag - 2.0 * d) * min(dx, dy)

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
        for (nx_cell, ny_cell), step_cost in _neighbors8_with_cost(cx, cy, nx, ny, d):
            if not free_mask[ny_cell][nx_cell]:
                continue
            dx = nx_cell - cx
            dy = ny_cell - cy
            # Prevent corner cutting on diagonal moves.
            if dx != 0 and dy != 0:
                if not (free_mask[cy][cx + dx] and free_mask[cy + dy][cx]):
                    continue
            neighbor = (nx_cell, ny_cell)
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return []


def _line_of_sight_free(a: Cell, b: Cell, free_mask: Grid) -> bool:
    ny = len(free_mask)
    nx = len(free_mask[0]) if ny > 0 else 0
    for x, y in _bresenham_line(a, b):
        if x < 0 or y < 0 or x >= nx or y >= ny:
            return False
        if not free_mask[y][x]:
            return False
    return True


def _smooth_path_cells(path: List[Cell], free_mask: Grid, max_skip_cells: int = 8) -> List[Cell]:
    if len(path) <= 2:
        return list(path)

    skip_cap = max(2, int(max_skip_cells))
    smoothed: List[Cell] = [path[0]]
    anchor_idx = 0
    n = len(path)
    while anchor_idx < n - 1:
        next_idx = min(n - 1, anchor_idx + skip_cap)
        while next_idx > anchor_idx + 1:
            if _line_of_sight_free(path[anchor_idx], path[next_idx], free_mask):
                break
            next_idx -= 1
        smoothed.append(path[next_idx])
        anchor_idx = next_idx

    return smoothed


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


def _save_visibility_png(
    path: pathlib.Path,
    values: List[List[int]],
    bounds: Tuple[float, float, float, float],
    title: str,
) -> None:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        return

    arr = np.array(values, dtype=np.uint8)
    if arr.size <= 0:
        return

    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=float)
    visible = arr == 255
    free_not_visible = arr == 80
    blocked = arr == 0

    rgba[blocked] = [0.10, 0.10, 0.10, 0.20]
    rgba[free_not_visible] = [0.60, 0.60, 0.60, 0.20]
    rgba[visible] = [0.00, 0.70, 0.20, 0.60]

    min_x, min_y, max_x, max_y = bounds
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=180)
    ax.imshow(
        rgba,
        extent=[min_x, max_x, min_y, max_y],
        origin="lower",
        interpolation="nearest",
        zorder=1,
    )
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def evaluate_layout(
    layout: Dict[str, Any],
    baseline_layout: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()

    resolution = as_float(config.get("grid_resolution_m"), 0.10)
    robot_radius = as_float(config.get("robot_radius_m"), 0.30)
    occupancy_exclude_raw = config.get("occupancy_exclude_categories")
    if occupancy_exclude_raw is None:
        occupancy_exclude_raw = ["floor"]
    occupancy_exclude_categories = _to_lower_set(occupancy_exclude_raw)
    start_xy = config.get("start_xy") or [0.8, 0.8]
    goal_xy = config.get("goal_xy") or [5.0, 5.0]

    room_mask, occ, bounds = _build_room_and_occupancy(
        layout, resolution, occupancy_exclude_categories
    )
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

    path_cells_raw = _astar_path(free_mask, start, goal, resolution=resolution)
    path_cells = _smooth_path_cells(
        path_cells_raw,
        free_mask,
        max_skip_cells=max(4, int(round(0.8 / max(resolution, 1e-9)))),
    )

    dist_to_occ = _distance_transform(occ, resolution)
    clr_min = 0.0
    bottleneck_cell: Optional[Cell] = None
    if path_cells_raw:
        clearance_values = []
        for ix, iy in path_cells_raw:
            clearance_values.append((dist_to_occ[iy][ix] - robot_radius, (ix, iy)))
        clr_min, bottleneck_cell = min(clearance_values, key=lambda x: x[0])

    sensor_cells = []
    if start in reachable:
        sensor_cells.append(start)
    sensor_path_cells = path_cells_raw if path_cells_raw else path_cells
    sensor_cells.extend(
        _sample_path_cells(
            sensor_path_cells,
            as_float(config.get("sample_step_m"), 0.50),
            resolution,
            int(config.get("max_sensor_samples", 10)),
        )
    )

    visible_free_path = _make_grid(nx, ny, False)
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
                visible_free_path[iy][ix] = True
                visible_count += 1

    c_vis = (visible_count / free_total) if free_total > 0 else 0.0
    delta_layout = _compute_delta_layout(layout, baseline_layout, as_float(config.get("lambda_rot"), 0.50))

    entry_cfg = config.get("entry_observability")
    entry_cfg = entry_cfg if isinstance(entry_cfg, dict) else {}
    entry_enabled = bool(entry_cfg.get("enabled", False))
    entry_mode = str(entry_cfg.get("mode") or "both").strip().lower()
    if entry_mode not in {"first_hit", "surface", "both"}:
        entry_mode = "both"
    tau_entry_p = as_float(entry_cfg.get("tau_p"), 0.02)
    tau_entry_v = as_float(entry_cfg.get("tau_v"), 0.30)

    free_mask_raw = _build_free_mask(room_mask, occ)
    visible_free_start, free_raw_total, visible_raw_total = _compute_visible_free_from_start(start, free_mask_raw, occ, room_mask)
    c_vis_start = (visible_raw_total / free_raw_total) if free_raw_total > 0 else 0.0

    ooe_c_hit = 0.0
    ooe_r_hit = 0.0
    ooe_c_surf = 0.0
    ooe_r_surf = 0.0
    ooe_per_object: List[Dict[str, Any]] = []
    entry_debug: Dict[str, Any] = {
        "enabled": entry_enabled,
        "mode": entry_mode,
        "tau_p": tau_entry_p,
        "tau_v": tau_entry_v,
        "C_vis_start": c_vis_start,
        "free_total_raw": free_raw_total,
        "visible_total_raw": visible_raw_total,
    }
    if entry_enabled and 0 <= start[0] < nx and 0 <= start[1] < ny:
        target_objects = _collect_entry_target_objects(layout, entry_cfg)
        per_object_map: Dict[str, Dict[str, Any]] = {
            str(target["id"]): {
                "id": str(target["id"]),
                "category": target["category"],
                "p_hit": 0.0,
                "v_surf": 0.0,
                "visible_side_area_m2": 0.0,
                "height_m": as_float(target.get("height_m"), 1.0),
                "recognized_hit": False,
                "recognized_surf": False,
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
                visible_free=visible_free_start,
                entry_cfg=entry_cfg,
            )
            ooe_c_surf = surface_metrics["OOE_C_obj_entry_surf"]
            ooe_r_surf = surface_metrics["OOE_R_rec_entry_surf"]
            for obj_id, item in surface_per_obj.items():
                if obj_id in per_object_map:
                    per_object_map[obj_id].update(item)
            entry_debug["surface"] = surface_debug

        for item in per_object_map.values():
            item["recognized_hit"] = bool(as_float(item.get("p_hit"), 0.0) >= tau_entry_p)
            item["recognized_surf"] = bool(as_float(item.get("v_surf"), 0.0) >= tau_entry_v)

        ooe_per_object = [per_object_map[obj_id] for obj_id in sorted(per_object_map.keys())]
        entry_debug["per_object"] = ooe_per_object

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
        "OOE_enabled": int(entry_enabled),
        "OOE_tau_p": tau_entry_p,
        "OOE_tau_v": tau_entry_v,
        "OOE_C_obj_entry_hit": ooe_c_hit,
        "OOE_R_rec_entry_hit": ooe_r_hit,
        "OOE_C_obj_entry_surf": ooe_c_surf,
        "OOE_R_rec_entry_surf": ooe_r_surf,
        "OOE_per_object": ooe_per_object,
    }

    debug = {
        "bounds": bounds,
        "resolution": resolution,
        "occupancy_exclude_categories": sorted(occupancy_exclude_categories),
        "start_cell": start,
        "goal_cell": goal,
        "path_cells": path_cells,
        "path_cells_raw": path_cells_raw,
        "bottleneck_cell": bottleneck_cell,
        "room_mask": room_mask,
        "occupancy": occ,
        "inflated": inflated,
        "free_mask": free_mask,
        "free_mask_raw": free_mask_raw,
        "reachable": reachable,
        "visible_free_path": visible_free_path,
        "visible_free_start": visible_free_start,
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
    free_mask_raw: Grid = debug.get("free_mask_raw") or free_mask
    reachable: set[Cell] = debug["reachable"]
    path_cells: List[Cell] = debug["path_cells"]
    path_cells_raw: List[Cell] = debug.get("path_cells_raw") or path_cells
    visible_free_path: Grid = debug.get("visible_free_path") or _make_grid(len(room_mask[0]) if room_mask else 0, len(room_mask), False)
    visible_free_start: Grid = debug.get("visible_free_start") or _make_grid(len(room_mask[0]) if room_mask else 0, len(room_mask), False)

    ny = len(room_mask)
    nx = len(room_mask[0]) if ny > 0 else 0

    occ_img = [[128 for _ in range(nx)] for _ in range(ny)]
    reach_img = [[128 for _ in range(nx)] for _ in range(ny)]
    c_vis_img = [[128 for _ in range(nx)] for _ in range(ny)]
    c_vis_start_img = [[128 for _ in range(nx)] for _ in range(ny)]

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

            if free_mask[iy][ix]:
                c_vis_img[iy][ix] = 255 if bool(visible_free_path[iy][ix]) else 80
            else:
                c_vis_img[iy][ix] = 0

            if free_mask_raw[iy][ix]:
                c_vis_start_img[iy][ix] = 255 if bool(visible_free_start[iy][ix]) else 80
            else:
                c_vis_start_img[iy][ix] = 0

    for ix, iy in path_cells:
        if 0 <= ix < nx and 0 <= iy < ny:
            reach_img[iy][ix] = 200

    _grid_to_pgm(out_dir / "occupancy.pgm", occ_img)
    _grid_to_pgm(out_dir / "reachability.pgm", reach_img)
    _grid_to_pgm(out_dir / "c_vis_free.pgm", c_vis_img)
    _grid_to_pgm(out_dir / "c_vis_start_free.pgm", c_vis_start_img)

    bounds = debug.get("bounds")
    resolution = as_float(debug.get("resolution"), 0.0)
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        bbox = (
            as_float(bounds[0], 0.0),
            as_float(bounds[1], 0.0),
            as_float(bounds[2], 0.0),
            as_float(bounds[3], 0.0),
        )
        _save_visibility_png(
            out_dir / "c_vis_area.png",
            c_vis_img,
            bbox,
            "C_vis visible-free area (path sensors)",
        )
        _save_visibility_png(
            out_dir / "c_vis_start_area.png",
            c_vis_start_img,
            bbox,
            "C_vis_start visible-free area (start only)",
        )

    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4 and resolution > 1e-9:
        min_x = as_float(bounds[0], 0.0)
        min_y = as_float(bounds[1], 0.0)
        path_xy = [[min_x + (ix + 0.5) * resolution, min_y + (iy + 0.5) * resolution] for ix, iy in path_cells]
        path_xy_raw = [[min_x + (ix + 0.5) * resolution, min_y + (iy + 0.5) * resolution] for ix, iy in path_cells_raw]
        write_json(
            out_dir / "path_cells.json",
            {
                "bounds": [as_float(bounds[0], 0.0), as_float(bounds[1], 0.0), as_float(bounds[2], 0.0), as_float(bounds[3], 0.0)],
                "resolution": resolution,
                "start_cell": list(debug.get("start_cell") or []),
                "goal_cell": list(debug.get("goal_cell") or []),
                "path_cells_raw": [[int(ix), int(iy)] for ix, iy in path_cells_raw],
                "path_xy_raw": path_xy_raw,
                "path_cells": [[int(ix), int(iy)] for ix, iy in path_cells],
                "path_xy": path_xy,
            },
        )

    if debug.get("task_points") is not None:
        write_json(out_dir / "task_points.json", debug["task_points"])
    entry_debug = debug.get("entry_observability")
    if isinstance(entry_debug, dict):
        write_json(
            out_dir / "entry_observability.json",
            {
                "enabled": bool(entry_debug.get("enabled", False)),
                "mode": str(entry_debug.get("mode") or "both"),
                "tau_p": as_float(entry_debug.get("tau_p"), 0.02),
                "tau_v": as_float(entry_debug.get("tau_v"), 0.30),
                "C_vis_start": as_float(entry_debug.get("C_vis_start"), 0.0),
                "free_total_raw": int(entry_debug.get("free_total_raw", 0)),
                "visible_total_raw": int(entry_debug.get("visible_total_raw", 0)),
            },
        )
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
        if entry_debug.get("per_object") is not None:
            write_json(
                out_dir / "ooe_per_object.json",
                {
                    "tau_p": as_float(entry_debug.get("tau_p"), 0.02),
                    "tau_v": as_float(entry_debug.get("tau_v"), 0.30),
                    "objects": entry_debug.get("per_object"),
                },
            )


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
