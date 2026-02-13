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

Grid = List[List[bool]]
Cell = Tuple[int, int]


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
    }


def merge_eval_config(base: Dict[str, Any], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if update:
        merged.update(update)
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

