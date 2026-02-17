from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

from eval_metrics import default_eval_config, evaluate_layout, merge_eval_config
from layout_tools import as_float, load_layout_contract, obb_corners_xy, point_in_polygon, write_json


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _score(metrics: Dict[str, Any], alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0, gamma: float = 0.5) -> float:
    cmax = 1.0
    penalty = 0.0
    if metrics.get("validity", 0) == 0:
        penalty += 5.0
    if metrics.get("R_reach", 0.0) <= 0.0:
        penalty += 2.0

    return (
        alpha * as_float(metrics.get("C_vis"), 0.0)
        + beta * as_float(metrics.get("R_reach"), 0.0)
        + eta * max(0.0, min(cmax, as_float(metrics.get("clr_min"), 0.0)))
        - gamma * as_float(metrics.get("Delta_layout"), 0.0)
        - penalty
    )


def _find_object(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if obj.get("id") == object_id:
            return obj
    return None


def _object_inside_room(obj: Dict[str, Any], room_poly: List[List[float]]) -> bool:
    corners = obb_corners_xy(obj)
    for x, y in corners:
        if not point_in_polygon(x, y, room_poly):
            return False
    return True


def _select_target_object(
    layout: Dict[str, Any],
    debug: Dict[str, Any],
    config: Dict[str, Any],
    changed_ids: Set[str],
    max_changed_objects: int,
) -> Optional[str]:
    movable = [obj for obj in layout.get("objects", []) if bool(obj.get("movable", True))]
    if not movable:
        return None

    if len(changed_ids) >= max_changed_objects:
        movable = [obj for obj in movable if obj.get("id") in changed_ids]
        if not movable:
            return None

    resolution = as_float(config.get("grid_resolution_m"), 0.1)
    bounds = debug["bounds"]
    bottleneck = debug.get("bottleneck_cell")

    if bottleneck is not None:
        bx = bounds[0] + (bottleneck[0] + 0.5) * resolution
        by = bounds[1] + (bottleneck[1] + 0.5) * resolution

        best = None
        best_dist = float("inf")
        for obj in movable:
            pose = obj.get("pose_xyz_yaw", [0.0, 0.0, 0.0, 0.0])
            dist = math.hypot(as_float(pose[0], 0.0) - bx, as_float(pose[1], 0.0) - by)
            if dist < best_dist:
                best_dist = dist
                best = obj.get("id")
        return best

    start_xy = None
    goal_xy = None
    task_points = debug.get("task_points")
    if isinstance(task_points, dict):
        start = (task_points.get("start") or {}).get("xy")
        goal = (task_points.get("goal") or {}).get("xy")
        if isinstance(start, (list, tuple)) and len(start) >= 2 and isinstance(goal, (list, tuple)) and len(goal) >= 2:
            start_xy = start
            goal_xy = goal

    if start_xy is None:
        start_xy = config.get("start_xy") or [0.8, 0.8]
    if goal_xy is None:
        goal_xy = config.get("goal_xy") or [5.0, 5.0]
    ax, ay = as_float(start_xy[0], 0.8), as_float(start_xy[1], 0.8)
    bx, by = as_float(goal_xy[0], 5.0), as_float(goal_xy[1], 5.0)

    best = None
    best_dist = float("inf")
    for obj in movable:
        pose = obj.get("pose_xyz_yaw", [0.0, 0.0, 0.0, 0.0])
        px, py = as_float(pose[0], 0.0), as_float(pose[1], 0.0)
        dist = _point_to_segment_distance(px, py, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best = obj.get("id")
    return best


def run_refinement(
    layout: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    config: Dict[str, Any],
    max_iterations: int,
    step_m: float,
    rot_deg: float,
    max_changed_objects: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    current = copy.deepcopy(layout)
    current_metrics, current_debug = evaluate_layout(current, baseline_layout, config)
    current_score = _score(current_metrics)

    room_poly = current["room"]["boundary_poly_xy"]
    changed_ids: Set[str] = set()
    logs: List[Dict[str, Any]] = []

    rot_rad = math.radians(rot_deg)

    for iteration in range(1, max_iterations + 1):
        target_id = _select_target_object(current, current_debug, config, changed_ids, max_changed_objects)
        if target_id is None:
            break

        base_obj = _find_object(current, target_id)
        if base_obj is None:
            break

        best_layout = None
        best_metrics = None
        best_debug = None
        best_score = current_score
        best_move = None

        for dx in (-step_m, 0.0, step_m):
            for dy in (-step_m, 0.0, step_m):
                for dtheta in (-rot_rad, 0.0, rot_rad):
                    if abs(dx) < 1e-12 and abs(dy) < 1e-12 and abs(dtheta) < 1e-12:
                        continue

                    candidate = copy.deepcopy(current)
                    obj = _find_object(candidate, target_id)
                    if obj is None:
                        continue

                    pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
                    pose[0] = as_float(pose[0], 0.0) + dx
                    pose[1] = as_float(pose[1], 0.0) + dy
                    pose[3] = as_float(pose[3], 0.0) + dtheta
                    obj["pose_xyz_yaw"] = pose

                    if not _object_inside_room(obj, room_poly):
                        continue

                    metrics, debug = evaluate_layout(candidate, baseline_layout, config)
                    if metrics.get("validity", 0) == 0:
                        continue

                    # Non-regression guard on reachability/clearance.
                    if as_float(metrics.get("R_reach"), 0.0) + 1e-9 < as_float(current_metrics.get("R_reach"), 0.0):
                        continue
                    if as_float(metrics.get("clr_min"), 0.0) + 1e-9 < as_float(current_metrics.get("clr_min"), 0.0):
                        continue

                    score = _score(metrics)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_layout = candidate
                        best_metrics = metrics
                        best_debug = debug
                        best_move = {"object_id": target_id, "dx": dx, "dy": dy, "dtheta_deg": math.degrees(dtheta)}

        if best_layout is None:
            break

        logs.append(
            {
                "iteration": iteration,
                "target_id": target_id,
                "move": best_move,
                "before": current_metrics,
                "after": best_metrics,
                "score_before": current_score,
                "score_after": best_score,
            }
        )

        changed_ids.add(target_id)
        current = best_layout
        current_metrics = best_metrics
        current_debug = best_debug
        current_score = best_score

    return current, current_metrics, logs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine layout with heuristic local search")
    parser.add_argument("--layout_in", required=True)
    parser.add_argument("--layout_out", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--log_out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--step_m", type=float, default=0.10)
    parser.add_argument("--rot_deg", type=float, default=15.0)
    parser.add_argument("--max_changed_objects", type=int, default=3)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = default_eval_config()
    if args.config:
        config = merge_eval_config(config, json.loads(pathlib.Path(args.config).read_text(encoding="utf-8-sig")))

    layout = load_layout_contract(pathlib.Path(args.layout_in))
    baseline = copy.deepcopy(layout)

    refined, metrics, logs = run_refinement(
        layout=layout,
        baseline_layout=baseline,
        config=config,
        max_iterations=args.max_iterations,
        step_m=args.step_m,
        rot_deg=args.rot_deg,
        max_changed_objects=args.max_changed_objects,
    )

    write_json(pathlib.Path(args.layout_out), refined)
    write_json(pathlib.Path(args.metrics_out), metrics)
    write_json(pathlib.Path(args.log_out), {"steps": logs})

    print(json.dumps({"metrics": metrics, "log_steps": len(logs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

