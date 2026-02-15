from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Dict, List, Set, Tuple

from layout_tools import (
    as_float,
    load_layout_contract,
    point_in_obb,
    point_in_polygon,
    room_bbox,
    wrap_angle,
    yaw_to_rad,
    write_json,
)


def _object_map(layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(obj.get("id")): obj for obj in layout.get("objects", [])}


def _compute_level1(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    map_a = _object_map(a)
    map_b = _object_map(b)
    ids_a = set(map_a.keys())
    ids_b = set(map_b.keys())

    id_match = ids_a == ids_b
    count_match = len(ids_a) == len(ids_b)

    asset_mismatches = []
    for obj_id in sorted(ids_a & ids_b):
        asset_a = str(map_a[obj_id].get("asset_id") or "")
        asset_b = str(map_b[obj_id].get("asset_id") or "")
        if asset_a != asset_b:
            asset_mismatches.append({"id": obj_id, "asset_a": asset_a, "asset_b": asset_b})

    return {
        "pass": bool(id_match and count_match and not asset_mismatches),
        "id_match": id_match,
        "count_match": count_match,
        "asset_match": len(asset_mismatches) == 0,
        "asset_mismatches": asset_mismatches,
        "count_a": len(ids_a),
        "count_b": len(ids_b),
    }


def _compute_level2(
    a: Dict[str, Any],
    b: Dict[str, Any],
    pos_tol_m: float,
    yaw_tol_rad: float,
) -> Dict[str, Any]:
    map_a = _object_map(a)
    map_b = _object_map(b)

    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    deltas = []
    pass_all = True

    for obj_id in common_ids:
        pose_a = map_a[obj_id].get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
        pose_b = map_b[obj_id].get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]

        dx = as_float(pose_a[0], 0.0) - as_float(pose_b[0], 0.0)
        dy = as_float(pose_a[1], 0.0) - as_float(pose_b[1], 0.0)
        delta_p = math.hypot(dx, dy)

        yaw_a = yaw_to_rad(pose_a[3] if len(pose_a) > 3 else 0.0)
        yaw_b = yaw_to_rad(pose_b[3] if len(pose_b) > 3 else 0.0)
        delta_theta = abs(wrap_angle(yaw_a - yaw_b))

        pos_ok = delta_p < pos_tol_m
        yaw_ok = delta_theta < yaw_tol_rad
        pass_all = pass_all and pos_ok and yaw_ok

        deltas.append(
            {
                "id": obj_id,
                "delta_position_m": delta_p,
                "delta_yaw_rad": delta_theta,
                "position_ok": pos_ok,
                "yaw_ok": yaw_ok,
            }
        )

    return {
        "pass": pass_all,
        "pos_tol_m": pos_tol_m,
        "yaw_tol_rad": yaw_tol_rad,
        "objects": deltas,
    }


def _layout_occupied_cells(
    layout: Dict[str, Any],
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> Set[Tuple[int, int]]:
    min_x, min_y, max_x, max_y = bounds
    nx = max(1, int(math.ceil((max_x - min_x) / resolution)))
    ny = max(1, int(math.ceil((max_y - min_y) / resolution)))

    room_poly = layout["room"]["boundary_poly_xy"]
    occupied: Set[Tuple[int, int]] = set()

    for iy in range(ny):
        y = min_y + (iy + 0.5) * resolution
        for ix in range(nx):
            x = min_x + (ix + 0.5) * resolution
            if not point_in_polygon(x, y, room_poly):
                continue
            for obj in layout.get("objects", []):
                if point_in_obb(x, y, obj):
                    occupied.add((ix, iy))
                    break
    return occupied


def _compute_iou(a: Dict[str, Any], b: Dict[str, Any], resolution: float) -> float:
    a_bounds = room_bbox(a["room"]["boundary_poly_xy"])
    b_bounds = room_bbox(b["room"]["boundary_poly_xy"])

    bounds = (
        min(a_bounds[0], b_bounds[0]),
        min(a_bounds[1], b_bounds[1]),
        max(a_bounds[2], b_bounds[2]),
        max(a_bounds[3], b_bounds[3]),
    )

    occ_a = _layout_occupied_cells(a, bounds, resolution)
    occ_b = _layout_occupied_cells(b, bounds, resolution)

    union = occ_a | occ_b
    if not union:
        return 1.0
    inter = occ_a & occ_b
    return len(inter) / float(len(union))


def compare_layouts(args: argparse.Namespace) -> Dict[str, Any]:
    layout_a = load_layout_contract(pathlib.Path(args.layout_a))
    layout_b = load_layout_contract(pathlib.Path(args.layout_b))

    level1 = _compute_level1(layout_a, layout_b)
    level2 = _compute_level2(layout_a, layout_b, args.pos_tol_m, math.radians(args.yaw_tol_deg))

    iou = _compute_iou(layout_a, layout_b, args.grid_resolution_m)
    iou_pass = iou > args.iou_threshold

    passed = bool(level1["pass"] and level2["pass"] and iou_pass)

    report = {
        "pass": passed,
        "level1": level1,
        "level2": level2,
        "occupancy_iou": {
            "value": iou,
            "threshold": args.iou_threshold,
            "pass": iou_pass,
        },
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two layout JSON files for reproducibility checks")
    parser.add_argument("--layout_a", required=True)
    parser.add_argument("--layout_b", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--pos_tol_m", type=float, default=0.02)
    parser.add_argument("--yaw_tol_deg", type=float, default=1.0)
    parser.add_argument("--grid_resolution_m", type=float, default=0.10)
    parser.add_argument("--iou_threshold", type=float, default=0.98)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    report = compare_layouts(args)
    if args.out:
        write_json(pathlib.Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
