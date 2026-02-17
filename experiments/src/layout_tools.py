from __future__ import annotations

import json
import math
import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

TWO_PI = 2.0 * math.pi


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def extract_json_payload(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text payload")
    return json.loads(text[start : end + 1])


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def yaw_to_rad(value: Any) -> float:
    raw = as_float(value, 0.0)
    if abs(raw) > TWO_PI + 1e-6:
        return math.radians(raw)
    return raw


def wrap_angle(angle_rad: float) -> float:
    return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi


def _slug_category(value: str) -> str:
    text = (value or "object").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text or "object"


def _build_room(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw.get("room"), dict):
        room = raw["room"]
        boundary = room.get("boundary_poly_xy") or []
        if boundary:
            return {
                "boundary_poly_xy": [[as_float(p[0]), as_float(p[1])] for p in boundary],
                "ceiling_height_m": as_float(room.get("ceiling_height_m"), 2.4),
            }

    outer_polygon = raw.get("outer_polygon")
    if isinstance(outer_polygon, list) and outer_polygon:
        boundary = []
        for p in outer_polygon:
            if isinstance(p, dict):
                boundary.append([as_float(p.get("X"), 0.0), as_float(p.get("Y"), 0.0)])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                boundary.append([as_float(p[0], 0.0), as_float(p[1], 0.0)])
        if boundary:
            return {
                "boundary_poly_xy": boundary,
                "ceiling_height_m": as_float(raw.get("ceiling_height_m"), 2.4),
            }

    sx = as_float(raw.get("area_size_X", raw.get("area_size_x", raw.get("area_width", 8.0))), 8.0)
    sy = as_float(raw.get("area_size_Y", raw.get("area_size_y", raw.get("area_height", 8.0))), 8.0)
    return {
        "boundary_poly_xy": [[0.0, 0.0], [sx, 0.0], [sx, sy], [0.0, sy]],
        "ceiling_height_m": 2.4,
    }


def _normalize_polygon_points(raw_poly: Any) -> List[List[float]]:
    points: List[List[float]] = []
    if not isinstance(raw_poly, list):
        return points
    for p in raw_poly:
        if isinstance(p, dict):
            x = as_float(p.get("X", p.get("x", 0.0)), 0.0)
            y = as_float(p.get("Y", p.get("y", 0.0)), 0.0)
            points.append([x, y])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            points.append([as_float(p[0], 0.0), as_float(p[1], 0.0)])
    return points


def _extract_rooms(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rooms = raw.get("rooms")
    if not isinstance(raw_rooms, list):
        return []

    rooms: List[Dict[str, Any]] = []
    for idx, room in enumerate(raw_rooms):
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("room_id") or room.get("room_name") or f"room_{idx+1}")
        room_name = str(room.get("room_name") or room_id)

        raw_poly = room.get("room_polygon")
        if not isinstance(raw_poly, list):
            raw_poly = room.get("boundary_poly_xy")
        poly = _normalize_polygon_points(raw_poly)
        if len(poly) < 3:
            continue

        openings: List[Dict[str, Any]] = []
        raw_openings = room.get("openings")
        if isinstance(raw_openings, list):
            for opening in raw_openings:
                if not isinstance(opening, dict):
                    continue
                openings.append(
                    {
                        "type": str(opening.get("type") or "opening"),
                        "X": as_float(opening.get("X", opening.get("x", 0.0)), 0.0),
                        "Y": as_float(opening.get("Y", opening.get("y", 0.0)), 0.0),
                        "Width": as_float(opening.get("Width", opening.get("width", 0.0)), 0.0),
                        "Height": as_float(opening.get("Height", opening.get("height", 0.0)), 0.0),
                        "SillHeight": as_float(opening.get("SillHeight", opening.get("sillHeight", 0.0)), 0.0),
                    }
                )

        rooms.append(
            {
                "room_id": room_id,
                "room_name": room_name,
                "room_polygon": poly,
                "openings": openings,
            }
        )

    return rooms


def _map_extension_object(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    category = str(
        raw_obj.get("category")
        or raw_obj.get("object_category")
        or raw_obj.get("object_type")
        or raw_obj.get("object_name")
        or "object"
    )

    length = as_float(raw_obj.get("Length", raw_obj.get("length", 1.0)), 1.0)
    width = as_float(raw_obj.get("Width", raw_obj.get("width", 1.0)), 1.0)
    height = as_float(raw_obj.get("Height", raw_obj.get("height", 1.0)), 1.0)

    x = as_float(raw_obj.get("X", raw_obj.get("x", 0.0)), 0.0)
    y = as_float(raw_obj.get("Y", raw_obj.get("y", 0.0)), 0.0)
    z = as_float(raw_obj.get("Z", raw_obj.get("z", 0.0)), 0.0)
    yaw = yaw_to_rad(raw_obj.get("rotationZ", raw_obj.get("yaw", 0.0)))

    return {
        "id": raw_obj.get("id"),
        "category": category,
        "asset_query": str(raw_obj.get("asset_query") or raw_obj.get("object_name") or category),
        "asset_id": str(raw_obj.get("asset_id") or ""),
        "scale": list(raw_obj.get("scale") or [1.0, 1.0, 1.0]),
        "size_lwh_m": [length, width, height],
        "pose_xyz_yaw": [x, y, z, yaw],
        "movable": bool(raw_obj.get("movable", True)),
    }


def _map_contract_object(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    size = raw_obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
    pose = raw_obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]

    length = as_float(size[0] if len(size) > 0 else 1.0, 1.0)
    width = as_float(size[1] if len(size) > 1 else 1.0, 1.0)
    height = as_float(size[2] if len(size) > 2 else 1.0, 1.0)

    x = as_float(pose[0] if len(pose) > 0 else 0.0, 0.0)
    y = as_float(pose[1] if len(pose) > 1 else 0.0, 0.0)
    z = as_float(pose[2] if len(pose) > 2 else 0.0, 0.0)
    yaw = yaw_to_rad(pose[3] if len(pose) > 3 else 0.0)

    return {
        "id": raw_obj.get("id"),
        "category": str(raw_obj.get("category") or "object"),
        "asset_query": str(raw_obj.get("asset_query") or raw_obj.get("category") or "object"),
        "asset_id": str(raw_obj.get("asset_id") or ""),
        "scale": list(raw_obj.get("scale") or [1.0, 1.0, 1.0]),
        "size_lwh_m": [length, width, height],
        "pose_xyz_yaw": [x, y, z, yaw],
        "movable": bool(raw_obj.get("movable", True)),
    }


def _normalize_ids(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objects:
        category = _slug_category(str(obj.get("category") or "object"))
        grouped.setdefault(category, []).append(obj)

    normalized: List[Dict[str, Any]] = []
    for category in sorted(grouped.keys()):
        group = grouped[category]
        group.sort(
            key=lambda x: (
                as_float((x.get("pose_xyz_yaw") or [0.0])[0], 0.0),
                as_float((x.get("pose_xyz_yaw") or [0.0, 0.0])[1], 0.0),
            )
        )
        for idx, obj in enumerate(group):
            copied = dict(obj)
            copied["id"] = f"{category}_{idx:02d}"
            normalized.append(copied)
    return normalized


def normalize_layout(raw_layout: Dict[str, Any], layout_id: str | None = None, source: str = "v0") -> Dict[str, Any]:
    room = _build_room(raw_layout)
    rooms = _extract_rooms(raw_layout)

    raw_objects: List[Dict[str, Any]] = []
    if isinstance(raw_layout.get("objects"), list):
        raw_objects = [o for o in raw_layout["objects"] if isinstance(o, dict)]
    elif isinstance(raw_layout.get("area_objects_list"), list):
        raw_objects = [o for o in raw_layout["area_objects_list"] if isinstance(o, dict)]

    objects: List[Dict[str, Any]] = []
    for raw_obj in raw_objects:
        if "pose_xyz_yaw" in raw_obj or "size_lwh_m" in raw_obj:
            objects.append(_map_contract_object(raw_obj))
        else:
            objects.append(_map_extension_object(raw_obj))

    objects = _normalize_ids(objects)

    layout_name = layout_id or str(raw_layout.get("layout_id") or raw_layout.get("area_name") or "layout_unknown")

    return {
        "meta": {
            "layout_id": layout_name,
            "source": source,
            "unit": "m",
            "timestamp": utc_now_iso(),
        },
        "room": room,
        "rooms": rooms,
        "objects": objects,
    }


def load_layout_contract(layout_path: pathlib.Path) -> Dict[str, Any]:
    raw = read_json(layout_path)
    if isinstance(raw.get("meta"), dict) and isinstance(raw.get("room"), dict) and isinstance(raw.get("objects"), list):
        return normalize_layout(raw, layout_id=raw.get("meta", {}).get("layout_id"), source=raw.get("meta", {}).get("source", "v0"))
    return normalize_layout(raw, layout_id=layout_path.stem, source="v0")


def room_bbox(boundary_poly_xy: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [as_float(p[0], 0.0) for p in boundary_poly_xy]
    ys = [as_float(p[1], 0.0) for p in boundary_poly_xy]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = as_float(polygon[i][0], 0.0), as_float(polygon[i][1], 0.0)
        xj, yj = as_float(polygon[j][0], 0.0), as_float(polygon[j][1], 0.0)

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def obb_corners_xy(obj: Dict[str, Any]) -> List[Tuple[float, float]]:
    size = obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]

    length = as_float(size[0], 1.0)
    width = as_float(size[1], 1.0)
    x, y, _, yaw = as_float(pose[0], 0.0), as_float(pose[1], 0.0), as_float(pose[2], 0.0), yaw_to_rad(pose[3])

    hx = 0.5 * length
    hy = 0.5 * width
    local = [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    corners: List[Tuple[float, float]] = []
    for lx, ly in local:
        wx = x + cos_yaw * lx - sin_yaw * ly
        wy = y + sin_yaw * lx + cos_yaw * ly
        corners.append((wx, wy))
    return corners


def point_in_obb(x: float, y: float, obj: Dict[str, Any]) -> bool:
    size = obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]

    length = as_float(size[0], 1.0)
    width = as_float(size[1], 1.0)
    cx, cy, _, yaw = as_float(pose[0], 0.0), as_float(pose[1], 0.0), as_float(pose[2], 0.0), yaw_to_rad(pose[3])

    dx = x - cx
    dy = y - cy
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)

    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy

    return abs(local_x) <= 0.5 * length and abs(local_y) <= 0.5 * width

