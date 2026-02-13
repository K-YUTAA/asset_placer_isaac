from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import subprocess
from typing import Any, Dict, List, Tuple

from layout_tools import (
    extract_json_payload,
    normalize_layout,
    obb_corners_xy,
    read_json,
    read_text,
    utc_now_iso,
    write_json,
)


def _git_commit_hash(repo_root: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _vector_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(v[0], v[1])
    if length < 1e-12:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _project(corners: List[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    values = [_dot(c, axis) for c in corners]
    return min(values), max(values)


def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _obb_overlap(corners_a: List[Tuple[float, float]], corners_b: List[Tuple[float, float]]) -> bool:
    axes: List[Tuple[float, float]] = []
    for corners in (corners_a, corners_b):
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            edge = _vector_sub(p1, p0)
            normal = _normalize((-edge[1], edge[0]))
            if normal != (0.0, 0.0):
                axes.append(normal)

    for axis in axes:
        proj_a = _project(corners_a, axis)
        proj_b = _project(corners_b, axis)
        if not _interval_overlap(proj_a, proj_b):
            return False
    return True


def _build_collision_report(layout_v0: Dict[str, Any]) -> Dict[str, Any]:
    objects = layout_v0.get("objects", [])
    collisions = []

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            corners_a = obb_corners_xy(a)
            corners_b = obb_corners_xy(b)
            if _obb_overlap(corners_a, corners_b):
                collisions.append(
                    {
                        "a": a.get("id"),
                        "b": b.get("id"),
                        "type": "obb_overlap",
                    }
                )

    return {
        "collision_count": len(collisions),
        "collisions": collisions,
    }


def _build_asset_manifest(layout_v0: Dict[str, Any]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {}
    for obj in layout_v0.get("objects", []):
        obj_id = str(obj.get("id"))
        query = str(obj.get("asset_query") or obj.get("category") or "object")
        asset_id = str(obj.get("asset_id") or "")
        entry = {
            "asset_query": query,
            "chosen": {"asset_id": asset_id, "score": 1.0 if asset_id else 0.0},
            "topk": [{"asset_id": asset_id, "score": 1.0}] if asset_id else [],
        }
        manifest[obj_id] = entry
    return manifest


def _load_or_generate_raw_payload(args: argparse.Namespace, out_dir: pathlib.Path) -> Dict[str, Any]:
    raw_path = out_dir / "llm_response_raw.json"

    if args.llm_cache_mode == "read":
        if not raw_path.exists():
            raise FileNotFoundError(f"llm_cache_mode=read but cache is missing: {raw_path}")
        payload = read_json(raw_path)
        if "text" not in payload:
            payload = {"text": json.dumps(payload, ensure_ascii=False), "cache_mode": "read"}
        return payload

    if args.layout_input:
        layout_input_path = pathlib.Path(args.layout_input)
        text = read_text(layout_input_path)
        payload = {
            "text": text,
            "source": str(layout_input_path),
            "cache_mode": "write",
        }
    else:
        hints_text = ""
        hints_path = pathlib.Path(args.hints_path)
        if hints_path.exists() and hints_path.suffix.lower() in {".txt", ".md", ".json"}:
            hints_text = read_text(hints_path)

        synthetic = {
            "area_name": f"synthetic_{pathlib.Path(args.sketch_path).stem}",
            "area_size_X": 8.0,
            "area_size_Y": 8.0,
            "area_objects_list": [],
            "notes": hints_text[:2000],
        }
        payload = {
            "text": json.dumps(synthetic, ensure_ascii=False, indent=2),
            "source": "synthetic",
            "cache_mode": "write",
        }

    write_json(raw_path, payload)
    return payload


def run_v0_freeze(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    raw_payload = _load_or_generate_raw_payload(args, out_dir)
    raw_text = str(raw_payload.get("text", ""))
    parsed = extract_json_payload(raw_text)

    layout_id = args.layout_id or pathlib.Path(args.sketch_path).stem or "layout"
    layout_llm = parsed
    layout_v0 = normalize_layout(layout_llm, layout_id=layout_id, source="v0")

    llm_response_parsed = {
        "ok": True,
        "parsed_at": utc_now_iso(),
        "layout_id": layout_id,
        "payload": layout_llm,
    }

    collision_report = _build_collision_report(layout_v0)
    asset_manifest = _build_asset_manifest(layout_v0)

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    commit_hash = _git_commit_hash(repo_root)

    run_manifest = {
        "created_at": utc_now_iso(),
        "layout_id": layout_id,
        "model": args.model,
        "prompt": args.prompt_name,
        "reasoning": args.reasoning,
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "llm_cache_mode": args.llm_cache_mode,
        "inputs": {
            "sketch_path": str(args.sketch_path),
            "hints_path": str(args.hints_path),
            "layout_input": str(args.layout_input) if args.layout_input else None,
        },
        "commit_hash": commit_hash,
        "eval_params": {
            "grid_resolution_m": args.grid_resolution,
            "robot_radius_m": args.robot_radius,
            "start_xy": [args.start_x, args.start_y],
            "goal_xy": [args.goal_x, args.goal_y],
        },
        "placement_params": {
            "max_iterations": args.max_iterations,
            "push_step_m": args.push_step,
            "placement_order": args.placement_order,
        },
    }

    write_json(out_dir / "layout_llm.json", layout_llm)
    write_json(out_dir / "layout_v0.json", layout_v0)
    write_json(out_dir / "llm_response_parsed.json", llm_response_parsed)
    write_json(out_dir / "asset_manifest.json", asset_manifest)
    write_json(out_dir / "collision_report.json", collision_report)
    write_json(out_dir / "run_manifest.json", run_manifest)

    return {
        "layout_llm": str(out_dir / "layout_llm.json"),
        "layout_v0": str(out_dir / "layout_v0.json"),
        "collision_report": str(out_dir / "collision_report.json"),
        "run_manifest": str(out_dir / "run_manifest.json"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create reproducible v0 baseline artifacts")
    parser.add_argument("--sketch_path", required=True)
    parser.add_argument("--hints_path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--llm_cache_mode", choices=["write", "read"], default="write")

    parser.add_argument("--layout_input", default=None, help="Optional JSON input used instead of live LLM call")
    parser.add_argument("--layout_id", default=None)

    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--prompt_name", default="prompt_2")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--grid_resolution", type=float, default=0.10)
    parser.add_argument("--robot_radius", type=float, default=0.30)
    parser.add_argument("--start_x", type=float, default=0.8)
    parser.add_argument("--start_y", type=float, default=0.8)
    parser.add_argument("--goal_x", type=float, default=5.0)
    parser.add_argument("--goal_y", type=float, default=5.0)

    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--push_step", type=float, default=0.10)
    parser.add_argument("--placement_order", default="category_then_area")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    outputs = run_v0_freeze(args)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
