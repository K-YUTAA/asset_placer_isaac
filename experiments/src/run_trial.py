from __future__ import annotations

import argparse
import csv
import json
import pathlib
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict

from eval_metrics import default_eval_config, evaluate_layout, merge_eval_config
from layout_tools import load_layout_contract, write_json
from refine_heuristic import run_refinement
from run_v0_freeze import run_v0_freeze


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _append_metrics_csv(csv_path: pathlib.Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "layout_id",
        "method",
        "seed",
        "status",
        "error_msg",
        "C_vis",
        "R_reach",
        "clr_min",
        "Delta_layout",
        "Adopt",
        "validity",
        "runtime_sec",
        "run_dir",
    ]

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_trial(args: argparse.Namespace) -> Dict[str, Any]:
    trial_cfg = json.loads(pathlib.Path(args.trial_config).read_text(encoding="utf-8-sig"))

    trial_id = str(trial_cfg.get("trial_id") or f"trial_{_timestamp()}")
    layout_id = str(trial_cfg.get("layout_id") or "layout")
    method = str(trial_cfg.get("method") or "original")
    seed = int(trial_cfg.get("seed", 0))

    run_dir = pathlib.Path(args.out_root) / f"{trial_id}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_eval_config()
    if args.eval_config:
        cfg = merge_eval_config(cfg, json.loads(pathlib.Path(args.eval_config).read_text(encoding="utf-8-sig")))
    cfg = merge_eval_config(cfg, trial_cfg.get("eval", {}))

    inputs = trial_cfg.get("inputs", {})

    try:
        freeze_args = Namespace(
            sketch_path=str(inputs.get("sketch_path", "")),
            hints_path=str(inputs.get("hints_path", "")),
            seed=seed,
            out_dir=str(run_dir / "v0"),
            llm_cache_mode=str(trial_cfg.get("llm_cache_mode", "write")),
            layout_input=inputs.get("layout_input"),
            layout_id=layout_id,
            model=str(trial_cfg.get("model", "gpt-5.2")),
            prompt_1_name=str(trial_cfg.get("prompt_1_name", "prompt_1")),
            prompt_2_name=str(trial_cfg.get("prompt_2_name", trial_cfg.get("prompt_name", "prompt_2"))),
            prompt_name=str(trial_cfg.get("prompt_name", "prompt_2")),
            reasoning=str(trial_cfg.get("reasoning", "high")),
            temperature=float(trial_cfg.get("temperature", 0.0)),
            top_p=float(trial_cfg.get("top_p", 1.0)),
            grid_resolution=float(cfg.get("grid_resolution_m", 0.1)),
            robot_radius=float(cfg.get("robot_radius_m", 0.3)),
            start_x=float((cfg.get("start_xy") or [0.8, 0.8])[0]),
            start_y=float((cfg.get("start_xy") or [0.8, 0.8])[1]),
            goal_x=float((cfg.get("goal_xy") or [5.0, 5.0])[0]),
            goal_y=float((cfg.get("goal_xy") or [5.0, 5.0])[1]),
            max_iterations=int(trial_cfg.get("placement_max_iterations", 30)),
            push_step=float(trial_cfg.get("placement_push_step", 0.1)),
            placement_order=str(trial_cfg.get("placement_order", "category_then_area")),
        )

        v0_outputs = run_v0_freeze(freeze_args)
        layout_v0 = load_layout_contract(pathlib.Path(v0_outputs["layout_v0"]))

        baseline_layout = layout_v0
        selected_layout = layout_v0
        refine_log = None
        baseline_task_points = None
        if isinstance(cfg.get("task"), dict) and cfg.get("task"):
            _, baseline_debug = evaluate_layout(layout_v0, baseline_layout, cfg)
            baseline_task_points = baseline_debug.get("task_points")
            if isinstance(baseline_task_points, dict):
                cfg["start_xy"] = (baseline_task_points.get("start") or {}).get("xy") or cfg.get("start_xy")
                cfg["goal_xy"] = (baseline_task_points.get("goal") or {}).get("xy") or cfg.get("goal_xy")

        if method in {"heuristic", "proposed"}:
            refined_layout, refined_metrics, refine_steps = run_refinement(
                layout=layout_v0,
                baseline_layout=baseline_layout,
                config=cfg,
                max_iterations=int(trial_cfg.get("refine_max_iterations", 30)),
                step_m=float(trial_cfg.get("refine_step_m", 0.1)),
                rot_deg=float(trial_cfg.get("refine_rot_deg", 15.0)),
                max_changed_objects=int(trial_cfg.get("refine_max_changed_objects", 3)),
            )
            selected_layout = refined_layout
            write_json(run_dir / "layout_refined.json", refined_layout)
            write_json(run_dir / "metrics_refined.json", refined_metrics)
            refine_log = {"steps": refine_steps}
            write_json(run_dir / "refine_log.json", refine_log)

        metrics, debug = evaluate_layout(selected_layout, baseline_layout, cfg)
        write_json(run_dir / "metrics.json", metrics)

        summary = {
            "trial_id": trial_id,
            "layout_id": layout_id,
            "method": method,
            "seed": seed,
            "status": "ok",
            "error_msg": "",
            "run_dir": str(run_dir),
            **metrics,
        }

        write_json(
            run_dir / "trial_manifest.json",
            {
                "trial_config": trial_cfg,
                "resolved_eval_config": cfg,
                "v0_outputs": v0_outputs,
                "summary": summary,
                "refine_log": refine_log,
                "debug_meta": {
                    "path_length_cells": len(debug.get("path_cells") or []),
                    "bottleneck_cell": debug.get("bottleneck_cell"),
                    "task_points": baseline_task_points,
                    "task_points_final": debug.get("task_points"),
                },
            },
        )

    except Exception as exc:
        summary = {
            "trial_id": trial_id,
            "layout_id": layout_id,
            "method": method,
            "seed": seed,
            "status": "error",
            "error_msg": str(exc),
            "run_dir": str(run_dir),
            "C_vis": "",
            "R_reach": "",
            "clr_min": "",
            "Delta_layout": "",
            "Adopt": "",
            "validity": "",
            "runtime_sec": "",
        }
        write_json(run_dir / "trial_error.json", summary)

    _append_metrics_csv(pathlib.Path(args.out_root) / "metrics.csv", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one trial: v0 -> eval -> optional refine")
    parser.add_argument("--trial_config", required=True)
    parser.add_argument("--eval_config", default=None)
    parser.add_argument("--out_root", required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_trial(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()

