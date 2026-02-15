# Experiments Workspace (Eval Loop v1)

This directory is the experiment layer described in
`2026-02-13-experiment-plan-step-by-step.md`.

It is intentionally isolated from the extension runtime code so that:
- baseline generation and evaluation loops can be reproduced;
- heuristic/optimization experiments can iterate quickly;
- results can be compared and exported for paper figures.

## Directory Layout

```text
experiments/
  README.md
  fixtures/
    sketches/
    hints/
  configs/
    trials/
    eval/
  src/
    layout_tools.py
    task_points.py
    run_v0_freeze.py
    compare_layout.py
    eval_metrics.py
    refine_heuristic.py
    run_trial.py
  baselines/
  results/
  runs/
  cache/
```

## Quick Start

1. Create baseline artifacts (v0 freeze):
```bash
python experiments/src/run_v0_freeze.py \
  --sketch_path experiments/fixtures/sketches/example.png \
  --hints_path experiments/fixtures/hints/example.txt \
  --layout_input json/living_room2_layout_gpt-5.2_202601291702.json \
  --seed 1 \
  --out_dir experiments/runs/demo_v0 \
  --llm_cache_mode write
```

2. Compare reproducibility:
```bash
python experiments/src/compare_layout.py \
  --layout_a experiments/runs/demo_v0/layout_v0.json \
  --layout_b experiments/runs/demo_v0/layout_v0.json \
  --out experiments/runs/demo_v0/compare_report.json
```

3. Evaluate metrics:
```bash
python experiments/src/eval_metrics.py \
  --layout experiments/runs/demo_v0/layout_v0.json \
  --config experiments/configs/eval/default_eval.json \
  --out experiments/runs/demo_v0/metrics.json \
  --debug_dir experiments/runs/demo_v0/debug
```

4. Run one trial:
```bash
python experiments/src/run_trial.py \
  --trial_config experiments/configs/trials/sample_exp_a.json \
  --eval_config experiments/configs/eval/default_eval.json \
  --out_root experiments/results
```

## Notes

- This layer supports both extension-style layout JSON (`area_objects_list`) and
  normalized experiment JSON (`room` + `objects`).
- `eval_metrics.py` supports `eval.task` to auto-resolve start/goal (snapped to free cells).
  - `--debug_dir` also writes `task_points.json` alongside the PGM maps.
- `run_trial.py` stores the resolved points in `trial_manifest.json` (`debug_meta.task_points`).
- `run_v0_freeze.py` supports `llm_cache_mode=read/write` for deterministic re-runs.
- Scripts use only Python standard library for portability.
