# WSL + Isaac Sim Development Rules

## Goal
- Develop Isaac Sim extensions from WSL.
- Run Isaac Sim on Windows only.

## Canonical Paths
- Edit project files from WSL using:
  - `/home/sskk/win/isaacsim`
  - (symlink to `/mnt/c/Users/tmaru/isaacsim`)
- Prefer this WSL path in commands and file references.

## Runtime Policy
- Do not install or run Isaac Sim inside WSL.
- Launch Windows Isaac Sim only, using one of:
  - `/mnt/c/Users/tmaru/isaacsim/_build/windows-x86_64/release/isaac-sim.bat` (5.1.0-rc.19)
  - `/mnt/c/Users/tmaru/AppData/Local/ov/pkg/isaac-sim-4.5.0/isaac-sim.bat`
  - `/mnt/c/Users/tmaru/AppData/Local/ov/pkg/isaac-sim-4.1.0/isaac-sim.bat`
- If these paths change, detect the current `isaac-sim.bat` under `C:\\Users\\tmaru\\AppData\\Local\\ov\\pkg`.
- From SSH/WSL sessions, avoid direct `./isaac-sim.bat`; use `/home/sskk/launch_isaac_windows.sh` to start with an explicit Windows launch flow.

## Repository-Specific Rules
- Canonical repository root:
  - `/home/sskk/win/isaacsim/source/extensions/my.research.asset_placer_isaac`
- Extension source lives under:
  - `my/research/asset_placer_isaac/`
- Experiment/eval loop lives under:
  - `experiments/`
- Runtime-loaded extension path on Windows is:
  - `C:\\Users\\tmaru\\isaacsim\\_build\\windows-x86_64\\release\\exts\\my.research.asset_placer_isaac`
- Keep source/runtime path linkage (junction/symlink) consistent when validating in Isaac Sim.
- Use `uv` as the default Python dependency/runtime manager for this repository.
- Keep `pyproject.toml` and `uv.lock` as the source of truth for Python dependencies.
- Never use `pip` or `uv pip` in this repository.
- Allowed dependency/runtime commands are:
  - `uv add ...` (add dependencies)
  - `uv sync ...` (sync environment)
  - `uv run ...` (execute scripts/tools)

## Git Workflow
- Prefer small, reviewable commits.
- Commit only files related to the requested task.
- Do not commit generated artifacts unless explicitly requested:
  - `experiments/runs/`
  - `experiments/results/`
  - `experiments/cache/`
- Before push, run at least a smoke validation for changed scripts.

## Implementation Policy
- Keep backward compatibility unless explicitly breaking by request.
- For layout semantics:
  - If `size_mode` is missing, keep `world` behavior.
  - `local` mode must keep `Length/Width/Height` as object-local axes.
- Prefer minimal diffs over broad refactors.
- Avoid accidental line-ending churn and avoid permission-only diffs.

## Validation Policy
- Validate extension behavior with Windows Isaac Sim logs/runtime output.
- If a workflow attempts WSL-native Isaac Sim execution, switch back to Windows runtime flow.
- Before running Python scripts, sync dependencies with:
  - `uv sync --extra experiments`
- For Python changes in `experiments/src`, run:
  - `uv run python -m py_compile experiments/src/*.py` (or equivalent per-file compile)
- For eval loop changes, run at least one smoke trial:
  - `uv run python experiments/src/run_trial.py ...`
- When task-point / observability logic changes, verify debug outputs under `--debug_dir`.

## Editing Policy
- Keep line endings as LF where possible.
- Avoid permission-only edits (`chmod`-only diffs) on `/mnt/c` files.
- Do not move project sources into Linux-only paths unless explicitly requested.

## Safety
- Do not use destructive git commands (`reset --hard`, forced checkout) unless explicitly requested.
- Do not revert unrelated user changes.
