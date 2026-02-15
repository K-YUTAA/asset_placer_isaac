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

## 実験で何をするか（概要）

この `experiments/` は、家具レイアウト（2D平面 + 家具のOBB）に対して
「評価（metrics）」と「改善（heuristic refinement）」を**再現可能に回す**ための実験レイヤーです。

- 入力: レイアウトJSON（extension形式 `area_objects_list` または 正規化形式 `room` + `objects`）
- 出力: 評価指標（`metrics.json` / `metrics.csv`）とデバッグ可視化（PGM + `task_points.json`）
- 比較: `original`（元）/ `heuristic`（局所探索改善）/ `proposed`（将来拡張枠、現状はheuristic相当）

注意: `run_v0_freeze.py` は現状「LLM呼び出し」は行いません（`--layout_input` の既存JSONを評価に流す用途が中心です）。

## 実験の流れ（Eval Loop v1）

1. `run_v0_freeze.py`: レイアウトを読み込み、評価に使える「正規化レイアウト契約（layout contract）」へ変換して保存します。
2. `eval_metrics.py`: occupancyグリッドを作り、到達可能性・通路の狭さ・可視化範囲・レイアウト変更量を評価します。
3. `refine_heuristic.py`（任意）: `eval_metrics.py` を繰り返し呼び出し、家具の微小移動/回転でスコア改善を狙います。
4. `run_trial.py`: 上記を1つの trial として束ね、`trial_manifest.json` と `metrics.csv` に結果を集約します。

## 各スクリプトの役割（I/O）

### `layout_tools.py`

- 役割: 入力JSONの揺れを吸収し、評価/改善が扱える共通表現に正規化します。
- 出力形式（layout contract）:
  - `room.boundary_poly_xy`: 部屋境界ポリゴン（m）
  - `objects[*].size_lwh_m = [L, W, H]`（m）
  - `objects[*].pose_xyz_yaw = [x, y, z, yaw]`（yawはrad、ただし入力がdegっぽい場合は自動変換）

### `run_v0_freeze.py`（baseline生成）

- 入力:
  - `--layout_input`: 既存のレイアウトJSON（extension形式でも可）
  - `--llm_cache_mode`: `write/read`（`llm_response_raw.json` を使った再現実行用）
- 出力（例: `experiments/runs/demo_v0/`）:
  - `layout_llm.json`: 入力（またはモック生成）をそのまま保存
  - `layout_v0.json`: 正規化した layout contract
  - `llm_response_parsed.json`, `llm_response_raw.json`
  - `asset_manifest.json`, `collision_report.json`, `run_manifest.json`

### `eval_metrics.py`（評価器）

- 入力: `--layout`（layout contractに正規化してから評価）、`--config`（評価パラメータ）
- 出力:
  - `--out`: `metrics.json`
  - `--debug_dir` 指定時:
    - `occupancy.pgm`: 家具占有（0=占有, 255=free）
    - `reachability.pgm`: 到達可能性 + 経路の簡易可視化
    - `task_points.json`: start/goal の解決結果（`eval.task` 有効時）

### `task_points.py`（タスク点の解決）

- 役割: start/goal を「入口ドア」「ベッド脇」から自動決定し、freeセルへスナップします。
- 有効化: eval config の `task` セクションが存在する場合（未指定なら従来の `start_xy/goal_xy`）。

### `refine_heuristic.py`（改善ループ / 局所探索）

- 役割: レイアウトを少しずつ動かし、評価スコア `_score()` を改善する候補を探索します。
- 特徴:
  - まずボトルネック（最小クリアランスセル）があれば、その近傍の家具を優先して動かします。
  - それが無い場合は start-goal 直線に近い家具を優先して動かします。

### `compare_layout.py`（再現性チェック）

- 役割: 2つのレイアウトが「同じもの」と見なせるかを段階的にチェックします。
  - Level1: `id`/個数/`asset_id` の一致
  - Level2: 位置差 `delta_position_m < pos_tol_m` と yaw差 `delta_yaw < yaw_tol`
  - Occupancy IoU: グリッド占有の IoU が閾値以上

### `run_trial.py`（trial runner）

- 役割: `v0 -> eval -> (optional refine) -> eval` を1試行として回し、結果を保存します。
- 出力（例: `experiments/results/<trial_id>_<timestamp>/`）:
  - `metrics.json`（最終レイアウトの評価）
  - `layout_refined.json` / `metrics_refined.json`（methodが `heuristic/proposed` のとき）
  - `trial_manifest.json`（再現用の設定とメタデータ）
  - `experiments/results/.../metrics.csv`（集計）

## 評価指標（定義と式）

評価は `eval_metrics.py` の実装に準拠します（離散グリッド上の2D評価です）。

### グリッドとマスク

- 解像度: $r$（`grid_resolution_m`）
- 部屋境界ポリゴンからBBoxを取り、セル中心が部屋内のセルを `room_mask` とします。
- 家具OBBに含まれるセル中心を `occ`（occupied）とします。
- ロボット半径 $R$（`robot_radius_m`）をセル半径に変換し、`occ` を膨張して `inflated` を作ります。
- 走行可能セル: `free_mask = room_mask AND (NOT inflated)`

### `validity`

start/goal が `free_mask` 上にあるかで定義します:

$$
validity = \\mathbb{1}[start \\in Free] \\cdot \\mathbb{1}[goal \\in Free]
$$

`eval.task` が有効な場合、start/goal は必ず近傍の free セルへスナップされます（`task_points.json` 参照）。

### `R_reach`（到達可能率）

start から BFS で到達できる free セル集合を $Reachable$ とすると:

$$
R_{reach} = \\frac{|Reachable|}{|Free|}
$$

### `clr_min`（経路最小クリアランス）

free_mask 上で A* により start->goal の経路セル列 $P$ を得ます。
`occ` からの距離変換（近傍の占有セルまでの距離）を $d(c)$ とし、経路上のクリアランスを
$d(c) - R$ として評価します:

$$
clr_{min} = \\min_{c \\in P} (d(c) - R)
$$

実装上は $\\max(0, clr_{min})$ にクランプします。

### `C_vis`（可視化面積率）

センサ位置セル集合 $S$（start + 経路上のサンプル点）から、
freeセル $c$ への line-of-sight（Bresenham）が `occ`/部屋外に当たらないとき $c$ を可視とします。
可視freeセル数を $|Visible|$ とすると:

$$
C_{vis} = \\frac{|Visible|}{|Free|}
$$

### `Delta_layout`（レイアウト変更量）

baseline と現在の movable 物体集合を $\\mathcal{M}$ とし、物体 $i$ の平面移動量 $\\Delta p_i$、
yaw差 $\\Delta \\theta_i$、部屋BBox対角長 $D$、重み $w_i$（面積比）を用いて:

$$
\\Delta_{layout} = \\sum_{i \\in \\mathcal{M}} w_i\\left(\\frac{\\Delta p_i}{D} + \\lambda_{rot}\\frac{|\\Delta\\theta_i|}{\\pi}\\right)
$$

ここで $w_i = \\frac{A_i}{\\sum_j A_j}$, $A_i = L_i W_i$（平面占有面積）です。

### `Adopt`（しきい値による採否）

$$
Adopt = \\mathbb{1}[R_{reach} \\ge \\tau_R \\land clr_{min} \\ge \\tau_{clr} \\land C_{vis} \\ge \\tau_V \\land \\Delta_{layout} \\le \\tau_{\\Delta}]
$$

## Notes

- This layer supports both extension-style layout JSON (`area_objects_list`) and
  normalized experiment JSON (`room` + `objects`).
- `eval_metrics.py` supports `eval.task` to auto-resolve start/goal (snapped to free cells).
  - `--debug_dir` also writes `task_points.json` alongside the PGM maps.
- `run_trial.py` stores the resolved points in `trial_manifest.json` (`debug_meta.task_points`).
- `run_v0_freeze.py` supports `llm_cache_mode=read/write` for deterministic re-runs.
- Scripts use only Python standard library for portability.
