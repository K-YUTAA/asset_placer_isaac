# Codex実装指示書: Eval Loop v1 の Task Points（入口スタート + ベッド脇ゴール）

対象: `K-YUTAA/asset_placer_isaac`（一次情報）

関連ノート:
- `[[2026-02-16_bedside_goal_and_start_spec]]`
- `[[2026-02-14-eval-metrics-evaluate_layout-report]]`

## 目的

- start を「入口の Sliding Door 中心」基準で自動決定する。
- goal を「ベッド脇（移乗・介助位置）」基準で自動決定する。
- 壁厚/境界セルで `validity=0` になる事故を減らすため、start/goal を必ず free セルへスナップする。
- 5部屋を横並び比較してもタスク意味がブレないよう、決定規則をコードに固定する。

## 前提（現状コードの一次参照）

- `experiments/src/eval_metrics.py` が評価器の本体。
- `experiments/src/run_trial.py` が trial runner。
- `experiments/src/refine_heuristic.py` が改善ループ。
- レイアウト契約は `experiments/src/layout_tools.py:normalize_layout()` が生成する。
- Python は標準ライブラリのみで回っている（依存追加しない）。

## 作業ブランチ

- 推奨: `exp/eval-loop-v1` から新規に `exp/eval-loop-v1-taskpoints` を切って作業。

## 実装スコープ

- 変更対象は原則 `experiments/` 配下のみ。
- 既存の `start_xy` / `goal_xy` 指定は後方互換として残す。

## 実装TODO（Codexにやらせる内容）

1. 新規モジュールを追加する
ファイル: `experiments/src/task_points.py`

- 役割: layout + config から start/goal を一意に解決し、free セルへスナップする。
- 依存: `layout_tools` と `eval_metrics` 内部のグリッド変換関数を使いたい場合は、必要な最小関数を移すか、同等処理を `task_points.py` 側に置く。

2. 評価器に統合する
ファイル: `experiments/src/eval_metrics.py`

- `evaluate_layout()` の冒頭で `task_points.resolve_task_points(...)` を呼ぶ。
- `start_xy`/`goal_xy` を上書きして以降の処理（BFS/A*/LoS/距離変換）が同じ start/goal を使うようにする。
- `debug` に以下を追加する。

`debug.task_points` の例:
```json
{
  "start": {"mode": "entrance_slidingdoor_center", "xy": [1.23, 0.56], "cell": [12, 5]},
  "goal": {"mode": "bedside", "xy": [4.56, 3.21], "cell": [45, 32]},
  "selectors": {"door_id": "door_00", "bed_id": "bed_00"},
  "snap": {"max_radius_cells": 30, "moved_start": true, "moved_goal": true}
}
```

- `--debug_dir` 指定時に `task_points.json` を保存する（PGMと一緒に原因切り分けしやすくする）。

3. ヒューリスティック改善側も同じ start/goal を参照させる
ファイル: `experiments/src/refine_heuristic.py`

- `_select_target_object()` が `config.start_xy/goal_xy` を見る箇所を、解決済み start/goal を見るようにする。
- 方法A: `evaluate_layout()` の `debug.task_points` を参照し、XY を取り出して直線距離計算に使う。
- 方法B: `task_points.resolve_task_points()` を直接呼び、start/goal を得る。

4. trial runner 側で manifest に「解決済み start/goal」を残す
ファイル: `experiments/src/run_trial.py`

- `layout_v0` をロードした後で解決済み start/goal を計算し、`trial_manifest.json` に保存する。
- 可能なら `resolved_eval_config` にも `start_xy`/`goal_xy` を反映し、再現時に同じ点が使われるようにする。

5. 設定JSONの例を更新する
ファイル: `experiments/configs/eval/default_eval.json`
ファイル: `experiments/configs/trials/sample_exp_a.json`
ファイル: `experiments/README.md`

- `task` セクションを追加し、最小実験がそのまま実行できる例にする。

## 設定仕様（最小で運用できる形）

- 既存: `start_xy`, `goal_xy` は残す。
- 新規: `task` があれば優先して自動決定する。

`eval` もしくは `eval_config` 内の例:
```json
{
  "grid_resolution_m": 0.10,
  "robot_radius_m": 0.30,
  "task": {
    "start": {
      "mode": "entrance_slidingdoor_center",
      "in_offset_m": 0.40,
      "door_selector": {"strategy": "largest_opening"}
    },
    "goal": {
      "mode": "bedside",
      "offset_m": 0.60,
      "bed_selector": {"strategy": "first"},
      "choose": "closest_to_room_centroid"
    },
    "snap": {"max_radius_cells": 30}
  }
}
```

## DoD（完了条件）

- 旧 config（`start_xy/goal_xy`）で既存スクリプトが壊れない。
- `task.start/goal` を有効化すると、境界セル事故で `validity=0` になる頻度が下がる。
- `--debug_dir` で `task_points.json` が出て、start/goal がどの物体から決まったか追える。
- `refine_heuristic.py` が解決済み start/goal を使い、改善対象選択が評価と矛盾しない。
