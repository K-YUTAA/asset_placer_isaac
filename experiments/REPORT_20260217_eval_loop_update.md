# 実装・実験報告書（2026-02-17）

## 1. 実施概要
- 対象リポジトリ: `my.research.asset_placer_isaac`
- 実施ブランチ: `exp/eval-loop-v1`
- 反映コミット: `a5d32ad`
- 主目的:
  - eval-loop可視化の強化
  - task points (`s0/s/t/c/g_bed`) と経路可視化の整合
  - `C_vis_start` の常時計算化
  - 経路導出の改善（8近傍A* + 平滑化）
  - room境界（内壁）を考慮した経路制約

## 2. 主要変更点

### 2.1 `experiments/src/task_points.py`
- `g_bed` 候補点生成を修正。
- 仕様に合わせて「ベッド長辺側の2辺」から候補を生成し、`c` に近い側を採用。

### 2.2 `experiments/src/plot_layout_json.py`
- `start-goal` を単純直線で結ぶ描画を廃止。
- `path_cells.json` の実経路を優先描画（`--path_json`対応、`task_points.json`隣接の`path_cells.json`自動読込）。
- 指標オーバーレイに `OOE_*` を追加。

### 2.3 `experiments/src/eval_metrics.py`
- `C_vis_start` を `entry_observability.enabled` に依存せず常時計算。
- `metrics.json` に `OOE_enabled`, `OOE_tau_p`, `OOE_tau_v` を追加。
- デバッグ出力を拡張:
  - `path_cells.json`（raw/smoothed両方）
  - `c_vis_free.pgm`, `c_vis_start_free.pgm`
  - `c_vis_area.png`, `c_vis_start_area.png`
  - `entry_observability.json`, `ooe_per_object.json`
- 経路探索を 4近傍A* から 8近傍A*（corner-cut防止）へ変更。
- LOSベースの経路平滑化を追加（過度なショートカット防止あり）。
- `rooms` / `openings` を使って内壁を occupancy に反映。
  - 共有境界を壁として占有化。
  - 開口は `type="opening"` のみ通行可扱い。
  - `door`/`window` は通行不可（壁を切り欠かない）。

### 2.4 `experiments/src/layout_tools.py`
- 正規化レイアウトに `rooms` 情報を保持する処理を追加。
- これにより eval 側で room境界を利用可能にした。

## 3. 問題対応（問い合わせベース）
- 問題: `start-goal` 線が壁を貫通して見える。
  - 対応: 直線描画を廃止し、A*実経路を描画。
- 問題: `g_bed` が短辺側に出る。
  - 対応: 候補生成を長辺側ルールへ修正。
- 問題: `C_vis_start=0` になる。
  - 対応: entry観測有効/無効に関わらず計算するよう変更。
- 問題: `C_vis/C_vis_start` 可視領域の図が不足。
  - 対応: 専用マスクPGM + PNG可視化を追加。
- 問題: 内壁・トイレ扉の通過。
  - 対応: room共有境界を壁化、`door`開口を通行不可化。

## 4. 実験実行内容

### 4.1 使用レイアウト
- `C:/Users/tmaru/isaacsim/json/kugayama_A_18_5.93*3.04_v2_layout_gpt-5.2_202602171530.json`

### 4.2 実行ケース
- Case A: 既定設定（OOE無効）
  - 出力: `experiments/runs/kugayama_A18_20260217_retest/default/`
- Case B: OOE有効
  - 出力: `experiments/runs/kugayama_A18_20260217_retest/entry_on/`
- Case C: 8近傍A* + 平滑化適用後
  - 出力: `experiments/runs/kugayama_A18_20260217_retest_smooth8/default/`
- Case D: 内壁・扉制約適用後
  - 出力: `experiments/runs/kugayama_A18_20260217_retest_walls/default/`

## 5. 定量結果

### 5.1 Case A（retest/default）
- `C_vis`: `0.9144736842`
- `C_vis_start`: `0.6810954064`
- `R_reach`: `0.8503289474`
- `clr_min`: `0.0414213562`
- `OOE_enabled`: `0`

### 5.2 Case B（retest/entry_on）
- `C_vis`: `0.9144736842`
- `C_vis_start`: `0.6810954064`
- `R_reach`: `0.8503289474`
- `OOE_C_obj_entry_hit`: `0.1274094336`
- `OOE_R_rec_entry_hit`: `0.8333333333`
- `OOE_C_obj_entry_surf`: `0.3148618066`
- `OOE_R_rec_entry_surf`: `0.5714285714`

### 5.3 Case C（smooth8/default）
- `C_vis`: `0.9177631579`
- `C_vis_start`: `0.6810954064`
- `R_reach`: `0.8503289474`
- 経路長（cell）: raw `40` → smoothed `6`

### 5.4 Case D（walls/default）
- `C_vis`: `0.9024390244`
- `C_vis_start`: `0.6392405063`
- `R_reach`: `0.8255159475`
- 経路長（cell）: raw `40` → smoothed `6`

## 6. 壁・扉貫通の検証
- 強制テスト（トイレ側→居室側を扉横断しないと到達不可な条件）を実施。
- 設定例: `start=[2.85, 3.70], goal=[1.2, 3.70]`
- 結果: `path len = 0`（経路不成立）
- 確認ファイル:
  - `experiments/runs/_tmp_wall_check_metrics.json`
  - `experiments/runs/_tmp_wall_check_debug/path_cells.json`

## 7. 主要生成物
- 経路・task points重ね描画:
  - `experiments/runs/kugayama_A18_20260217_retest_walls/default/plot_with_bg.png`
- 可視領域:
  - `experiments/runs/kugayama_A18_20260217_retest_walls/default/debug/c_vis_area.png`
  - `experiments/runs/kugayama_A18_20260217_retest_walls/default/debug/c_vis_start_area.png`
- 経路詳細:
  - `experiments/runs/kugayama_A18_20260217_retest_walls/default/debug/path_cells.json`
- OOE詳細（entry_on時）:
  - `experiments/runs/kugayama_A18_20260217_retest/entry_on/debug/ooe_per_object.json`

## 8. 現時点の注意点
- 今回のコミットは広範囲（77 files）で、docs・inputs・core・experimentsを含む。
- `door` を通行不可扱いにしたため、将来的に「通行可能な扉」を扱う場合は設定化が必要。
- 内壁占有は room境界ベースのため、JSON品質（rooms/openings定義）に依存する。

## 9. 次アクション提案
- `door` 通行可/不可をタイプ別・ID別に切替可能にする。
- 壁/開口占有のデバッグ可視化（`internal_walls.pgm` 等）を追加。
- `run_trial.py` から可視化画像を自動生成する導線を標準化。
