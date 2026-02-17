# Experiments Changelog

`experiments/` 配下の評価・実験スクリプト専用の変更履歴です。  
（Extension 本体の変更履歴は `docs/CHANGELOG.md` を参照）

## [Unreleased]

### Changed
- `eval_metrics.py` の占有グリッド計算で、デフォルトで `floor` カテゴリを障害物から除外するように変更。
- 新規設定キー `occupancy_exclude_categories` を追加（未指定時は `["floor"]`）。
- 旧データ互換として、カテゴリ欠落時でも `id` が `floor` 系なら除外されるフォールバックを追加。
- `task_points.py` のデバッグ情報に `anchors` を追加し、`s0/s/t/c/g_bed` を出力。
- `plot_layout_json.py` に `--metrics_json` / `--task_points_json` を追加。
  - 指標値のオーバーレイ表示
  - タスク点 `s0,s,t,c,g_bed` と start/goal の可視化

### Impact
- 床オブジェクトが評価セルを全面占有して `validity=0` になる問題を回避。
- 既存設定を変更しなくても、通常ケースで `R_reach` / `C_vis` / `clr_min` が正しく算出される。
- 評価結果とタスク点を1枚のレイアウト画像で確認できる。
