# 可視化実装計画（今後の実装）

日付: 2026-02-17  
対象: `experiments/` 評価ループ（`run_trial.py`, `eval_metrics.py`, `plot_layout_json.py`）

## 1. 目的
- 試行ごとの結果を画像だけで素早く確認できるようにする。
- `original / heuristic / proposed` の比較を同じ形式で出力する。
- 手作業デバッグを減らすため、試行後に可視化を自動生成する。

## 2. この計画でやらないこと
- Isaac Sim 本体レンダリングの変更。
- 拡張UIパネルの変更。
- 指標定義（V/L/Mなど）の変更。

## 3. 成果物
- 試行ごとのレイアウト画像（自動生成）。
- 3手法比較パネル画像（横並び）。
- オプション重ね描き:
- Start/Goal
- 経路
- 衝突・閉塞セル
- 可視化率ヒートマップ
- 指標サマリ（V, L, M, overlap, path系）を注記した画像。

## 4. 段階実装

### Step A: `run_trial.py` から自動プロット呼び出し
- 各手法実行後に `plot_layout_json.py` を自動実行。
- 出力先を試行ディレクトリ配下に統一:
- `plots/layout_<method>.png`
- 設定項目を追加:
- `viz.enable`
- `viz.with_bg_image`
- `viz.bg_image_path`
- `viz.bg_crop_mode`

受け入れ条件:
- `viz.enable=true` で、1試行ごとに手法別画像が必ず出る。

### Step B: 3手法比較パネル
- 新規スクリプト `experiments/src/plot_trial_panel.py` を追加。
- 入力:
- `original / heuristic / proposed` の3JSON
- 必要なら共通背景画像
- 出力:
- `plots/panel_methods.png`（1x3）

受け入れ条件:
- 手作業合成なしで比較画像1枚が作れる。

### Step C: タスク点オーバーレイ（Start/Goal/Path）
- manifest/debug出力から task points を読み込み。
- Start/Goal と経路を重ね描き。
- 色を固定:
- start: 緑
- goal: 赤
- path: 青

受け入れ条件:
- 生JSONを開かなくても経路妥当性を判断できる。

### Step D: 衝突・到達可能領域オーバーレイ
- 評価内部の occupancy/free mask を再利用。
- 表示切替を追加:
- blocked cells
- reachable region
- collision cells
- 透過オーバーレイ画像と合成画像を出力。

受け入れ条件:
- 衝突原因が画像上で直接追える。

### Step E: 可視化率ヒートマップ
- 取得可能な可視化率計算中間データを利用。
- 部屋領域にアルファ合成ヒートマップを描画。
- 出力:
- `plots/visibility_<method>.png`
- `plots/visibility_panel.png`

受け入れ条件:
- 観測性の差を手法間で視覚比較できる。

### Step F: 指標注記（画像内テキスト）
- `metrics.json` を読み込んでパネル画像右側に注記。
- 表示項目:
- V, L, M, overlap count
- path length, reachable ratio
- 数値フォーマットを固定して再現性を担保。

受け入れ条件:
- 1コマンドでレポート向け図を出せる。

## 5. 実装順（推奨）
1. Step A
2. Step B
3. Step C
4. Step D
5. Step F
6. Step E

理由:
- まず自動化と比較図を固め、その後に重ね描きを増やす。
- ヒートマップは可読性調整が必要になりやすいため最後にする。

## 6. 変更予定ファイル
- `experiments/src/run_trial.py`: 可視化フック追加。
- `experiments/src/plot_layout_json.py`: ベース描画（必要なら関数分離）。
- `experiments/src/plot_trial_panel.py`: 新規追加。
- `experiments/src/eval_metrics.py`: 必要最小限の中間データ公開。
- `experiments/README.md`: 実行方法と出力物を追記。

## 7. テスト計画
- スモーク:
- `uv run python experiments/src/run_trial.py ...`
- 手法別プロットが出ることを確認。
- 回帰:
- 可視化ON/OFFで指標値が変わらないことを確認。
- 目視:
- 座標と部屋境界が一致する。
- Start/Goal が想定セルにある。
- 経路が free-space と整合する。

## 8. リスクと対策
- リスク: 背景画像との位置合わせずれ。
- 対策: `bg_extent` 手動指定とアンカー補正を併用。
- リスク: 重ね描きが多すぎて見づらい。
- 対策: レイヤ切替と用途別画像出力を用意。
- リスク: 実行時間増加。
- 対策: バッチ時は `viz.enable=false` を既定にする。

## 9. 完了条件
- 1試行で以下が生成される:
- 手法別レイアウト画像
- 比較パネル画像
- 必要なオーバーレイ画像
- 指標値は従来と一致
- README に実行コマンドと出力先が明記されている。
