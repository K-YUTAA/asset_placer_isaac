# 実験計画（Step by Step）

## ゴール
- 既存の Isaac Sim 拡張（asset_placer_isaac）の本体を大きく壊さずに、再現可能な評価実験を回す。
- Baseline（v0生成）と、評価・修正（heuristic/最適化）を分離して検証する。
- 論文用には、最終的に修正後JSONを Isaac + USD Search で可視化して提示する。

## 基本方針
- 拡張本体（`my/research/asset_placer_isaac/`）は最小変更。
- 実験ロジックは実験層（Python）で外出し実装。
- 作業は `exp/*` ブランチで進める。
- 再現性は「LLM完全一致」ではなく「評価入力として同等」で判定。

---

## Step 0: ブランチと実験ディレクトリを準備

### 目的
- main/master を汚さず、安全に実験を進める。

### 作業
- `master` から `exp/eval-loop-v1` を作成。
- 実験用ディレクトリを作成（例：`experiments/`）。

### 推奨構成
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

### DoD
- `exp/eval-loop-v1` で作業開始できる。
- `experiments/` の骨組みがある。

---

## Step 1: v0凍結（Baseline入力を作る）

### 目的
- 比較の基準となる `layout_v0.json` を安定生成する。

### 作業
- `run_v0_freeze.py` を作成（実験層）。
- 入力：画像、寸法、プロンプト（任意）、モデル設定。
- 出力：`layout_v0.json`, `run_manifest.json`, `collision_report.json` など。

### 最低限の固定出力
- `layout_v0.json`
- `llm_response_raw.json`
- `llm_response_parsed.json`
- `collision_report.json`
- `run_manifest.json`

### run_manifest に必須
- model / prompt / reasoning / tokens
- seed / temperature / top_p（使えなくても記録）
- commit hash
- 評価パラメータ（grid_resolution, robot_radius, start/goal）
- 配置補正パラメータ（反復回数、押し出しステップ、順序）

### DoD
- 同じ入力で2回実行し、比較可能な `layout_v0` が出る。

---

## Step 2: 再現性チェックを自動化（compare_layout.py）

### 目的
- 「この入力で実験してよいか」を機械判定する。

### 作業
- `compare_layout.py` を作成。
- 入力：`layout_v0_A.json`, `layout_v0_B.json`。
- 2段階判定を実装。

### 判定ルール
#### Level 1（ハード一致 / 必須）
- object id集合一致
- object数一致
- asset_id一致（可能なら scale / variant も）

#### Level 2（ソフト一致 / 許容）
- 位置差 `||Δt|| < 0.02m`
- 角度差 `|wrap(Δθ)| < 1deg`
- 推奨追加：2D占有 IoU `> 0.98`

### DoD
- PASS/FAIL と理由が出る。
- FAIL時に `run_manifest` 参照で原因追跡できる。

---

## Step 3: 基本評価器（Evaluator v1）をPythonで実装

### 目的
- JSONベースで主指標を安定計算する。

### 作業
- `eval_metrics.py` を作成。
- 入力：`layout_v0.json`（または refined JSON）。
- 出力：`metrics.json` と debug画像（occupancy/reachability）。

### まず実装する指標
- `R_reach`（到達率）
- `C_vis`（可視性）
- `Delta_layout`（変更量）
- `Adopt`（閾値判定）

### DoD
- 単一ケースで数値とデバッグ画像が出る。

---

## Step 4: heuristic修正器（Refiner v1）を実装

### 目的
- 評価値を改善する最小修正ループを作る。

### 作業
- `refine_heuristic.py` を作成。
- ループ：`評価 -> 1手修正 -> 再評価`。
- 出力：`layout_refined.json`, `refine_log.json`。

### 修正対象（初期）
- 位置微調整（x/y）
- 角度微調整（rotationZ）
- 通路確保優先の単純ルール

### DoD
- 少なくとも1ケースで `R_reach` か `C_vis` が改善。

---

## Step 5: trial定義とRunnerを作る

### 目的
- 実験条件をJSON化し、機械的に実行する。

### 作業
- `configs/trials/*.json` を作成。
- `run_trial.py` で trial 読み込み -> v0 -> evaluator -> refiner を実行。
- `results/metrics.csv` へ1行追記。

### trial最小項目
- `layout_id`, `method`, `seed`
- `sensor`（例: lidar/window）
- `eval`（grid_resolution, robot_radius, start/goal）

### DoD
- trial1件をCLIで回し、`metrics.csv` に結果が残る。

---

## Step 6: 実験行列を回す（Exp-A優先）

### 目的
- 比較実験を必要数集める。

### 作業
- まず Exp-A（layout × method × seed）を実行。
- 失敗trialは `status/error` を残して再実行可能にする。

### DoD
- 目標試行数の結果が `metrics.csv` に揃う。

---

## Step 7: 論文用の可視化データを作る

### 目的
- 主張に必要な図表を先に固定する。

### 作業
- `results/figures/` に散布図・箱ひげ・ヒートマップを生成。
- Python評価結果と図を対応付ける。

### DoD
- 主要図（改善前後比較）が再生成できる。

---

## Step 8: 最終見せ方（Isaac + USD Search）

### 目的
- 修正後JSONを見栄えの良いシーンとして提示する。

### 作業
- `layout_refined.json` を Isaac Sim 側で配置。
- USD Searchで最終アセット反映。
- 論文掲載用のレンダ画像を保存。

### DoD
- 指標改善と見た目改善が同時に示せる。

---

## Step 9: 報告テンプレートに落とす

### 目的
- 進捗を継続的に共有できる形にする。

### 作業
- 各実験ごとに
  - 何を変えたか
  - 指標がどう変わったか
  - 次に何をするか
  を定型で記録。

### DoD
- 1実験1メモで追跡可能。

---

## 今の着手優先順位（実行順）
1. Step 0（ブランチ + 実験骨組み）
2. Step 1（run_v0_freeze）
3. Step 2（compare_layout 2段階）
4. Step 3（Evaluator v1）
5. Step 4（Refiner v1）

この順番なら、最短で「再現可能な改善ループ」まで到達できる。

---

## 追記（2026-02-13）: 数式ベース実装仕様の決め打ち

方針・分割（v0固定 / JSONで評価と修正 / Isaacは最終可視化）とStep順は維持したまま、
実装迷いをなくすための仕様を固定する。

## 追記0: データ契約（先に固定）

Step1-3の詰まりを防ぐため、実験層での正スキーマを先に決める。

### 0.1 `layout_v0.json`（最小必須）

```jsonc
{
  "meta": {
    "layout_id": "layout01",
    "source": "v0",
    "unit": "m",
    "timestamp": "2026-02-10T00:00:00Z"
  },
  "room": {
    "boundary_poly_xy": [[0, 0], [W, 0], [W, H], [0, H]],
    "ceiling_height_m": 2.4
  },
  "objects": [
    {
      "id": "bed_00",
      "category": "bed",
      "asset_query": "hospital bed",
      "asset_id": "omni:xxxx",
      "scale": [1.0, 1.0, 1.0],
      "size_lwh_m": [2.0, 1.0, 0.6],
      "pose_xyz_yaw": [1.2, 2.8, 0.0, 1.57],
      "movable": false
    }
  ]
}
```

### 0.1.1 安定IDの正規化ルール（必須）

- LLMのID揺れ対策として、実験層で `id` を再採番する。
- ルール: `category` ごとに `(x, y)` 辞書順でソートし、`category_00, category_01...` を付与。
- 目的: `compare_layout` が無駄にFAILしないようにする。

### 0.2 `metrics.json`（最小必須）

```jsonc
{
  "trial_id": "expA_layout01_original_seed1",
  "C_vis": 0.42,
  "R_reach": 0.91,
  "clr_min": 0.62,
  "Delta_layout": 0.08,
  "Adopt": 1,
  "validity": 1,
  "runtime_sec": 3.2
}
```

## Step 1 追記: `run_v0_freeze.py` の固定仕様

### 1.1 I/O固定

- 入力: `--sketch_path --hints_path --seed --out_dir --llm_cache_mode {write|read}`
- 出力（必須）:
- `layout_llm.json`（LLM parse直後）
- `layout_v0.json`（補正 + asset選択後、評価の正）
- `llm_response_raw.json`
- `llm_response_parsed.json`
- `asset_manifest.json`
- `run_manifest.json`
- `collision_report.json`

### 1.2 LLM cache（再現性の要）

- `llm_cache_mode=write`: 初回はLLM呼び出し + raw保存。
- `llm_cache_mode=read`: 2回目以降はraw再利用（LLM未呼び出し）。

### 1.3 `asset_manifest.json` 最小仕様

```jsonc
{
  "bed_00": {
    "asset_query": "hospital bed",
    "chosen": {"asset_id": "omni:xxxx", "score": 0.812},
    "topk": [
      {"asset_id": "omni:xxxx", "score": 0.812},
      {"asset_id": "omni:yyyy", "score": 0.809}
    ]
  }
}
```

## Step 2 追記: `compare_layout.py` の数式確定

### 2.1 Level2差分定義

物体 \(i\) のA/B姿勢を \(\mathbf{t}_i=(x_i,y_i), \theta_i\) とする。

\[
\Delta p_i = \left\lVert \mathbf{t}_i^A - \mathbf{t}_i^B \right\rVert_2
\]

\[
\Delta \theta_i = \left| \mathrm{wrap}\left(\theta_i^A - \theta_i^B\right) \right|,\quad
\mathrm{wrap}(\phi)=((\phi+\pi)\bmod 2\pi)-\pi
\]

- 判定: \(\Delta p_i < 0.02\) [m]
- 判定: \(\Delta \theta_i < \pi/180\) [rad]

### 2.2 占有IoU（推奨）

\[
IoU = \frac{|A \cap B|}{|A \cup B|}
\]

- `grid_resolution_m = 0.1`
- 合格基準: `IoU > 0.98`

## Step 3 追記: Evaluator(v1) 数式・手順

### 3.1 OBB占有グリッド

各家具 \(i\) を床面の回転矩形（OBB）で表現:
- サイズ \((L_i, W_i)\)
- 位置 \(\mathbf{t}_i=(x_i,y_i)\)
- 角度 \(\theta_i\)

\[
\mathbf{p}' = R(-\theta_i)\left(\mathbf{p}-\mathbf{t}_i\right),\quad
R(\alpha)=
\begin{bmatrix}
\cos\alpha & -\sin\alpha\\
\sin\alpha & \cos\alpha
\end{bmatrix}
\]

\[
\mathrm{inside}_i(\mathbf{p})=
\mathbf{1}\left(|p'_x|\le L_i/2 \land |p'_y|\le W_i/2\right)
\]

\[
O(\mathbf{p}) = \mathbf{1}\left(\exists i:\mathrm{inside}_i(\mathbf{p})\right)
\]

部屋外セルは占有扱いとする。

### 3.1.1 inflate（ロボ半径反映）

\[
n=\left\lceil \frac{r}{\delta} \right\rceil
\]

- \(r\): robot radius
- \(\delta\): grid resolution
- \(O\) をモルフォロジー膨張し \(O_r\) を得る。

### 3.2 到達率 \(R_{reach}\)

\[
\mathcal{F}=\{\mathbf{p}\mid O_r(\mathbf{p})=0\}
\]

startセル \(s\) からBFSで到達可能集合 \(\mathcal{R}\) を求める:

\[
R_{reach}=\frac{|\mathcal{R}|}{|\mathcal{F}|}
\]

### 3.3 最小クリアランス \(clr_{min}\)

inflate前の占有 \(O\) から距離変換 \(d(\mathbf{p})\) を計算。
経路 \(P\)（A*）上で:

\[
clr_{min} = \min_{\mathbf{p}\in P} \left(d(\mathbf{p})-r\right)
\]

経路がない場合は `clr_min=0`, `Adopt=0` とする。

### 3.4 可視性 \(C_{vis}\)

自由セル \(\mathcal{F}\)、センサ集合 \(S\)（start + 経路上サンプル）を使う。

\[
\mathrm{LOS}(s,\mathbf{p})=
\mathbf{1}\left(\text{線分 } s\to\mathbf{p} \text{ が占有と交差しない}\right)
\]

\[
\mathrm{VIS}(\mathbf{p})=
\mathbf{1}\left(\exists s\in S:\mathrm{LOS}(s,\mathbf{p})=1\right)
\]

\[
C_{vis}=\frac{1}{|\mathcal{F}|}\sum_{\mathbf{p}\in\mathcal{F}}\mathrm{VIS}(\mathbf{p})
\]

- LOSはBresenhamまたはray marchingで判定。
- 安定化の推奨: 経路上0.5m間隔、最大10点を \(S\) に追加。

### 3.5 変更量 \(\Delta_{layout}\)

初期レイアウト（上付き0）との比較:

\[
\Delta p_i=\left\lVert \mathbf{t}_i-\mathbf{t}_i^0 \right\rVert_2,\quad
\Delta \theta_i=\left|\mathrm{wrap}\left(\theta_i-\theta_i^0\right)\right|
\]

\[
\Delta_{layout}=
\sum_{i\in\mathcal{M}} w_i\left(
\frac{\Delta p_i}{D}+\lambda\frac{\Delta\theta_i}{\pi}
\right)
\]

- \(\mathcal{M}\): movable家具
- \(w_i\): 例として床面積重み
- \(D\): 部屋対角長

### 3.6 導入判定 \(Adopt\)

\[
Adopt=
\mathbf{1}\left(
R_{reach}\ge\tau_R \land
clr_{min}\ge\tau_{clr} \land
C_{vis}\ge\tau_V \land
\Delta_{layout}\le\tau_{\Delta}
\right)
\]

初期閾値例:
- \(\tau_R=0.9\)
- \(\tau_{clr}=0.2\) m
- \(\tau_V=0.4\)
- \(\tau_{\Delta}=0.15\)

## Step 4 追記: Refiner v1 決め打ち

### 4.1 ループ仕様

- 最大反復: `T=30`
- 候補:
- 並進 \(\Delta x,\Delta y\in\{-\delta,0,+\delta\}\), \(\delta=0.1\)m
- 回転 \(\Delta\theta\in\{-15^\circ,0,+15^\circ\}\)
- 1手で1家具、総変更家具数 `K<=3`

### 4.2 不可侵制約

- 部屋境界外に出ない
- 重大衝突を増やさない
- `R_reach` または `clr_min` を悪化させない

### 4.3 対象家具選択

- 経路あり: 最狭点
\[
\mathbf{p}^*=\arg\min_{\mathbf{p}\in P}\left(d(\mathbf{p})-r\right)
\]
- 最狭点に最も近いmovable家具を選択
\[
i^*=\arg\min_i\left\lVert \mathbf{t}_i-\mathbf{p}^* \right\rVert
\]
- 経路なし: start-goal直線に最も近い家具

### 4.4 採用スコア

\[
J=\alpha C_{vis}+\beta R_{reach}+\eta\,\mathrm{clip}(clr_{min},0,c_{max})
-\gamma\Delta_{layout}-\mathrm{Penalty}
\]

- 初期係数例: \(\alpha=\beta=1,\eta=1,\gamma=0.5\)
- 候補全評価で \(J\) 最大を採用
- `refine_log.jsonl` に before/after と採否理由を残す

## Step 5-6 追記: trial/runner運用固定

### 5.1 method列挙

- `original`
- `prompt_sensor`
- `heuristic`
- `proposed`

### 5.2 `metrics.csv` 必須列

- `status`
- `error_msg`

失敗trialを隠さず保存する。

## 任意追記: Exp-B誤差定義

真値姿勢 \(\mathbf{x}^*=(x^*,y^*,\theta^*)\)、推定 \(\hat{\mathbf{x}}\):

\[
e=\sqrt{(\hat{x}-x^*)^2+(\hat{y}-y^*)^2+\kappa^2\mathrm{wrap}(\hat{\theta}-\theta^*)^2}
\]

\[
E_{xy}=\sqrt{\frac{1}{J}\sum_{j=1}^{J}e_j^2}
\]

## 最重要の追記箇所（実装前チェック）

1. Step1に `layout_llm.json` / `layout_v0.json` の2段保存 + LLM cache read/write を明記
2. Step2に `wrap` 定義と occupancy IoU 数式を明記
3. Step3に OBB / inflate / `C_vis` / `R_reach` / `clr_min` / `Delta_layout` / `Adopt` の数式を明記
