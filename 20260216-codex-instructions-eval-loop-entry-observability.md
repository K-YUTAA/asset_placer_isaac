# Codex指示書: Eval Loop に「入室直後の環境把握」指標（Entry Observability; OOE）を追加する

作成日: 2026-02-16

## 0. ゴール（何を追加するか）
現状の `C_vis` は「経路上の複数点」からの床面 free 可視率で、運用全体の見通しには効く。一方で現場的には「入室直後に停止する点 S から、動き出す前に環境をどれだけ把握できるか」が重要。

そこで評価器に、入室直後の固定観測点 `S` のみを使う可視・観測指標を追加する。

追加する指標は 2 段構え。

- `C_vis_start`: 点 `S` からの床面 free の見通し（既存 `C_vis` の start-only 版）
- `OOE`（Object Observability at Entry）: 点 `S` から「家具がどれだけ把握できるか」を per-object とスカラーで出す

`OOE` は 1 個の方法に固定せず、研究的に強い 2 種類の proxy を並走させる。

- first-hit レイ（LiDAR風）: 「入室直後にまず何が見えるか」を安定に出す
- surface proxy（境界サンプル）: 「家具の何% / 面積が見えているか」に近い量を出す

## 1. 前提（この実装はどこに乗せるか）
- ブランチ: `exp/eval-loop-v1-taskpoints`（Task Points: 入口 start + ベッド脇 goal + snap_to_free が入っている前提）
- 実装場所: `experiments/src/eval_metrics.py` を主戦場にする
- `evaluate_layout()` が唯一の基準点（refine/run_trial からも同じ start/goal を参照）
- 観測点 `S` は `task_points` により確定した start を使う（`S0`（ドア中心）→ `S`（室内側へ押し込み）→ `snap_to_free` を通した最終点）

## 2. 指標定義（数式）

### 2.1 `C_vis_start`（床面の見通し）
- `S` を固定し、`room_mask && !occ`（非 inflate の free）を対象に LoS で可視 free セル集合 `V(S)` を求める

- 定義:

$$
C_{vis\_start} = \frac{|V(S)|}{|F|}, \quad
F = \{x \mid room\_mask(x)=1 \land occ(x)=0\}
$$

### 2.2 `OOE (first-hit)`（入室家具観測）: LiDAR風 first-hit レイ
「画像認識に足りる面積」を厳密にやると重いので、まずは 2D グリッドで再現性が高い first-hit レイをコア指標にする。

- レイ:

$$
r_k(t)=S + t[\cos\alpha_k,\sin\alpha_k]^\top,\quad k=1..K,\;t\ge 0
$$

- 各レイの first-hit 物体 ID を `o_k` とする（家具 / 壁 / none）。

- 家具 `i` の観測割合（角度占有率）:

$$
p_i = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[o_k=i],\quad p_i\in[0,1]
$$

- 集約（スカラー）:

$$
C^{obj,hit}_{entry} = \frac{\sum_{i\in\mathcal{I}} w_i p_i}{\sum_{i\in\mathcal{I}} w_i}
$$

- 認識可能家具率（「何割の家具を把握できたか」）:

$$
R^{rec,hit}_{entry} = \frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\mathbf{1}[p_i\ge\tau_p]
$$

推奨初期値（固定して再現性を確保する）
- `K = 720`（0.5deg）
- `tau_p = 0.02`（720本なら 14本相当の角度幅）

### 2.3 `OOE (surface)`（任意）: 境界サンプルによる「見える割合/面積」proxy
家具の「何%が確認できているか」「表面積が確保できているか」に寄せるため、境界サンプルで可視側面の割合を近似する。

- 家具 `i` の床面 OBB フットプリントをグリッドに落とした占有セル集合を `O_i`、その 4-neighbor 境界セル集合を `∂O_i` とする。

- `S` からの可視 free セル集合 `V(S)`（2.1 と同様）を使い、境界セル `b ∈ ∂O_i` が「見える」とは次で近似する。

$$
visible(b)=\mathbf{1}\left(\exists n\in N_4(b): n\in V(S)\right)
$$

- 家具 `i` の可視率（境界サンプル比）:

$$
v_i = \frac{\sum_{b\in \partial O_i} visible(b)}{|\partial O_i|},\quad v_i\in[0,1]
$$

- 2D から面積っぽい値にするため、家具を「縦プリズム」とみなし側面積を近似する。

$$
A^{side}_i \approx 2(L_i+W_i)H_i,\quad A^{vis}_i \approx A^{side}_i v_i
$$

- 集約（スカラー）:

$$
C^{obj,surf}_{entry} = \frac{\sum_{i\in\mathcal{I}} w_i v_i}{\sum_{i\in\mathcal{I}} w_i}
$$

- 認識可能家具率（割合閾値）:

$$
R^{rec,surf}_{entry} = \frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\mathbf{1}[v_i\ge\tau_v]
$$

推奨初期値（感度分析を前提に固定）
- `tau_v = 0.3`

## 3. 2.5D（高さ）: センサー高さで遮蔽を変える
2D だけだと「低い手前家具が奥を遮ってゼロ」になりやすい。最小拡張としてセンサー高さ `h_s` による occupancy を作る。

- センサー高さ `h_s` における占有:

$$
occ_{sense}(x;h_s)=\mathbf{1}\left(\exists j: x\in B_j \land H_j\ge h_s\right)
$$

- `B_j` は家具 `j` の床面 OBB フットプリント、`H_j` は家具高さ。
- 壁・固定区画は常に遮蔽（`H=+∞` 扱い）。

実装上の優先順位
- まず `size_lwh_m[2]` を高さとして使う
- 無い/0 の場合はカテゴリ既定値（config で固定）

補足（重要）
- 2.5D は first-hit の遮蔽破綻を減らすのが主目的。対象家具カテゴリを大物中心に固定すると（chair を除外など）指標が安定する。

## 4. 実装設計（eval_metrics.py）

### 4.1 追加する config
`experiments/configs/eval/default_eval.json` と `default_eval_config()` に追加。

- `entry_observability.enabled`（bool）
- `entry_observability.mode`（str: `first_hit` / `surface` / `both`）
- `entry_observability.exclude_categories`（list[str], 例 `floor, door, window`）
- `entry_observability.target_categories`（optional list[str]）
- `entry_observability.height_by_category_m`（dict[str,float]）

first-hit 用
- `entry_observability.sensor_height_m`（float, 例 0.6）
- `entry_observability.num_rays`（int, 例 720）
- `entry_observability.max_range_m`（float, 例 10.0 or room diag）
- `entry_observability.tau_p`（float, 例 0.02）

surface 用
- `entry_observability.tau_v`（float, 例 0.3）

### 4.2 新規ヘルパ（first-hit）
`experiments/src/eval_metrics.py` に関数を追加。

- `_build_occ_sense_and_obj_id_grid(layout, bounds, resolution, room_mask, h_s, height_by_category_m, exclude_categories)`
  出力: `occ_sense`（bool grid）, `obj_id_grid`（int grid）, `id_to_meta`（id→{category,height,...}）
  方針: `room_mask` 内だけを対象に、各家具の OBB 内セルを塗る
  `obj_id_grid` の意味: 「そのセルを占有している物体 id」（壁は `WALL_ID=-1` など）

- `_raycast_first_hit(sensor_cell, end_cell, occ_sense, obj_id_grid, room_mask)`
  方針: 既存 `_bresenham_line(a,b)` を流用して最短で作る
  手順: `line` を先頭から辿り、最初に `occ_sense==1` のセルに当たったら `obj_id_grid` を返す
  停止条件: ルーム外に出たら `NONE` 扱いで停止

- `_compute_entry_observability_first_hit(start_cell, occ_sense, obj_id_grid, id_to_meta, config)`
  手順: `K` 本の角度を生成（`0..2pi` の等間隔）
  手順: `max_range_m` をセル数に変換して `end_cell` を作る
  手順: 各レイの hit id を集計して `p_i` を計算
  手順: `C_obj_entry_hit` と `R_rec_entry_hit` を計算

### 4.3 新規ヘルパ（surface）
- `_compute_entry_observability_surface(layout, start_cell, occ, room_mask, bounds, resolution, visible_free, config)`
  方針: 2.3 の境界サンプルを実装し、`v_i` と `A_vis_i` を per-object で返す
  既存実装がある場合: 既存の `obj_vis_start` 系を OOE(surface) の出力にぶら下げる

### 4.4 `evaluate_layout()` への接続
- `entry_observability.enabled==true` のときだけ計算する
- `entry_observability.mode` に応じて first-hit / surface / 両方を計算する
- 既存の `metrics` に追加する

推奨の出力キー（解析しやすい形）
- `C_vis_start`
- `OOE_C_obj_entry_hit`
- `OOE_R_rec_entry_hit`
- `OOE_C_obj_entry_surf`
- `OOE_R_rec_entry_surf`
- `OOE_per_object`（list[dict]）

`OOE_per_object` の 1 件例
- `id`（str）
- `category`（str）
- `p_hit`（float, first-hit）
- `v_surf`（float, surface proxy）
- `visible_side_area_m2`（float, surface proxy）
- `height_m`（float）

## 5. デバッグ出力（--debug_dir）
`evaluate_layout(..., debug_dir=...)` のときに保存して原因切り分けできるようにする。

- `debug/ooe_occ_sense.pgm`: `occ_sense` を可視化（first-hit）
- `debug/ooe_obj_id.json`: `id_to_meta` を保存
- `debug/ooe_hits.json`: `hit_counts` と `p_i` を保存
- `debug/ooe_surface.json`: `v_i` と `A_vis_i` を保存（surface）
- 任意: `debug/ooe_rays.json`: 代表 N 本（例 36 本）だけレイのセル列を保存

## 6. 性能設計（refine の評価回数に耐える）
- `evaluate_layout()` は refine 中に多数回呼ばれる。
- `entry_observability.enabled` をデフォルト false にして、実験用 config で true にする。
- もしくは refine 側で中間評価は `enabled=false`、最終採用レイアウトだけ true で評価する。

## 7. テスト計画（DoD）
自動テストが無い場合は、最低限この手順を通す。

- DoD-1: `eval_metrics.py` を 1 つの layout で実行して `metrics.json` に `OOE_*` が出る
- DoD-2: `--debug_dir` を付けて `ooe_occ_sense.pgm` と `ooe_hits.json`（+ `ooe_surface.json`）が出る
- DoD-3: 家具を 1 つ手前に置いたレイアウトと置かないレイアウトで `OOE_*` が直感通り変化する

## 8. コミット分割案
- commit-1: config 追加（default_eval.json + default_eval_config）
- commit-2: eval_metrics に OOE(first-hit) 実装（occ_sense + raycast + metrics 出力）
- commit-3: eval_metrics に OOE(surface) 実装（境界サンプル + per-object 出力）
- commit-4: README/実験手順の追記（メトリクスの意味と debug の見方）

## 9. 実装前にユーザーが決める値（固定して再現性を担保）
- `sensor_height_m`（例 0.6 or 1.0）
- `num_rays`（例 720 or 1440）
- `tau_p`（例 0.02）
- `tau_v`（例 0.3）
- `exclude_categories` と `target_categories`
- `height_by_category_m`（カテゴリ高さの既定値）
