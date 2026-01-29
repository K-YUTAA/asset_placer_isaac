# 詳細設計書: USD Search Placer（my.research.asset_placer_isaac）

最終更新: 2026-01-19  
対象: Isaac Sim / Omniverse Kit Extension

## 1. 目的・背景
本拡張は、2Dの間取り画像と寸法情報を入力として、AIでレイアウトJSONを生成し、USD Search（ベクトル検索）を用いてアセットを自動配置するための研究用拡張です。  
研究用途として「AI解析 → JSON生成 → 3D配置 → 置換/調整」の一連の実験が行えることを重視しています。

## 2. できること（機能一覧）
- **画像+寸法からAIでJSON生成**（Step1/Step2/Step3）
- **JSONロードによる直接配置**
- **USD Searchによるアセット検索・配置（参照配置）**
- **床/壁/窓の手続き生成**
- **アセット回転オフセットの保存・一括反映**
- **ブラックリスト登録と置換（Replace Only対応）**
- **入力ファイルのプレビュー（画像/寸法/プロンプト）**
- **生成JSONのプレビュー**
- **USD Search Tester（検索結果をUI表示・ブラックリスト）**
- **AI Advanced Settings（モデル/推論/verbosity/画像詳細/トークン）**
- **承認ワークフロー（Step1後のApprove/Reject）**

## 3. 全体構成
```
my/research/asset_placer_isaac/
├── core/                    # UI/設定/操作/配置の中核
│   ├── extension_app.py      # IExtエントリ・初期化
│   ├── ui.py                 # UI構築
│   ├── handlers.py           # UIイベント
│   ├── commands.py           # AI/検索/配置
│   ├── settings.py           # 設定保存
│   ├── state.py              # 回転/ブラックリスト/メタデータ
│   └── constants.py          # プロンプト・モデル一覧
├── backend/
│   ├── ai_processing.py      # OpenAI処理（Step1/Step2/Step3）
│   └── file_utils.py         # 画像/テキスト読み込み
├── procedural/
│   ├── floor_generator.py    # 床生成
│   ├── wall_generator.py     # 壁/開口/ガラス生成
│   └── door_detector.py      # ドア判定補助
└── tests/                    # テストモジュール
```

## 4. UI設計
### 4.1 メインウィンドウ
`ui.Window("USD Search Placer")`  
2タブ構成:
- **Tab1: Generate from Image**
  - 画像/寸法/プロンプト選択
  - AI Model選択 + Advanced Settings
  - OpenAI API Key入力
  - Search Root URL
  - Search Testerボタン
  - Asset Orientation Offset領域
  - 承認チェックボックス
  - Generate JSON / Preview JSON
  - AI分析結果表示 + Approve/Rejectボタン
- **Tab2: Load from File**
  - JSONファイル選択 → 配置開始
  - Search Root URL / Search Tester / Orientation Offset

### 4.2 サブウィンドウ
- **AI Advanced Settings**
  - Step1/Step2モデル、推論強度、verbosity、画像detail、max tokens
- **Selected File Preview**
  - 画像/寸法/プロンプトのプレビュー
  - `ByteImageProvider` で画像表示
- **Generated JSON Preview**
  - 生成されたJSONを読み取り専用表示
- **Blacklisted Assets**
  - ブラックリストの閲覧、削除、全消去
- **USD Search Tester**
  - クエリ入力、検索結果サムネイル表示、ブラックリスト登録

## 5. 設定・永続化
### 5.1 `extension_settings.json`
保存項目:
- `openai_api_key`
- `search_root_url`
- `image_path`, `dimensions_path`, `prompt1_path`, `prompt2_path`
- `json_output_dir`
- `model_index`
- `ai_step1_model_index`, `ai_step2_model_index`
- `ai_reasoning_effort_index`, `ai_text_verbosity_index`, `ai_image_detail_index`
- `ai_max_output_tokens`
- `asset_blacklist`, `asset_blacklist_keys`

### 5.2 `asset_rotation_offsets.json`
アセットURLごとの回転オフセット（度）を保存。

### 5.3 キャッシュ
- `_asset_up_axis_cache`: asset_url → upAxis
- `_asset_size_cache`: asset_url → size_bytes
- `_glass_material_cache`: root_prim_path → Glass Material

## 6. AI処理設計（backend）
### 6.1 Step1: 画像解析
- 関数: `backend/ai_processing.py::step1_analyze_image()`
- 入力: 画像Base64、Prompt1、寸法テキスト、モデル名
- OpenAI Responses API を優先使用し、失敗時は `chat.completions` にフォールバック
- 推論強度/verbosity/画像detail/トークン上限は UI の Advanced Settings で上書き可能
- 既定値: `reasoning=high`, `text_verbosity=high`, `image_detail=high`, `max_output_tokens=16000`
- 出力: 解析テキスト + 統計（時間/トークン）

### 6.2 Step2: JSON生成
- 関数: `backend/ai_processing.py::step2_generate_json()`
- Prompt2 + Step1解析 + 寸法 + 画像 を統合したプロンプトで JSON を生成
- Responses API を優先、フォールバック時は `response_format={"type": "json_object"}` を使用
- 生成JSONは `_extract_json_from_text()` で厳密/抜粋パース対応

### 6.3 Step3: 衝突検出
- 関数: `backend/ai_processing.py::step3_check_collisions()`
- AABBで衝突ペアを検出
- Floorは衝突対象から除外
- 高さ方向の簡易モデルのため、縦方向の重なり判定は過検出の可能性あり

### 6.4 プロンプト解決
`core/settings.py::_resolve_prompt_text()`  
優先順位:  
1) UIで指定されたファイルパス  
2) `prompt_1.txt` / `prompt_2.txt`（相対パス）  
3) `core/` 配下の同名ファイル  
4) `core/data/` 配下  
5) 拡張ルート直下  
6) 見つからない場合は `core/constants.py` のデフォルト文字列

## 7. JSONスキーマと座標系
### 7.1 スキーマ（要約）
```
{
  "area_name": string,
  "area_size_X": number,
  "area_size_Y": number,
  "room_polygon": [ { "X": number, "Y": number }, ... ],    // 任意
  "windows": [ { "X": number, "Y": number, "Width": number, "Height": number, "SillHeight": number } ], // 任意
  "area_objects_list": [
    {
      "object_name": string,
      "category": string,
      "search_prompt": string,
      "X": number,
      "Y": number,
      "Length": number,
      "Width": number,
      "Height": number,
      "rotationZ": number
    }
  ]
}
```

### 7.2 座標系
- **X**: 左が +X、右が -X  
- **Y**: 奥行き（画像上方向が +Y）  
- **Z**: 上方向（高さ）  
- **単位**: meters

### 7.3 rotationZ
`0 / 90 / 180 / 270` の4値を想定。  
実配置ではアセットごとの回転オフセットが加算される。

## 8. アセット検索と配置
### 8.1 配置フロー
1. `layout_json` からオブジェクト配列を抽出
2. 床/ドア/窓を分類
3. 各オブジェクトに対して検索クエリを作成
4. ベクトル検索APIからUSD URLを取得
5. USD Reference として配置
6. `_apply_transform()` でスケール/回転/位置合わせ

### 8.2 Search Root
UIの `Search Root URL` は `omniverse://` で開始する必要があります。  
末尾が `/` でなければ自動補完されます。

### 8.3 参照配置
配置は **USD Reference** 方式（コピーではない）。  
Undo対象外になるため、実験前のステージ保存推奨。

### 8.4 トランスフォーム適用
`core/commands.py::_apply_transform()` の主要処理:
- 参照アセットのロード完了を最大10フレーム待機
- BBoxから元サイズを取得
- 目標サイズ（Length/Width/Height）に合わせてスケールを計算
- 回転Zと回転オフセットを合成
- upAxisがYの場合は `RotateX(90)` を挿入
- 回転+スケール後のBBoxから床面合わせのZを計算
- Translateを `(X, Y, -min_z_after_rot_scale)` に設定
- 配置メタデータを `asset_placer.*` に保存

## 9. USD Search（ベクトル検索）
### 9.1 外部API
`http://192.168.11.65:30080/search` へ POST  
Basic認証: `omniverse:tsukuverse`  
返却はUSDファイルURLを直接利用。

### 9.2 フィルタ
- `asset_blacklist`（URL）
- `asset_blacklist_keys`（同一性キー）
- 同一性キー: `basename + ext + size_bytes`

### 9.3 重複回避・再検索
- `limit` を段階的に拡張
- `min_unique` に到達するまで候補を増やす
- Replace時は `min_unique=10` を要求

## 10. 手続き生成
### 10.1 床
`procedural/floor_generator.py`
- **矩形床**: `UsdGeom.Cube` + 厚み0.1m（上面がZ=0）
- **ポリゴン床**: `UsdGeom.Mesh` + ファン分割（簡易三角化）

### 10.2 壁
`procedural/wall_generator.py`
- **矩形壁**: 東西南北の4面を生成
- **ポリゴン壁**: 外周エッジに沿って生成
- **壁厚**: デフォルト0.10m
- **開口**: Door/Windowの位置から壁セグメントを分割
- **端マージン**: `EDGE_END_MARGIN=0.1` で壁端の欠損を抑制

### 10.3 窓ガラス
- 窓開口に対して `UsdGeom.Cube` を作成
- `UsdPreviewSurface` で半透明（opacity=0.2, ior=1.5）

## 11. 回転オフセット / 置換 / ブラックリスト
### 11.1 回転オフセット
- `Use Selected Prim` で参照Primを探索
- `Save & Apply` で同一アセットのPrimを一括更新
- 保存先: `asset_rotation_offsets.json`

### 11.2 ブラックリスト
- URLブラックリストと同一性キーの二重管理
- `Blacklist & Replace` で即時置換
- `Replace Only` はブラックリスト追加なし

### 11.3 置換ロジック
- 選択Primのメタデータから検索クエリを再構築
- `exclude_urls` と `exclude_keys` で同一モデルの再選出を防止
- 置換後も元の配置寸法で再スケール

## 12. USD Search Tester
### 12.1 目的
検索精度確認とブラックリスト管理のUIテスト用

### 12.2 挙動
- `target_valid=10` になるまで再検索
- サムネイルは `SEARCH_TEST_THUMB_SIZE=96`
- 表示上限 `SEARCH_TEST_DISPLAY_LIMIT=10`
- ブラックリストされた候補は即時再描画で除外

## 13. 非同期/イベント設計
- AI処理/配置処理は `asyncio.ensure_future` で非同期実行
- Vector Searchは `run_in_executor` で同期HTTPを非同期化
- `ui.Window` 破棄は `next_update_async` を使い、イベント中destroyを回避
- FilePickerも同様に次フレームで安全破棄

## 14. ログと観測
- すべて `omni.log` を使用
- AI/検索/配置の主要ステップはINFOで出力
- エラーはWARN/ERRORで記録

## 15. 依存関係
- Omni: `omni.ext`, `omni.ui`, `omni.usd`, `omni.client`, `omni.kit.app`, `omni.kit.window.filepicker`
- USD: `pxr`（Usd/UsdGeom/Sdf/Gf/UsdShade）
- AI: `openai`（AsyncOpenAI）
- その他: `requests`, `numpy`, `PIL`, `cv2`, `base64`

## 16. 現在テストしている内容（研究用途）
- 画像→JSON→配置までの通しフロー（Step1/Step2/Step3）
- ポリゴン床 + 窓開口 + ガラス生成の整合性
- 置換機能（Blacklist/Replace Only）と同一性キーの挙動
- 回転オフセットの一括反映（同一参照Prim）
- Search Testerの10件確保とサムネイル表示

## 17. 既知の制約・注意点（要約）
- USD Searchの精度はプロンプト/名前依存
- AI処理のキャンセル機能なし
- USD Reference配置はUndo不可
- 衝突検出は簡易（高さ方向で過検出あり）
- 対応画像形式はJPEG/PNG/BMPが中心
- Search Root URLの設定ミスで検索不可

## 18. 出力ファイル
- 生成JSON: `<project_root>/json` に `{image_basename}_layout_{model}_{timestamp}.json`
- 設定: `<extension_dir>/extension_settings.json`
- 回転オフセット: `<extension_dir>/asset_rotation_offsets.json`
