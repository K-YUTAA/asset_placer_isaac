# Overview

USD Search Placer for Isaac Sim - AI駆動の家具配置拡張機能

## システム概要

USD Search Placerは、NVIDIA Omniverse Isaac Sim向けの拡張機能で、**2Dの間取り画像と寸法情報を入力として、OpenAI GPTを使用してレイアウトJSONを生成し、USD Search（ベクトル検索）を用いてNucleusサーバーからアセットを自動検索・配置する研究用ツール**です。

### 主な目的

研究用途として「**AI解析 → JSON生成 → 3D配置 → 置換/調整**」の一連の実験が行えることを重視しています。

## システムの動作フロー

### 方法1: 画像から生成（AI駆動）

```
[入力]
  ↓
1. フロアプラン画像 + 寸法テキスト + プロンプト
  ↓
2. Step1: OpenAI GPTで画像解析
  ├─ 部屋の構造を理解
  ├─ 家具の種類と位置を識別
  └─ 解析結果をテキストで出力（承認ワークフローあり）
  ↓
3. Step2: 解析結果からレイアウトJSON生成
  ├─ 構造化されたJSON形式で出力
  ├─ 家具の位置、サイズ、回転を定義
  └─ 座標系の自動変換（Y-Up → Z-Up、cm → m）
  ↓
4. Step3: 衝突検出（オプション）
  ├─ AABB（軸並行境界ボックス）で衝突を検出
  └─ 問題のある配置を報告
  ↓
5. USD Searchでアセット検索
  ├─ 各家具のobject_nameから検索クエリを生成
  ├─ Nucleusサーバーでベクトル検索を実行
  └─ 最適なUSDアセットを選択
  ↓
6. USD Stageに配置
  ├─ USD Referenceとして配置（コピーではない）
  ├─ 位置、スケール、回転を自動適用
  ├─ 床・壁・窓を手続き的に生成
  └─ メタデータを保存
  ↓
[出力: 3Dシーン]
```

### 方法2: JSONファイルから直接配置

```
[入力: 既存のレイアウトJSON]
  ↓
1. JSONファイルを読み込み
  ↓
2. バリデーション（構造チェック）
  ↓
3. USD Searchでアセット検索
  ↓
4. USD Stageに配置
  ↓
[出力: 3Dシーン]
```

## 主要機能

### 🤖 AI駆動のレイアウト生成

- **Step1: 画像解析**
  - フロアプラン画像をOpenAI GPTで解析
  - 部屋の構造と家具の配置を理解
  - 解析結果をテキストで出力（承認ワークフロー対応）

- **Step2: JSON生成**
  - 解析結果から構造化されたレイアウトJSONを生成
  - 家具の位置（X, Y）、サイズ（Length, Width, Height）、回転（rotationZ）を定義
  - 座標系の自動変換（Y-Up → Z-Up、cm → m）

- **Step3: 衝突検出**
  - AABB（軸並行境界ボックス）で衝突を検出
  - 問題のある配置を報告

### 🔍 USD Search統合

- **ベクトル検索**
  - Nucleusサーバー上のUSDアセットを自然言語クエリで検索
  - 検索API: `http://192.168.11.65:30080/search`（Basic認証）

- **自動配置**
  - 検索結果から最適なアセットを選択
  - USD Referenceとして配置（コピーではない）
  - 位置、スケール、回転を自動適用

- **ブラックリスト機能**
  - 不要なアセットをブラックリストに登録
  - 再検索時に除外される
  - URLと同一性キーで二重管理

- **置換機能**
  - 配置済みアセットを別のアセットに置き換え可能
  - 「Blacklist & Replace」と「Replace Only」モード

### 🏗️ 手続き的生成

- **床生成**
  - 矩形またはポリゴン形状の床を自動生成
  - `UsdGeom.Cube`または`UsdGeom.Mesh`を使用

- **壁生成**
  - 部屋の外周に沿って壁を自動生成
  - 開口部（ドア・窓）で壁セグメントを分割
  - 壁厚: デフォルト0.10m

- **窓ガラス生成**
  - 窓開口部に半透明ガラスマテリアルを自動生成
  - `UsdPreviewSurface`でopacity=0.2, ior=1.5

### ⚙️ 高度な設定

- **AIモデル選択**
  - Step1/Step2で異なるモデルを選択可能
  - GPT-4o-mini、GPT-4o、GPT-4o-reasoningなど

- **推論強度調整**
  - `low`, `medium`, `high`, `xhigh`から選択

- **詳細度調整**
  - テキスト詳細度（`low`, `medium`, `high`）
  - 画像詳細度（`low`, `high`）

- **トークン上限**
  - 出力トークン数の上限を設定（デフォルト: 16000）

### 🎯 座標系と変換

- **座標系統一**
  - 右=+X / 左=-Xに統一された座標系
  - X反転セーフティ機能

- **自動変換**
  - Y-Up座標系からZ-Up座標系への自動変換
  - cm単位からm単位への自動変換

- **回転オフセット**
  - アセットごとの回転オフセットを保存
  - 同一アセットのPrimを一括更新

### 💾 レイアウト自動復元

- **Quick Layout対応**
  - 起動時にQuick Layoutを自動ロード
  - Quick Save/Quick Loadに対応

- **設定の永続化**
  - APIキー、検索ルートURL、ファイルパスなどを自動保存
  - `extension_settings.json`に保存

## UI構成

### メインウィンドウ

**Tab1: Generate from Image**
- 画像/寸法/プロンプトファイルの選択
- AI Model選択 + Advanced Settings
- OpenAI API Key入力
- Search Root URL設定
- Search Testerボタン
- Asset Orientation Offset領域
- 承認チェックボックス
- Generate JSON / Preview JSONボタン
- AI分析結果表示 + Approve/Rejectボタン

**Tab2: Load from File**
- JSONファイル選択
- Search Root URL / Search Tester / Orientation Offset
- Place Assetsボタン

### サブウィンドウ

- **AI Advanced Settings**: Step1/Step2モデル、推論強度、verbosity、画像detail、max tokens
- **Selected File Preview**: 画像/寸法/プロンプトのプレビュー
- **Generated JSON Preview**: 生成されたJSONを読み取り専用表示
- **Blacklisted Assets**: ブラックリストの閲覧、削除、全消去
- **USD Search Tester**: クエリ入力、検索結果サムネイル表示、ブラックリスト登録

## データフロー

### JSONスキーマ

```json
{
  "area_name": "LivingRoom",
  "area_size_X": 5.0,
  "area_size_Y": 6.0,
  "room_polygon": [
    { "X": -2.5, "Y": -3.0 },
    { "X": 2.5, "Y": -3.0 },
    { "X": 2.5, "Y": 3.0 },
    { "X": -2.5, "Y": 3.0 }
  ],
  "windows": [
    {
      "X": 1.2,
      "Y": 3.0,
      "Width": 1.2,
      "Height": 1.0,
      "SillHeight": 0.9
    }
  ],
  "area_objects_list": [
    {
      "object_name": "Sofa",
      "category": "Furniture",
      "search_prompt": "modern sofa",
      "X": 1.2,
      "Y": 0.8,
      "Length": 2.0,
      "Width": 0.9,
      "Height": 0.8,
      "rotationZ": 0
    }
  ]
}
```

### 座標系

- **X**: 水平方向（右=+X、左=-X）
- **Y**: 奥行き方向（画像上方向=+Y）
- **Z**: 高さ方向（上=+Z）
- **単位**: メートル（m）

## システム構成

### ディレクトリ構造

```
my.research.asset_placer_isaac/
├── config/
│   └── extension.toml          # 拡張機能マニフェストとエントリーポイント
├── docs/                       # ドキュメント
├── my/research/asset_placer_isaac/
│   ├── core/                   # コア機能
│   │   ├── extension_app.py    # IExtエントリーポイント
│   │   ├── ui.py               # UI構築
│   │   ├── handlers.py         # UIイベントハンドラ
│   │   ├── commands.py         # アセット配置コマンド
│   │   ├── settings.py         # 設定管理
│   │   ├── state.py            # 状態管理（回転オフセット、ブラックリスト）
│   │   └── constants.py        # 定数・プロンプト・モデル一覧
│   ├── backend/                 # バックエンド処理
│   │   ├── ai_processing.py    # OpenAI処理（Step1/Step2/Step3）
│   │   └── file_utils.py       # ファイルユーティリティ
│   ├── procedural/             # 手続き的生成
│   │   ├── floor_generator.py  # 床生成
│   │   ├── wall_generator.py   # 壁生成
│   │   └── door_detector.py   # ドア検出
│   └── tests/                  # テスト
└── data/                       # リソース
    ├── icon.png
    └── preview.png
```

### エントリーポイント

`config/extension.toml`で定義：
- `my.research.asset_placer_isaac.core.extension_app`

### 主要モジュール

- **`core/extension_app.py`**: 拡張機能のエントリーポイント、UIの初期化
- **`core/commands.py`**: USD Search、アセット配置、トランスフォーム適用
- **`backend/ai_processing.py`**: OpenAI APIとの通信、画像解析、JSON生成
- **`procedural/wall_generator.py`**: 壁と窓ガラスの生成
- **`procedural/floor_generator.py`**: 床の生成

## 技術スタック

### 依存関係

- **Omniverse**: `omni.ext`, `omni.ui`, `omni.usd`, `omni.client`, `omni.kit.app`
- **USD**: `pxr` (Usd, UsdGeom, Sdf, Gf, UsdShade)
- **AI**: `openai` (AsyncOpenAI)
- **画像処理**: `opencv-python`, `PIL` (Pillow)
- **数値計算**: `numpy`

### 外部サービス

- **OpenAI API**: 画像解析とJSON生成
- **USD Search API**: Nucleusサーバー上のアセット検索
- **Omniverse Nucleus**: USDアセットのストレージ

## 設定ファイル

### `extension_settings.json`

保存項目:
- `openai_api_key`: OpenAI APIキー
- `search_root_url`: NucleusサーバーのアセットルートURL
- `image_path`, `dimensions_path`, `prompt1_path`, `prompt2_path`: ファイルパス
- `json_output_dir`: JSON出力ディレクトリ
- `model_index`: デフォルトモデル
- `ai_step1_model_index`, `ai_step2_model_index`: Step1/Step2のモデル
- `ai_reasoning_effort_index`, `ai_text_verbosity_index`, `ai_image_detail_index`: AI設定
- `ai_max_output_tokens`: トークン上限
- `asset_blacklist`, `asset_blacklist_keys`: ブラックリスト

### `asset_rotation_offsets.json`

アセットURLごとの回転オフセット（度）を保存。

## 最新の機能追加（2026-01-20）

### ✅ 安定性の向上
- 窓ガラスマテリアル接続エラー修正
- エラーハンドリング強化

### ✅ 座標系の整合
- 左右座標系の統一（右=+X / 左=-X）
- X反転セーフティ機能

### ✅ 機能改善
- 窓の自動生成抑止
- レイアウト自動復元機能

## 関連ドキュメント

- [ARCHITECTURE.md](ARCHITECTURE.md) - システム設計とワークフロー
- [API_REFERENCE.md](API_REFERENCE.md) - API仕様
- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) - 詳細設計書
- [SETUP.md](SETUP.md) - セットアップガイド
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - トラブルシューティング
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) - 既知の問題
