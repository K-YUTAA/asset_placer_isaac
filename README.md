# USD Search Placer for Isaac Sim

AI駆動の家具配置拡張機能 - フロアプラン画像から3Dシーンを自動生成

## 概要

USD Search Placerは、NVIDIA Omniverse Isaac Sim向けの拡張機能で、2Dの間取り画像と寸法情報を入力として、OpenAI GPTを使用してレイアウトJSONを生成し、USD Search（ベクトル検索）を用いてNucleusサーバーからアセットを自動検索・配置する研究用ツールです。

## 主な機能

### 🤖 AI駆動のレイアウト生成
- **画像解析（Step1）**: フロアプラン画像をOpenAI GPTで解析し、部屋の構造と家具の配置を理解
- **JSON生成（Step2）**: 解析結果から構造化されたレイアウトJSONを自動生成
- **衝突検出（Step3）**: 生成されたレイアウトの衝突を検出し、問題を報告

### 🔍 USD Search統合
- **ベクトル検索**: Nucleusサーバー上のUSDアセットを自然言語クエリで検索
- **自動配置**: 検索結果から最適なアセットを選択し、USD Referenceとして配置
- **ブラックリスト機能**: 不要なアセットをブラックリストに登録し、再検索を防止
- **置換機能**: 配置済みアセットを別のアセットに置き換え可能

### 🏗️ 手続き的生成
- **床生成**: 矩形またはポリゴン形状の床を自動生成
- **壁生成**: 部屋の外周に沿って壁を自動生成（開口部対応）
- **窓ガラス生成**: 窓開口部に半透明ガラスマテリアルを自動生成

### ⚙️ 高度な設定
- **AIモデル選択**: Step1/Step2で異なるモデルを選択可能
- **推論強度調整**: `low`, `medium`, `high`, `xhigh`から選択
- **詳細度調整**: テキスト詳細度と画像詳細度を個別に設定
- **トークン上限**: 出力トークン数の上限を設定

### 🎯 座標系と変換
- **座標系統一**: 右=+X / 左=-Xに統一された座標系
- **自動変換**: Y-Up座標系からZ-Up座標系への自動変換
- **単位変換**: cm単位からm単位への自動変換
- **回転オフセット**: アセットごとの回転オフセットを保存・一括適用

### 💾 レイアウト自動復元
- **Quick Layout対応**: 起動時にQuick Layoutを自動ロード
- **設定の永続化**: APIキー、検索ルートURL、ファイルパスなどを自動保存

## 最新の追加機能（2026-01-20）

### ✅ 安定性の向上
- **窓ガラスマテリアル接続エラー修正**: `UsdShade`の接続を正しい型で実行
- **エラーハンドリング強化**: ガラスマテリアル生成失敗時でも壁生成を継続

### ✅ 座標系の整合
- **左右座標系の統一**: Step1/Step2プロンプトを右=+X / 左=-Xに統一
- **rotationZの修正**: 左右定義を座標系に合わせて修正
- **X反転セーフティ**: JSON生成時に「左=+X」が検出された場合のみ自動反転

### ✅ 機能改善
- **窓の自動生成抑止**: 画像に明示されていない窓は生成しないルールを追加
- **レイアウト自動復元**: Quick Save/Quick Loadに対応した自動復元機能

## セットアップ

### 1. 設定ファイルの準備

```bash
cp my/research/asset_placer_isaac/extension_settings.json.example my/research/asset_placer_isaac/extension_settings.json
cp .claude/settings.local.json.example .claude/settings.local.json
```

### 2. 設定の編集

`extension_settings.json`を編集して以下を設定：
- `openai_api_key`: OpenAI APIキー
- `search_root_url`: Omniverse NucleusサーバーのアセットルートURL（`omniverse://`で開始）

**重要**: `extension_settings.json`や`.claude/settings.local.json`に実際のAPIキーをコミットしないでください！

### 3. 依存関係のインストール

拡張機能は以下のPythonパッケージを使用します（`config/extension.toml`で自動インストール）：
- `opencv-python==4.10.0.84`
- `openai`
- `pillow`
- `numpy>=1.21.2`

## 使用方法

### 方法1: 画像から生成（Tab 1）

1. **入力ファイルの選択**:
   - フロアプラン画像（JPEG/PNG/BMP）
   - 寸法テキストファイル
   - （オプション）カスタムプロンプトファイル

2. **AI設定の調整**:
   - AIモデルの選択
   - Advanced Settingsで推論強度、詳細度などを調整

3. **生成実行**:
   - "Generate JSON"ボタンをクリック
   - Step1で画像解析（承認ワークフローあり）
   - Step2でJSON生成
   - Step3で衝突検出

4. **アセット配置**:
   - 生成されたJSONをプレビュー
   - "Place Assets"ボタンで配置開始

### 方法2: JSONファイルから直接配置（Tab 2）

1. **JSONファイルの選択**:
   - 既存のレイアウトJSONファイルを選択

2. **配置実行**:
   - "Place Assets"ボタンで即座に配置開始

## プロジェクト構造

```
my.research.asset_placer_isaac/
├── config/
│   └── extension.toml          # 拡張機能マニフェスト
├── docs/                        # ドキュメント
│   ├── README.md               # このファイル
│   ├── Overview.md             # 概要
│   ├── ARCHITECTURE.md         # アーキテクチャ
│   ├── API_REFERENCE.md        # APIリファレンス
│   ├── SYSTEM_DESIGN.md        # 詳細設計書
│   ├── SETUP.md                # セットアップガイド
│   ├── TROUBLESHOOTING.md      # トラブルシューティング
│   └── CHANGELOG.md            # 変更履歴
├── my/research/asset_placer_isaac/
│   ├── core/                   # コア機能
│   │   ├── extension_app.py    # エントリーポイント
│   │   ├── ui.py               # UI構築
│   │   ├── handlers.py         # UIイベントハンドラ
│   │   ├── commands.py         # アセット配置コマンド
│   │   ├── settings.py         # 設定管理
│   │   ├── state.py            # 状態管理
│   │   └── constants.py        # 定数・プロンプト
│   ├── backend/                 # バックエンド処理
│   │   ├── ai_processing.py    # OpenAI処理
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

## JSONスキーマ

レイアウトJSONの構造：

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

## 主な機能の詳細

### USD Search統合
- **検索API**: `http://192.168.11.65:30080/search`へのPOSTリクエスト
- **認証**: Basic認証（`omniverse:tsukuverse`）
- **ベクトル検索**: 自然言語クエリでアセットを検索
- **重複回避**: ブラックリストと同一性キーで重複を防止

### アセット配置
- **参照配置**: USD Referenceとして配置（コピーではない）
- **自動スケーリング**: JSONの`Length`, `Width`, `Height`に基づいて自動スケール
- **回転処理**: `rotationZ`とアセット固有の回転オフセットを合成
- **座標変換**: Y-UpからZ-Upへの自動変換、cmからmへの変換

### 手続き的生成
- **床**: 矩形またはポリゴン形状に対応
- **壁**: 外周エッジに沿って自動生成、開口部で分割
- **窓ガラス**: 半透明マテリアル（opacity=0.2, ior=1.5）

## トラブルシューティング

詳細は[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)を参照してください。

### よくある問題

1. **USD Searchが結果を返さない**
   - Search Root URLが正しく設定されているか確認
   - `omniverse://`で開始する必要があります

2. **AI処理が失敗する**
   - OpenAI APIキーが正しく設定されているか確認
   - ネットワーク接続を確認

3. **アセットが正しく配置されない**
   - 回転オフセットを確認・調整
   - 座標系が正しいか確認（右=+X）

## 開発者向け情報

### エントリーポイント
`config/extension.toml`で定義：
- `my.research.asset_placer_isaac.core.extension_app`

### 依存関係
- **Omniverse**: `omni.ext`, `omni.ui`, `omni.usd`, `omni.client`, `omni.kit.app`
- **USD**: `pxr` (Usd, UsdGeom, Sdf, Gf, UsdShade)
- **AI**: `openai` (AsyncOpenAI)
- **その他**: `requests`, `numpy`, `PIL`, `cv2`

### テスト
```bash
# テストの実行
python -m pytest my/research/asset_placer_isaac/tests/
```

## ライセンス

NVIDIA Proprietary License

## リポジトリ

[GitHub Repository](https://github.com/K-YUTAA/asset_placer_isaac.git)

## 関連ドキュメント

- [Overview](docs/Overview.md) - 機能概要
- [Architecture](docs/ARCHITECTURE.md) - システム設計
- [API Reference](docs/API_REFERENCE.md) - API仕様
- [System Design](docs/SYSTEM_DESIGN.md) - 詳細設計書
- [Setup Guide](docs/SETUP.md) - セットアップガイド
- [Troubleshooting](docs/TROUBLESHOOTING.md) - トラブルシューティング
- [Changelog](docs/CHANGELOG.md) - 変更履歴
