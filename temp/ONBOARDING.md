# オンボーディング - glue-factory komainu_colmap 訓練実装

## プロジェクト情報

- **リポジトリ**: yuki-inaho/glue-factory
- **ブランチ**: `claude/update-readme-docs-011CV1sZs4vLP9sAsAvVsCmw`
- **開始日**: 2025-11-11
- **主目的**: TDD方法論を用いたkomainu_colmapデータセットの訓練パイプライン実装

## 現在の状況

### 完了した作業

1. **環境構築** - 完了
   - UVパッケージマネージャー設定完了
   - 119パッケージインストール（base 108 + extra 5 + dev）
   - 不足依存関係の修正: tensordict, plotly
   - 全インポート検証済み

2. **データセット実装** - 完了
   - `gluefactory/datasets/komainu_colmap.py` 作成
   - ColmapImagePairsDatasetを継承
   - komainu_colmap COLMAP再構成から30画像をロード
   - 10,771個の3Dポイントを用いた共視ペア生成

3. **テストスイート** - 完了
   - `tests/test_komainu_colmap.py` 作成
   - TDDアプローチに従った9個の包括的テスト
   - 全テスト合格 (9/9)
   - テストカバー範囲: 初期化、views.txt生成、ペア抽出、データロード、前処理、ポーズ計算、ホモグラフィ計算、sparse depth生成

4. **Sparse Depth実装** - 完了
   - COLMAP 3Dポイントからの深度取得実装
   - `_generate_sparse_depth()` ヘルパー関数（103行）
   - キーポイント位置での深度サンプリング
   - depth_keypoints0/1, valid_depth_keypoints0/1 生成

5. **設定ファイル** - 完了
   - `gluefactory/configs/data/komainu_colmap.yaml` 作成
   - `gluefactory/configs/komainu_train_homography.yaml` 作成（ホモグラフィベース）
   - `gluefactory/configs/komainu_train_depth.yaml` 作成（Sparse Depthベース）

6. **バグ修正** - 完了
   - PyTorch 2.1.2互換性問題（torch.compiler.set_stance）修正
   - scene_list設定でrootを単一シーンとして扱うよう修正
   - テストアサーションを正しいテンソル構造に修正
   - KeypointsをトップレベルにコピーしてDepthMatcher対応

### 進行中

- **訓練パイプライン検証**: Extractor設定追加でDry run実行可能

### 保留/次のステップ

- Dry run訓練実行（extractor設定追加後）
- num_matchable改善の確認
- 実際の訓練実行（1エポック以上）
- コードフォーマット・Linting追加
- README.md更新（komainu_colmap使用方法）

## 環境構築

### 前提条件

- Python 3.10以上
- UVパッケージマネージャー
- CUDA対応GPU（推奨）

### セットアップ手順

```bash
# UVをインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係の同期
uv sync --extra extra --dev

# インストール確認
uv run python -c "import gluefactory; print('OK')"
```

### 主要な依存関係

- PyTorch 2.1.2 with CUDA
- pycolmap 0.6.1
- tensordict（カスタムバッチング）
- plotly 6.4.0（可視化）
- pytest（テスト）
- hydra-core（設定管理）

## プロジェクト構造

### 主要ファイル

```
gluefactory/
├── datasets/
│   └── komainu_colmap.py          # データセットローダー（新規）
├── configs/
│   ├── data/
│   │   └── komainu_colmap.yaml    # データセット設定（新規）
│   ├── komainu_train_homography.yaml  # 訓練設定 - ホモグラフィ（新規）
│   └── komainu_train_depth.yaml       # 訓練設定 - Sparse Depth（新規）
├── trainer.py                      # PyTorch 2.1互換性修正済み
tests/
└── test_komainu_colmap.py         # テストスイート（新規、9テスト）
temp/
├── workdoc_komainu_training.md    # TDD作業計画
├── workdoc_sparse_depth_implementation.md  # Sparse Depth実装作業記録
├── COLMAP_LEARNING_PROCESS.md     # 学習プロセス詳細解説（新規）
└── ONBOARDING.md                   # 本ファイル
data/
└── komainu_colmap/                # データセット（30画像）
    ├── images/
    ├── sparse/0/
    ├── views.txt                   # 自動生成
    └── covisibility/               # 自動生成ペア
```

### データセット構造

komainu_colmapデータセットの内容:
- 30枚のRGB画像（`data/komainu_colmap/images/`）
- COLMAPスパース再構成（`data/komainu_colmap/sparse/0/`）
- 10,771個の3Dポイント
- 自動生成されたビューポーズと共視ペア（237ペア）

## テスト実行

```bash
# 全komainu_colmapテストを実行
uv run pytest tests/test_komainu_colmap.py -v

# 特定のテストを実行
uv run pytest tests/test_komainu_colmap.py::TestKomainuColmapDataset::test_dataset_init -v

# 出力付きで実行
uv run pytest tests/test_komainu_colmap.py -v -s
```

期待される出力: 9個のテスト全て合格

## 訓練

### 現在の設定

訓練で使用:
- **Extractor**: SuperPoint（凍結、事前訓練済み）
- **Matcher**: LightGlue（訓練可能）
- **Ground Truth方式1**: Homography matcher（平面仮定）
- **Ground Truth方式2**: Depth matcher（Sparse Depth、推奨）
- **データセット**: komainu_colmap（237ペア）

### 訓練コマンド

```bash
# Dry run（0エポック）- Homographyベース
uv run python -m gluefactory.train komainu_train_homography --conf komainu_train_homography train.epochs=0 --overwrite

# Dry run（0エポック）- Sparse Depthベース（推奨）
uv run python -m gluefactory.train komainu_train_depth --conf komainu_train_depth train.epochs=0 --overwrite

# 実訓練（3エポック）
uv run python -m gluefactory.train komainu_train_depth --conf komainu_train_depth
```

### 2つのGround Truth方式の比較

| 方式 | num_matchable | 精度 | 適用範囲 |
|------|---------------|------|----------|
| Homography | 1.8 pts/pair | 平面仮定誤差あり | 平面的シーンのみ |
| Sparse Depth | 10+ pts/pair（期待） | SfM実測値 | 一般的3Dシーン |

**推奨**: Sparse Depth方式（`komainu_train_depth`）

## よくある問題と修正方法

### 問題: ModuleNotFoundError: tensordict
**修正**: `uv add tensordict` または pyproject.toml に追加確認

### 問題: ModuleNotFoundError: plotly
**修正**: `uv add plotly`

### 問題: torch.compiler.set_stance AttributeError
**修正**: trainer.py で既にパッチ済み（条件付きデコレータ）

### 問題: NotADirectoryError with README.md/covisibility
**修正**: データセット設定で `scene_list: ["."]` を使用してrootを単一シーンとして扱う

### 問題: テスト失敗（テンソルインデックス）
**修正**: データはTensorDict、リストでバッチ化されていない - [0]なしで直接アクセス

### 問題: Dry runでKeyError（depth_keypoints等）
**修正**: Extractor設定が必要、または訓練時は自動設定される

## 開発方法論

本プロジェクトは以下に従う:
- **TDD（テスト駆動開発）**: テストを先に書き、その後実装
- **t-wadaスタイル**: Red → Green → Refactor サイクル
- **DRY/KISS/SOLID原則**
- **行動カウント**: 全ての行動を追跡、20行動ごとにリマインダー
- **作業文書化**: temp/workdoc_*.md で詳細ログ維持

## 技術詳細

### Sparse Depth生成プロセス

1. **COLMAP 3Dポイント抽出**
   - 画像ごとに観測された3Dポイントを取得
   - points3D.xyzでワールド座標取得

2. **カメラ座標変換**
   - T_world2cam.transform()でカメラ座標系へ
   - depth = p3D_cam[2]（Z座標）

3. **ピクセル投影**
   - K @ (p3D_cam / depth) でピクセル座標計算
   - カメラ内部パラメータKを使用

4. **キーポイント深度サンプリング**
   - 最近傍探索（threshold=5px）
   - valid_maskで有効性を明示的にマーク

### 損失関数

```python
# Assignment NLL Loss
loss = -sum(gt_assignment * log(pred_assignment)) / N

where:
  pred_assignment: [N, M+1] (LightGlue出力)
  gt_assignment: [N, M+1] (Depth Matcherから生成)
```

詳細は `temp/COLMAP_LEARNING_PROCESS.md` を参照。

## 次のステップ

1. **訓練パイプライン検証**
   - Extractor設定追加でDry run実行
   - num_matchable改善確認（1.8 → 10+）
   - match_recall向上確認

2. **実訓練実行**
   - 1エポック以上の訓練
   - チェックポイント保存確認
   - Lossグラフ確認

3. **コード品質**
   - blackフォーマット追加
   - flake8 Linting追加
   - スタイルガイド遵守確認

4. **ドキュメント**
   - README.md更新（komainu_colmap使用方法）
   - 訓練例追加
   - 設定オプション文書化

## 参考資料

- メイン作業計画: `temp/workdoc_komainu_training.md`
- Sparse Depth作業記録: `temp/workdoc_sparse_depth_implementation.md`
- 学習プロセス詳解: `temp/COLMAP_LEARNING_PROCESS.md`
- 環境ログ: `temp/workdoc_nov11.md`
- 設定例: `gluefactory/configs/lightglue_megadepth.yaml`
- 親データセット: `gluefactory/datasets/colmap_image_pairs.py`

## 連絡・質問

問題や質問がある場合:
- temp/ 内の作業文書を確認
- テストスイートで使用例を確認
- 既存設定をパターンマッチングの参考に

---

**最終更新**: 2025-11-11
**状態**: データセット・テスト・Sparse Depth実装完了、訓練パイプライン検証待ち
