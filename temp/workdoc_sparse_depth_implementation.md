# 作業計画書 兼 記録書: Sparse Depth実装（COLMAP 3Dポイントベース）

---

**日付:** 2025年11月11日
**作業ディレクトリ・リポジトリ:** `/home/user/glue-factory (yuki-inaho/glue-factory)`
**ブランチ:** `claude/update-readme-docs-011CV1sZs4vLP9sAsAvVsCmw`
**作業者:** Claude Code Assistant
**作業開始時刻:** 2025-11-11 11:08:13 UTC+0000

---

## 1. 作業目的

本作業は、以下の目標を達成するために実施します。

* **目標1:** COLMAPのsparse 3Dポイントから深度マップを生成する機能を実装
* **目標2:** TDD（テスト駆動開発）アプローチによる段階的・確実な実装
* **目標3:** depth_matcherを使用した、より正確なground truth生成の実現
* **目標4:** num_matchableの改善（現状1.8 → 期待値: 10以上）

---

## 2. 背景・現状課題

### 現在の実装状況
- ✅ ホモグラフィベースのground truth実装完了
- ✅ トレーニングパイプライン動作確認完了（dry run成功）
- ✅ 全8テスト通過

### 現在の課題
- ❌ **num_matchableが少ない**: 1.8ポイント/ペア（平面仮定の制約）
- ❌ **単一平面仮定**: z=0平面, d=2.0m固定が実シーンと合わない
- ❌ **match_recall 0.0**: matchableが少なすぎてマッチ生成困難

### 改善方針（ユーザー提案）
COLMAPのsparse 3Dポイントを活用:
1. Covisible 3D pointsを取得
2. 各ポイントをview0/view1に投影して深度計算
3. Sparse depth mapsを生成（キーポイント位置での深度のみ）
4. depth_matcherに切り替え

### 期待される効果
- ✅ **実測深度**: SfM由来の正確な3D構造使用
- ✅ **より多くのmatchable**: 実際の3D制約に基づく
- ✅ **既存実装活用**: depth_matcherがそのまま使える

---

## 3. 設計方針

### アーキテクチャ概要

#### データフロー
```
COLMAP 3D Points (self.reconstructions[scene].points3D)
  ↓
Covisible Points抽出 (image0, image1で共有される3Dポイント)
  ↓
カメラ座標系へ変換 (T_world2cam @ p3D_world)
  ↓
深度値取得 (p3D_cam[2])
  ↓
ピクセル座標へ投影 (K @ (p3D_cam / depth))
  ↓
Sparse Depth Maps生成
  - depth_keypoints0/1: キーポイント位置での深度
  - valid_depth_keypoints0/1: 有効性マスク
```

#### 実装箇所
- **ファイル**: `gluefactory/datasets/komainu_colmap.py`
- **メソッド**: `__getitem__` をさらに拡張
- **追加データキー**:
  - `depth_keypoints0`: view0のキーポイント位置での深度 [N, 1]
  - `valid_depth_keypoints0`: view0の深度有効性マスク [N, 1]
  - `depth_keypoints1`: view1のキーポイント位置での深度 [M, 1]
  - `valid_depth_keypoints1`: view1の深度有効性マスク [M, 1]

#### 依存関係
- pycolmap: 3Dポイント・カメラパラメータ取得
- torch: 行列演算・テンソル操作
- 既存のPoseクラス: ワールド→カメラ座標変換

---

## 4. 作業内容（詳細チェックリスト）

### フェーズ 1: 3Dポイント取得の理解・検証 (見積: 0.3h)

#### 手順 1-1: COLMAP 3Dポイント構造の確認
- [x] 🖐 **操作**: `uv run python -c "import pycolmap; sfm = pycolmap.Reconstruction('data/komainu_colmap/sparse/0'); print(f'Points3D: {len(sfm.points3D)}'); pt = list(sfm.points3D.values())[0]; print(f'Sample point: xyz={pt.xyz}, track_length={len(pt.track)}')"`
- [x] 🔎 **確認**: 3Dポイントの座標・track情報が取得できる
- [x] 🧪 **テスト**: point3D_structure_verified
- [x] 🛠 **エラー時対処**:
  - AttributeError → pycolmapのドキュメント確認
  - 空のpoints3D → sparse/0/のデータ確認

#### 手順 1-2: Covisibleポイントの確認
- [x] 🖐 **操作**: `gluefactory/datasets/pairs_from_colmap.py`の`extract_covisible_pairs()`を読み、covisible_point_idsの取得方法を理解
- [x] 🔎 **確認**: 画像ペアごとにcovisible_point_idsが抽出できることを確認
- [x] 🧪 **テスト**: covisible_extraction_understood
- [x] 🛠 **エラー時対処**: コードリーディングで理解

#### 手順 1-3: カメラパラメータ・ポーズの確認
- [x] 🖐 **操作**: 既存の`__getitem__`から取得できるK0, K1, T_0to1を確認
- [x] 🔎 **確認**: カメラ内部パラメータ・相対ポーズが利用可能
- [x] 🧪 **テスト**: camera_params_available
- [x] 🛠 **エラー時対処**: N/A（既に実装済み）

---

### フェーズ 2: テスト作成（TDD - Red Phase） (見積: 0.4h)

#### 手順 2-1: Sparse Depth生成テストの追加
- [x] 🖐 **操作**: `tests/test_komainu_colmap.py`に`test_sparse_depth_generation()`を追加
  - `depth_keypoints0`, `depth_keypoints1`が存在することを確認
  - `valid_depth_keypoints0`, `valid_depth_keypoints1`が存在することを確認
  - 深度値が正（positive）であることを確認
  - 有効なポイント数が0より大きいことを確認
  - 深度値が妥当な範囲（0.1m～10m程度）にあることを確認
- [x] 🔎 **確認**: テストが失敗する（実装がまだないため）
- [x] 🧪 **テスト**: `uv run pytest tests/test_komainu_colmap.py::TestKomainuColmapDataset::test_sparse_depth_generation -v` → FAILED
- [x] 🛠 **エラー時対処**: N/A（失敗が期待される）

#### 手順 2-2: Depth値の一貫性テスト追加
- [x] 🖐 **操作**: 同テストに、キーポイント位置と深度値の対応検証を追加
  - キーポイント数と深度配列のサイズが一致
  - valid_maskがbooleanテンソルであること
- [x] 🔎 **確認**: テストが失敗する（実装がまだないため）
- [x] 🧪 **テスト**: `uv run pytest tests/test_komainu_colmap.py::TestKomainuColmapDataset::test_sparse_depth_generation -v` → FAILED
- [x] 🛠 **エラー時対処**: N/A（失敗が期待される）

---

### フェーズ 3: 実装（TDD - Green Phase） (見積: 1.0h)

#### 手順 3-1: Covisible 3Dポイント取得の実装
- [x] 🖐 **操作**: `__getitem__`内で以下を実装:
  ```python
  # ペア情報から画像IDを取得
  idx0, idx1 = self.pairs[idx]
  name0 = self.images[idx0]
  name1 = self.images[idx1]

  # COLMAPのimage objectを取得
  img0 = self.reconstructions[scene].images[image_id0]
  img1 = self.reconstructions[scene].images[image_id1]

  # Covisible 3D pointsを取得
  point_ids0 = set(img0.point2D_ids)
  point_ids1 = set(img1.point2D_ids)
  covisible_point_ids = point_ids0 & point_ids1
  ```
- [ ] 🔎 **確認**: covisible_point_idsが取得できる
- [ ] 🧪 **テスト**: print文でデバッグ確認
- [ ] 🛠 **エラー時対処**:
  - AttributeError → pycolmap APIドキュメント確認
  - KeyError → image_idの取得方法確認

#### 手順 3-2: 3Dポイント→カメラ座標変換の実装
- [ ] 🖐 **操作**: 各covisibleポイントをカメラ座標系へ変換:
  ```python
  # ワールド座標系の3Dポイント
  p3D_world = points3D[point_id].xyz  # [3]

  # カメラ座標系へ変換 (T_world2cam0 @ p3D_world)
  # T_world2cam = Pose(R=cam.rotation_matrix(), t=cam.translation())
  p3D_cam0 = T_world2cam0.transform(torch.tensor(p3D_world))
  depth0 = p3D_cam0[2]
  ```
- [ ] 🔎 **確認**: depth値が計算できる
- [ ] 🧪 **テスト**: print文でdepth値の範囲確認
- [ ] 🛠 **エラー時対処**:
  - 負の深度 → カメラの後ろのポイントをフィルタ
  - 変換エラー → Poseクラスのメソッド確認

#### 手順 3-3: ピクセル座標への投影実装
- [ ] 🖐 **操作**: 3Dポイントをピクセル座標へ投影:
  ```python
  # 正規化座標
  p_normalized = p3D_cam0[:2] / depth0  # [2]

  # ピクセル座標 (K @ [x/z, y/z, 1]^T)
  pixel = K0 @ torch.cat([p_normalized, torch.ones(1)])  # [3]
  pixel_uv = pixel[:2]  # [2]
  ```
- [ ] 🔎 **確認**: pixel座標が画像範囲内にある
- [ ] 🧪 **テスト**: pixel座標の範囲確認（0～width, 0～height）
- [ ] 🛠 **エラー時対処**:
  - 画像外のポイント → 範囲外をフィルタ
  - 投影エラー → K行列の形状確認

#### 手順 3-4: キーポイント位置での深度サンプリング実装
- [ ] 🖐 **操作**: キーポイント位置に最も近い3Dポイントの深度を割り当て:
  ```python
  keypoints0 = data["keypoints0"]  # [N, 2]
  depth_keypoints0 = torch.zeros(len(keypoints0), 1)
  valid_depth_keypoints0 = torch.zeros(len(keypoints0), 1, dtype=torch.bool)

  for i, kp in enumerate(keypoints0):
      # kpに最も近いpixel_uvを持つ3Dポイントを探す
      # 距離が閾値以内なら深度を割り当て
      distances = torch.norm(projected_pixels - kp.unsqueeze(0), dim=-1)
      min_idx = torch.argmin(distances)
      if distances[min_idx] < threshold:  # 例: 5 pixels
          depth_keypoints0[i] = depths[min_idx]
          valid_depth_keypoints0[i] = True
  ```
- [ ] 🔎 **確認**: depth_keypoints0, valid_depth_keypoints0が生成される
- [ ] 🧪 **テスト**: print文で有効なポイント数を確認
- [ ] 🛠 **エラー時対処**:
  - 有効ポイントが0 → thresholdを広げる
  - パフォーマンス問題 → ベクトル化実装に最適化

#### 手順 3-5: view1でも同様の処理を実装
- [ ] 🖐 **操作**: view0と同様にview1でもdepth_keypoints1, valid_depth_keypoints1を生成
- [ ] 🔎 **確認**: 両ビューで深度が生成される
- [ ] 🧪 **テスト**: print文で両ビューの有効ポイント数確認
- [ ] 🛠 **エラー時対処**: view0の実装をリファクタして共通化

#### 手順 3-6: データへの追加
- [ ] 🖐 **操作**: 生成した深度データをdataに追加:
  ```python
  data["depth_keypoints0"] = depth_keypoints0
  data["valid_depth_keypoints0"] = valid_depth_keypoints0
  data["depth_keypoints1"] = depth_keypoints1
  data["valid_depth_keypoints1"] = valid_depth_keypoints1
  return data
  ```
- [ ] 🔎 **確認**: dataに4つの新キーが追加される
- [ ] 🧪 **テスト**: `test_sparse_depth_generation`が成功する
- [ ] 🛠 **エラー時対処**: KeyError → キー名確認

#### 手順 3-7: 全テストの実行
- [ ] 🖐 **操作**: `uv run pytest tests/test_komainu_colmap.py -v`
- [ ] 🔎 **確認**: 全テスト（9/9）がPASSED
- [ ] 🧪 **テスト**: test suite complete
- [ ] 🛠 **エラー時対処**:
  - 既存テスト失敗 → 回帰バグ、実装見直し
  - 新テスト失敗 → 実装の修正

---

### フェーズ 4: 設定ファイル更新 (見積: 0.2h)

#### 手順 4-1: depth_matcherへの切り替え
- [ ] 🖐 **操作**: `gluefactory/configs/komainu_train_homography.yaml`を編集
  - ground_truth: `matcher/homography_matcher` → `matcher/depth_matcher`
  - ファイル名変更検討: `komainu_train_homography.yaml` → `komainu_train_depth.yaml`
- [ ] 🔎 **確認**: YAML構文が有効
- [ ] 🧪 **テスト**: `uv run python -c "import yaml; yaml.safe_load(open('...'))"` → 成功
- [ ] 🛠 **エラー時対処**: YAMLSyntaxError → インデント確認

#### 手順 4-2: コメントの更新
- [ ] 🖐 **操作**: 設定ファイルの冒頭コメントを更新:
  ```yaml
  # Training configuration for komainu_colmap dataset
  # Uses sparse depth from COLMAP 3D points for ground truth
  # depth_matcher provides more accurate matching than homography
  ```
- [ ] 🔎 **確認**: コメントが明確
- [ ] 🧪 **テスト**: visual inspection
- [ ] 🛠 **エラー時対処**: N/A

---

### フェーズ 5: 訓練実行テスト (見積: 0.3h)

#### 手順 5-1: ドライランでの動作確認
- [ ] 🖐 **操作**: `uv run python -m gluefactory.train komainu_train_depth --conf komainu_train_depth train.epochs=0 --overwrite`
- [ ] 🔎 **確認**:
  - エラーなく起動
  - depth_matcherが正常に動作
  - num_matchableが改善（目標: >10）
  - 全119バッチが処理完了
- [ ] 🧪 **テスト**: dry_run_with_depth_test → 成功
- [ ] 🛠 **エラー時対処**:
  - KeyError (depth_keypoints) → キー名がdepth_matcherの期待と一致するか確認
  - RuntimeError → 深度値の形状・型確認

#### 手順 5-2: メトリクスの確認
- [ ] 🖐 **操作**: dry run出力からメトリクスを確認:
  - num_matchable: 前回1.8 → 改善値?
  - num_unmatchable: 前回510.1 → 変化?
  - match_recall: 前回0.0 → 改善?
- [ ] 🔎 **確認**: num_matchableが10以上に改善
- [ ] 🧪 **テスト**: metrics_improved
- [ ] 🛠 **エラー時対処**:
  - 改善なし → 深度サンプリングの閾値調整
  - 悪化 → 実装バグの確認

---

### フェーズ 6: コミット・文書化 (見積: 0.3h)

#### 手順 6-1: コードのコミット
- [ ] 🖐 **操作**: `git add gluefactory/datasets/komainu_colmap.py tests/test_komainu_colmap.py gluefactory/configs/komainu_train_depth.yaml`
- [ ] 🔎 **確認**: 変更がステージングされる
- [ ] 🧪 **テスト**: `git status` → Changes to be committed
- [ ] 🛠 **エラー時対処**: N/A

#### 手順 6-2: コミットメッセージ作成
- [ ] 🖐 **操作**: 詳細なコミットメッセージを作成:
  ```
  Implement sparse depth from COLMAP 3D points (TDD)

  TDD実装フロー:
  1. Red Phase: test_sparse_depth_generation追加、失敗確認
  2. Green Phase: __getitem__にsparse depth生成実装、テスト通過
  3. 全テスト確認: 9/9通過

  実装内容:
  - Covisible 3D pointsを取得
  - カメラ座標系へ変換して深度計算
  - ピクセル座標へ投影
  - キーポイント位置での深度サンプリング
  - depth_keypoints0/1, valid_depth_keypoints0/1を生成

  設定変更:
  - homography_matcher → depth_matcher
  - komainu_train_homography.yaml → komainu_train_depth.yaml

  結果:
  - num_matchable改善: 1.8 → XX.X
  - 全9テスト通過
  ```
- [ ] 🔎 **確認**: メッセージが明確
- [ ] 🧪 **テスト**: visual inspection
- [ ] 🛠 **エラー時対処**: N/A

#### 手順 6-3: git notesの追加
- [ ] 🖐 **操作**: `git notes add`で技術詳細を記録
  - 3D→2D投影の数式
  - サンプリング閾値の根拠
  - パフォーマンス考察
- [ ] 🔎 **確認**: notesが追加される
- [ ] 🧪 **テスト**: `git notes show` → 内容表示
- [ ] 🛠 **エラー時対処**: N/A

#### 手順 6-4: Push
- [ ] 🖐 **操作**: `git push -u origin claude/update-readme-docs-011CV1sZs4vLP9sAsAvVsCmw`
- [ ] 🔎 **確認**: pushが成功
- [ ] 🧪 **テスト**: リモートブランチ更新確認
- [ ] 🛠 **エラー時対処**:
  - ネットワークエラー → 最大4回リトライ（2s, 4s, 8s, 16s間隔）

---

## 5. 作業チェックリスト（サマリ）

### フェーズ 1: 3Dポイント取得の理解・検証
- [ ] COLMAP 3Dポイント構造確認
- [ ] Covisibleポイント確認
- [ ] カメラパラメータ・ポーズ確認

### フェーズ 2: テスト作成（TDD - Red Phase）
- [ ] Sparse Depth生成テスト追加
- [ ] Depth値一貫性テスト追加

### フェーズ 3: 実装（TDD - Green Phase）
- [ ] Covisible 3Dポイント取得
- [ ] 3Dポイント→カメラ座標変換
- [ ] ピクセル座標への投影
- [ ] キーポイント位置での深度サンプリング
- [ ] view1でも同様の処理
- [ ] データへの追加
- [ ] 全テストPASS (9/9)

### フェーズ 4: 設定ファイル更新
- [ ] depth_matcherへの切り替え
- [ ] コメント更新

### フェーズ 5: 訓練実行テスト
- [ ] ドライラン動作確認
- [ ] メトリクス改善確認

### フェーズ 6: コミット・文書化
- [ ] コードのコミット
- [ ] コミットメッセージ作成
- [ ] git notes追加
- [ ] Push

---

## 6. 完了の定義

- [ ] 観点1: 全テストケース（9/9）がPASSする
- [ ] 観点2: depth_matcherでdry runが正常完了する
- [ ] 観点3: num_matchableが10以上に改善する
- [ ] 観点4: コードがDRY/KISS/SOLID原則に従っている
- [ ] 観点5: 変更がコミット・pushされている

---

## 7. 注意事項

### DRY/KISS/SOLID原則
- **DRY**: view0とview1の処理を共通化（ヘルパー関数化）
- **KISS**: シンプルな最近傍探索（最適化は後回し）
- **SOLID**: 単一責任原則（深度生成ロジックを明確に分離）

### t-wada TDD
- **Red**: まずテストを書き、失敗を確認
- **Green**: 最小限の実装でテストを通す
- **Refactor**: 動作確認後にコード品質向上

### 暗黙的fallback禁止
- ❌ 深度が取得できない場合に0を返す → NG
- ✅ valid_maskで明示的に無効をマーク → OK
- ❌ エラーを無視して続行 → NG
- ✅ エラーを明示的にログ出力・例外送出 → OK

---

## 8. 作業記録

**重要な注意事項（必ず残すこと）：**

* 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` で現在時刻を取得し、**正確な日時を記録**してください。
* 各作業項目の **開始時刻・完了時刻** を記録してください。
* 作業内容は、**実行したコマンドや具体的な操作手順** を含めて記載してください。
* 結果・備考欄には、**成功／失敗、エラーメッセージ、解決方法、重要な気づき** を必ず記入してください。
* 複数フェーズがある場合、**フェーズごとに開始・完了** を記録してください。
* コード変更を行った場合、**変更したファイル名と内容の概要** を記録してください。
* エラーが発生した場合、**エラーメッセージ全文と、その原因・対処内容** をできるだけ具体的に記録してください。

この注意事項ブロックは、テンプレートとして常に残して利用してください。

### 作業ログ表

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2025-11-11 | 11:08:13 UTC | Claude | 作業計画書作成開始 | Sparse Depth実装用の詳細チェックリスト作成 |
| 2025-11-11 | 11:08:30 UTC | Claude | フェーズ1手順1-1: COLMAP 3Dポイント構造確認 | ✅成功：10,771ポイント、xyz座標・color・error・track要素にアクセス可能を確認 |
| 2025-11-11 | 11:09:00 UTC | Claude | Track要素アクセス確認 | ✅成功：track.elements[i].image_id, point2D_idxで観測情報取得可能 |
| 2025-11-11 | 11:09:30 UTC | Claude | Image object確認 | ✅成功：points2D[i].has_point3D()で3D対応確認、point3D_id取得可能 |
| 2025-11-11 | 11:10:00 UTC | Claude | Covisibleポイント抽出確認 | ✅成功：2画像間で50個のcovisibleポイント確認、set演算で効率的に抽出可能 |
| 2025-11-11 | 11:11:00 UTC | Claude | フェーズ1手順1-2: pairs_from_colmap.py確認 | ✅成功：extract_covisible_pairs()の実装を理解、__getitem__内でset演算による直接抽出が必要と判断 |
| 2025-11-11 | 11:12:00 UTC | Claude | フェーズ1手順1-3: カメラパラメータ確認 | ✅成功：既存__getitem__でK0, K1, T_0to1が利用可能を確認 |
| 2025-11-11 | 11:13:24 UTC | Claude | フェーズ1完了 | ✅フェーズ1全手順完了、3Dポイント・covisible・カメラパラメータへのアクセス方法を理解 |
| 2025-11-11 | 11:15:00 UTC | Claude | フェーズ2手順2-1: Sparse Depthテスト追加 | ✅成功：test_sparse_depth_generation()を追加、depth_keypoints0/1, valid_depth_keypoints0/1のテスト実装 |
| 2025-11-11 | 11:16:43 UTC | Claude | Red Phase確認テスト実行 | ✅成功：期待通り失敗（depth_keypoints0 key should be present）、Red Phase完了 |
| 2025-11-11 | 11:17:00 UTC | Claude | フェーズ3開始：実装準備 | データ構造確認、画像名→ID変換方法の調査 |
| 2025-11-11 | 11:18:00 UTC | Claude | 親クラス調査 | pairs_from_colmap.pyでreconstructionの取り扱いを確認 |
| 2025-11-11 | 11:19:00 UTC | Claude | _initメソッド拡張 | self.reconstructions辞書を追加、各シーンのCOLMAP reconstructionをキャッシュ |
| 2025-11-11 | 11:20:00 UTC | Claude | _generate_sparse_depthヘルパー関数実装 | 3Dポイント→カメラ座標変換→ピクセル投影→キーポイント深度サンプリングを実装（103行） |
| 2025-11-11 | 11:21:00 UTC | Claude | __getitem__拡張 | sparse depth生成ロジックを追加、depth_keypoints0/1, valid_depth_keypoints0/1を生成 |
| 2025-11-11 | 11:22:00 UTC | Claude | T_w2camアクセス修正 | data["view0"]["T_w2cam"]として正しいキーでPoseオブジェクトを取得 |
| 2025-11-11 | 11:23:00 UTC | Claude | テスト修正・デバッグ | keypointsアクセス方法を修正、extractorなしケースの対応検討 |
| 2025-11-11 | 11:24:00 UTC | Claude | テスト修正完了 | keypointsなしケースも適切に処理、全9テスト通過 |
| 2025-11-11 | 11:26:00 UTC | Claude | komainu_train_depth.yaml作成 | depth_matcher使用の設定ファイル作成 |
| 2025-11-11 | 11:27:00 UTC | Claude | Dry run実行・デバッグ | depth_matcherテスト、keypoints0/1をトップレベルに追加 |
| 2025-11-11 | 11:30:00 UTC | Claude | 実装完了・テスト全通過確認 | 全9テスト通過、実装完了 |

| 2025-11-11 | 11:35:00 UTC | Claude | ONBOARDING.md日本語翻訳 | ✅成功：オンボーディング資料を日本語化、commit e8ddeca & push完了 |
| 2025-11-11 | 11:40:00 UTC | Claude | Dry run検証（komainu_train_depth） | ❌失敗：RuntimeError in depth.py:71, テンソルサイズ不一致エラー |

---

**作業開始時刻:** 2025-11-11 11:08:13 UTC+0000
**現在時刻:** 2025-11-11 11:44:58 UTC+0000
**作業ステータス:** ⚠️ 実装完了・テスト通過、但し訓練パイプラインでエラー検出（要修正）

---

## 5. 検出された問題と詳細分析

### 問題 #1: Sparse Depth使用時の訓練パイプラインエラー

**発生日時:** 2025-11-11 11:26:00 UTC (Dry run実行時)

**症状:**
```
RuntimeError: The size of tensor a (2) must match the size of tensor b (0) at non-singleton dimension 1
```

**エラー発生箇所:**
```
File "/home/user/glue-factory/gluefactory/geometry/depth.py", line 71, in project
    kpi_3d_i = kpi_3d_i * di[..., None]
```

**完全なスタックトレース:**
```
File "/home/user/glue-factory/gluefactory/train.py", line 212, in <module>
    main_worker(0, conf, output_dir, args)
File "/home/user/glue-factory/gluefactory/train.py", line 108, in main_worker
    res = trainer.launch_training(output_dir, conf, device)
File "/home/user/glue-factory/gluefactory/trainer.py", line 1125, in launch_training
    trainer.train_loop(output_dir, dataset)
File "/home/user/glue-factory/gluefactory/trainer.py", line 1056, in train_loop
    self.run_eval(output_dir, dataset, writer)
File "/home/user/glue-factory/gluefactory/trainer.py", line 989, in run_eval
    eval_results = self.eval_loop(output_dir, eval_loader, max_iters=max_iters)
File "/home/user/glue-factory/gluefactory/trainer.py", line 912, in eval_loop
    results, pr_metrics, figures = run_evaluation(...)
File "/home/user/glue-factory/gluefactory/trainer.py", line 91, in run_evaluation
    losses, metrics = model.loss_metrics(pred, data)
File "/home/user/glue-factory/gluefactory/models/base_model.py", line 149, in loss_metrics
    return self.loss(pred, data)
File "/home/user/glue-factory/gluefactory/models/two_view_pipeline.py", line 118, in loss
    gt_pred = self.ground_truth({**data, **pred})
File "/home/user/glue-factory/gluefactory/models/matchers/depth_matcher.py", line 41, in _forward
    return self.match_with_depth(data)
File "/home/user/glue-factory/gluefactory/models/matchers/depth_matcher.py", line 56, in match_with_depth
    result = gt_generation.gt_matches_from_pose_depth(...)
File "/home/user/glue-factory/gluefactory/geometry/gt_generation.py", line 45, in gt_matches_from_pose_depth
    kp0_1, visible0, unmatchable0 = depth.project(kp0, d0, depth1, camera0, camera1, T_0to1, ccth=cc_th)
File "/home/user/glue-factory/gluefactory/geometry/depth.py", line 71, in project
    kpi_3d_i = kpi_3d_i * di[..., None]
RuntimeError: The size of tensor a (2) must match the size of tensor b (0) at non-singleton dimension 1
```

**根本原因の分析:**

1. **Dense Depth前提の実装**
   - `gluefactory/geometry/depth.py:71`のコードは、全てのキーポイントに対してdepth値が存在することを前提としている
   - `di` (depth_keypoints0)のテンソル形状が期待と異なる

2. **Sparse Depthの特性**
   - komainu_colmapのSparse Depthは、COLMAP 3Dポイントから生成される
   - 全てのキーポイントがCOLMAP 3Dポイントに対応するわけではない（5px閾値で最近傍探索）
   - `valid_depth_keypoints0/1`がboolマスクで有効なポイントを示している
   - 有効でないキーポイントはdepth=0.0で、valid_mask=Falseとなっている

3. **データフロー不一致**
   - `gt_generation.gt_matches_from_pose_depth()`は`depth_keypoints0/1`と`valid_depth_keypoints0/1`を受け取る
   - その後`depth.project(kp0, d0, depth1, ...)`を呼び出す
   - `d0`は`depth_keypoints0`から取得されるが、これは`valid_mask`でフィルタされていない全キーポイント分のテンソル
   - しかし`depth.py:71`は全てのdepthが有効であることを前提に処理する

4. **テンソルサイズ不一致の詳細**
   - `kpi_3d_i` = [batch_size, num_keypoints, 3] （3D座標）
   - `di` = [batch_size, num_valid_keypoints, 1] （有効なdepthのみ？）← ここが問題
   - 実際には`di`は[batch_size, num_keypoints, 1]であるべきだが、何らかの理由でサイズが異なる
   - エラーメッセージ "size of tensor a (2)" は恐らくnum_keypoints=2
   - "size of tensor b (0)" は有効なdepthが0個

**推定される具体的シナリオ:**
- あるバッチサンプルで、SuperPointが2個のkeypointsを抽出
- しかし、その2個のkeypointsのどちらもCOLMAP 3Dポイントから5px以内に対応点が見つからない
- その結果、`depth_keypoints0` = [2, 1] (全てゼロ), `valid_depth_keypoints0` = [2, 1] (全てFalse)
- `gt_matches_from_pose_depth()`が`d0 = depth_keypoints0`をそのまま使おうとする
- しかし、どこかで`valid_depth_keypoints0`でフィルタリングが行われ、`di`が空テンソル[0, 1]になる
- `kpi_3d_i`は元のkeypoints数[2, 3]なので、サイズ不一致エラー

**影響範囲:**
- `komainu_train_depth.yaml`を使用した訓練が実行できない
- テストスイート（`test_komainu_colmap.py`）は通過（データセットレベルの実装は正しい）
- 訓練パイプライン（depth_matcherとgt_generation）の実装がsparse depthに対応していない

**修正方針（検討中）:**

**方針A: Dense Depthマップを生成する**
- COLMAP 3Dポイントから補間してdense depth mapを生成
- Pros: 既存のdepth_matcher/gt_generationコードを変更不要
- Cons: 補間が不正確、計算コスト高、本質的にはsparseなデータをdenseに変換するのは情報の追加

**方針B: gt_generationをsparse depth対応に修正する**
- `gt_matches_from_pose_depth()`と`depth.project()`をsparse depth対応に修正
- valid_maskを考慮して、有効なkeypointsのみで処理
- Pros: 本質的な解決、sparse dataの扱いとして正しい
- Cons: 既存コード修正が必要、他のデータセット（MegaDepth等）への影響を確認必要

**方針C: Homography Matcherを使用する**
- `komainu_train_homography.yaml`を使用する（既に動作確認済み）
- Sparse Depthは将来の改善として保留
- Pros: 即座に訓練開始可能
- Cons: num_matchableが低い（1.8 pts/pair）、平面仮定の制約

**次のステップ:**
1. MegaDepthなど他のデータセットでdepth_matcherがどのようにdepthを扱っているか調査
2. Dense depth mapの生成方法を調査（既存実装があるか？）
3. 修正方針を決定し、実装
4. ユーザーに状況報告と方針相談

**関連ファイル:**
- `gluefactory/geometry/depth.py` (depth.project関数)
- `gluefactory/geometry/gt_generation.py` (gt_matches_from_pose_depth関数)
- `gluefactory/models/matchers/depth_matcher.py` (DepthMatcher)
- `gluefactory/datasets/komainu_colmap.py` (sparse depth生成実装)
- `gluefactory/configs/komainu_train_depth.yaml` (設定ファイル)

---

### Dense Depthマップの追加に関する設計アイデア

**仮にDense Depth (int16のmmスケール深度画像) を用意する場合のデータ構造案:**

#### 案1: 標準的なディレクトリ構造（MegaDepth方式）

```
data/komainu_colmap/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── depth/                    # 新規追加
│   ├── image_001.png         # int16, mmスケール, 画像と同じ解像度
│   ├── image_002.png
│   └── ...
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── views.txt
└── covisibility/
```

**メリット:**
- 既存MegaDepthデータセットと同じ構造で互換性が高い
- `gluefactory/datasets/`の既存実装を参考にできる
- ファイル名マッチングが容易（拡張子のみ変更）

**実装方法:**
```python
# komainu_colmap.pyの__getitem__内
depth_dir = self.root / "depth"
depth_path = depth_dir / f"{image_name.stem}.png"

if depth_path.exists():
    # int16 depth画像を読み込み
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    # mmスケールをメートルスケールに変換
    depth_m = depth_mm.astype(np.float32) / 1000.0
    # 0値（無効深度）を処理
    depth_m[depth_m == 0] = np.nan
    data["view0"]["depth"] = torch.from_numpy(depth_m)
```

#### 案2: COLMAPディレクトリ内に配置

```
data/komainu_colmap/
├── images/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   ├── points3D.bin
│   └── depth/              # sparse/0内に配置
│       ├── image_001.png
│       └── ...
├── views.txt
└── covisibility/
```

**メリット:**
- COLMAPの再構成結果と密接に関連することが明示的
- 複数のsparse再構成（sparse/0, sparse/1等）それぞれにdepthを持てる

**デメリット:**
- 標準的ではない、独自構造

#### 案3: メタデータファイルで柔軟に管理

```
data/komainu_colmap/
├── images/
├── depth_maps/             # 任意のディレクトリ名
│   └── ...
├── sparse/0/
├── depth_config.yaml       # 新規追加
├── views.txt
└── covisibility/
```

**depth_config.yaml:**
```yaml
depth_format: png           # ファイル形式
depth_scale: 1000.0         # mmスケール→メートル変換係数
depth_dir: depth_maps       # 深度マップディレクトリ
depth_invalid_value: 0      # 無効深度の値
file_mapping:               # 画像名→深度ファイル名のマッピング（省略可能）
  image_001.jpg: depth_001.png
  image_002.jpg: depth_002.png
```

**メリット:**
- 柔軟性が高い（スケール、形式、ディレクトリ位置を設定で変更可能）
- 複数の深度ソース（COLMAP sparse, dense reconstruction, センサー深度等）を切り替え可能

**デメリット:**
- 実装が複雑、追加のYAML解析が必要

#### 案4: npzアーカイブで一括管理

```
data/komainu_colmap/
├── images/
├── sparse/0/
├── depth_maps.npz          # 全画像の深度をnpzで一括保存
├── views.txt
└── covisibility/
```

**depth_maps.npz構造:**
```python
np.savez_compressed(
    "depth_maps.npz",
    image_001=depth_array_1,  # [H, W], float32, メートル単位
    image_002=depth_array_2,
    # ...
    metadata={"scale": "meters", "invalid_value": np.nan}
)
```

**メリット:**
- ファイル数が少ない（管理が容易）
- 圧縮により容量削減
- メタデータを同じファイルに含められる

**デメリット:**
- 一部の画像のみ更新する場合に不便
- メモリに全部ロードする必要がある可能性

#### 推奨案: **案1 (MegaDepth方式)**

**理由:**
1. **互換性:** MegaDepthなど既存データセットと同じ構造
2. **シンプル:** 実装が最も簡潔で理解しやすい
3. **保守性:** ファイル単位で管理、追加・削除が容易
4. **既存実装活用:** `gluefactory/datasets/megadepth.py`を参考にできる

**具体的なファイル配置:**
```
data/komainu_colmap/
├── images/
│   ├── DSC_0001.jpg        # オリジナル画像 (例: 4000x3000)
│   ├── DSC_0002.jpg
│   └── ...
├── depth/
│   ├── DSC_0001.png        # int16 PNG, mmスケール, 同解像度 (4000x3000)
│   ├── DSC_0002.png        # 値0 = 無効深度, 値1000 = 1.0m
│   └── ...
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── ...
```

**Dense Depth画像の仕様:**
- **フォーマット:** 16bit PNG (lossless)
- **スケール:** mmスケール（値1000 = 1.0メートル）
- **解像度:** 元画像と同じ解像度
- **無効値:** 0 (深度情報がない箇所)
- **値範囲:** 0-65535 (0m - 65.535m)
- **ファイル名:** 画像ファイル名の拡張子を.pngに変更

**Dense Depth生成方法（参考）:**
- **COLMAP dense reconstruction:** `colmap image_undistorter` + `colmap patch_match_stereo` → depth maps
- **外部センサー:** RGB-Dカメラ、LiDARスキャナー
- **深度推定モデル:** MiDaS, DPT, ZoeDepthなど
- **ステレオマッチング:** OpenCV StereoSGBM等

**データセット実装の修正（komainu_colmap.py）:**
```python
def __getitem__(self, idx):
    data = super().__getitem__(idx)

    # Dense depthマップの読み込み（存在する場合）
    depth_dir = self.root / "depth"
    if depth_dir.exists():
        name0, name1 = data["name"].split("/")

        # view0のdepth
        depth_path0 = depth_dir / f"{Path(name0).stem}.png"
        if depth_path0.exists():
            depth0_mm = cv2.imread(str(depth_path0), cv2.IMREAD_ANYDEPTH)
            depth0_m = depth0_mm.astype(np.float32) / 1000.0
            depth0_m[depth0_mm == 0] = np.nan  # 無効値をNaNに
            data["view0"]["depth"] = torch.from_numpy(depth0_m)

        # view1のdepth（同様）
        depth_path1 = depth_dir / f"{Path(name1).stem}.png"
        if depth_path1.exists():
            depth1_mm = cv2.imread(str(depth_path1), cv2.IMREAD_ANYDEPTH)
            depth1_m = depth1_mm.astype(np.float32) / 1000.0
            depth1_m[depth1_mm == 0] = np.nan
            data["view1"]["depth"] = torch.from_numpy(depth1_m)

    # Sparse depthは depth/ ディレクトリが存在しない場合のフォールバックとして残す
    # または、dense depthが存在する場合はsparse depth生成をスキップする

    return data
```

**設定ファイル（komainu_colmap.yaml）への追加:**
```yaml
name: komainu_colmap
root: komainu_colmap
# ...
depth_dir: depth              # Dense depthマップのディレクトリ (optional)
depth_scale: 1000.0           # mmスケール変換係数
use_sparse_depth: false       # false = dense depth優先, true = sparse depth生成
```

**メモリ・性能面の考慮:**
- Dense depth画像（例: 4000x3000 int16）は約23MB/枚
- 30画像で約690MB
- 訓練時にリサイズ（例: 640px）されるため、メモリ使用量は大幅に削減される
- COLMAPのdense reconstructionから生成する場合、前処理として一度実行しておく

---

**記録者:** Claude
**最終更新:** 2025-11-11 11:44:58 UTC+0000
