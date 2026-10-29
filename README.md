# Eye Makeup Preference Lab — NPZ読み込み専用版

既存の `Eyemake4/clip_embeddings.npz` を直接読み込みます。CLIP / PyTorch / 事前計算スクリプトは不要です。Python 3.9+ 対応。

## フォルダ構成

```
eye_makeup_npz_app/
├── app.py
├── preference.py
├── requirements.txt
└── Eyemake4/
    ├── clip_embeddings.npz
    ├── （既存のアイメイク画像300枚）
    └── ...
```

`Eyemake4` に現在の画像300枚と、生成済みの `clip_embeddings.npz` をそのままコピーしてください。画像がサブフォルダ内にあっても、NPZ内 `paths` と一致すれば表示できます。**NPZだけでは画像を表示できません**（NPZに画像本体は保存されていません）。

## インストールと起動

ターミナルで `app.py` があるフォルダに移動し、以下を実行します。

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Notebookから起動したい場合は `!streamlit run app.py` でコマンド実行できますが、別ターミナルでの実行がおすすめです。

## NPZの仕様

- `embeddings`: (画像枚数, 特徴量次元) float32、L2正規化済みを想定
- `paths`: `Eyemake4/` からの相対画像パス
- `model`: 例 `ViT-B/32`

## 学習方法

- 2枚の比較を4段階（A強/A弱/B弱/B強）で回答。スキップあり
- 選好ベクトルを差分特徴量で更新（decay=0.85、alpha=0.45）
- Safe/Stretch/Frontierの12回周期で4:6:2に相当する簡易的選択戦略
- 結果の上位12画像を表示
- CSVログ出力 / セッション再開始

現行のSafe/Stretch/Frontierは簡易ヒューリスティックであり、実験デザインを厳密に保証するものではありません。結果は研究用の暫定デモとしてください。
