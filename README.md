# Eye Makeup Preference Lab — 2-stage Render Edition

CLIPはアプリ上では実行しません。`Eyemake4/clip_embeddings.npz` の `paths` と `embeddings` を読み、同じフォルダの画像を表示します。

## 構成

- `backend/main.py`: FastAPI / API / 画像の配信
- `backend/preference.py`: 20問の4択比較と、10枚の5段階評価
- `web/`: 単一URLで動くHTML/CSS/JSフロント
- `Eyemake4/`: ユーザーが用意するNPZと実画像（同梱なし）

## 起動

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000/`。

Render の Start Command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Root Directory: 空欄（リポジトリ直下）。Build Command: `pip install -r requirements.txt`。

## アルゴリズム（試作版）

1. STEP 1: 20組、左右・両方好き・両方好みではないの4択。左右選択では選択画像を正例、非選択画像を負例。両方好きでは各画像を0.5ずつ正例、両方好みでない場合は0.5ずつ負例として蓄積し、`normalize(positive - negative)` を長期選好とする。計20問で確定して凍結。
2. STEP 2: 長期ベクトルとのコサイン類似度が高い未表示画像から10枚を提示（上位40枚の範囲で分散）。1〜5の評価を `(rating - 3) / 2` で重み付けして加算し、正規化して補正ベクトルとする。3は中立。
3. 最終ランキング: `normalize(0.7 * long_vector + 0.3 * short_vector)` との類似度順にTOP10を表示。1〜2と回答した画像は除外。

※ `70:30`、10枚、ペア選択方針はデモ用の暫定設定。研究に利用する前に固定・事前登録を推奨します。評価・推奨は確率的推定であり、適合度の保証ではありません。20問では個人の長期的な嗜好の安定性は実証できず、正しくは「20問から推定した長期選好ベクトル」です。

## 注意

- 画像はNPZに含まれません。実画像を `Eyemake4/` に保管してください。
- サーバーのメモリにセッションを保持。再起動で消え、複数ワーカーを使うとセッションが見えなくなるためデモは1ワーカー運用。
- 履歴CSVをダウンロードできます。個人情報を本番収集するなら永続DBと同意・アクセス管理が必要。
