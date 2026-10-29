# Eye Makeup Preference Lab — Full Stack

事前計算した CLIP 埋め込みから、2画像比較・選好学習・推薦を行うデモ。**CLIP、PyTorch、images/、事前計算スクリプトは不要です。**

## 構成

```text
eye_makeup_fullstack/
├── Eyemake4/
│   ├── clip_embeddings.npz   ← ご自身の生成済みファイル
│   └── (元画像300枚)          ← NPZのpathsで参照する元画像
├── backend/
│   ├── __init__.py
│   ├── main.py               ← FastAPI API / 画像配信 / CSV出力
│   └── preference.py         ← CLIPベクトル上の選好学習
├── frontend/
│   └── app.py                ← Streamlit画面（APIクライアント）
└── requirements.txt
```

NPZ は `embeddings`（N×512などのfloat配列）、`paths`（Eyemake4からの相対パス配列）、任意の `model` を含む形式に対応します。NPZに画像の実体は入っていないので、画像ファイルは必要です。

## 起動

Python 3.9以降を想定。プロジェクトのルートフォルダで実行：

```bash
python -m pip install -r requirements.txt
```

**ターミナル1（バックエンド）**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

- 接続確認: http://127.0.0.1:8000/api/health
- API仕様: http://127.0.0.1:8000/docs

**ターミナル2（フロントエンド）**
```bash
python -m streamlit run frontend/app.py
```

http://localhost:8501 を開きます。

バックエンドを別の場所で動かす場合、`EYEMAKE_API_URL` 環境変数を設定してください。例：`EYEMAKE_API_URL=http://127.0.0.1:8000 python -m streamlit run frontend/app.py`。

## API

| Method | Path | 説明 |
|---|---|---|
| GET | `/api/health` | 画像数・特徴量次元・不足画像数 |
| POST | `/api/sessions` | 選好学習セッション作成 |
| GET | `/api/sessions/{id}` | 現在の提示画像と履歴 |
| POST | `/api/sessions/{id}/votes` | 4択またはスキップを送信 |
| GET | `/api/sessions/{id}/recommendations?limit=12` | 推薦画像・スコア |
| GET | `/api/sessions/{id}/history.csv` | 評価履歴をCSV出力 |
| GET | `/images/{index}` | 元画像配信 |

投票 `choice` は `left_strong`, `left_slight`, `right_slight`, `right_strong`, `skip` のいずれか。

## 注意

- セッションはバックエンドの**メモリ上**に保持され、サーバー停止・再起動で失われます。必要な評価履歴はCSV出力してください。
- APIには認証がありません。**ローカル利用前提**です。インターネット公開には認証、永続データベース、利用者管理、CORS/プロキシ設定等が必要です。
- `safe/stretch/frontier=4:6:2` は既存版から引き継いだ**簡易サンプリング**です。厳密な研究実験用ロジックの実装ではありません。
- 元画像のパスはNPZ内の相対パスを解決します。`Eyemake4` 外へのアクセスは禁止しています。
