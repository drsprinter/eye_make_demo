# Eye Makeup Preference Lab — Render一体型版

FastAPIが `/` でHTML/CSS/JavaScript画面を配信します。StreamlitとCLIPの実行環境は不要です。

## データ配置

```
eye_makeup_render_web/
├── backend/main.py, backend/preference.py
├── web/index.html, web/app.js, web/style.css
├── Eyemake4/
│   ├── clip_embeddings.npz
│   └── （NPZのpathsに含まれる画像300枚）
├── requirements.txt
└── render.yaml
```

**注意:** NPZには画像本体は含まれていません。`Eyemake4/` に元画像も必要です。`paths` 配列内のファイル名・相対パスと一致するように置いてください。画像とNPZは配布ZIPに含まれていません。

## ローカル実行

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

- ブラウザ: http://localhost:8000/
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## Render

- Root Directory: 空欄（リポジトリのルートがこの構成の場合）
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- GitHubに `web/`, `backend/`, `Eyemake4/` のデータ、`requirements.txt` をコミットし、再デプロイしてください。
- 公開URLの `/` がデモ画面になります。

API: セッション作成、2画像比較、4段階投票、スキップ、推薦、CSV出力、画像配信。学習履歴はメモリ上に保持され、サーバー再起動で消去されます。Renderの複数ワーカー/インスタンス構成ではセッション共有されません。個人情報・識別情報は送信しないでください。

既存の `frontend/` は旧Streamlit版の参考ファイルです。新しい公開フロントは `web/` です。
