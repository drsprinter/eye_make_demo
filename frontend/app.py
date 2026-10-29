"""Streamlit frontend. API server required on port 8000."""
import os
import requests
import streamlit as st

API = os.environ.get('EYEMAKE_API_URL', 'http://127.0.0.1:8000').rstrip('/')
st.set_page_config(page_title='Eye Makeup Preference Lab', page_icon='👁️', layout='wide')
st.title('👁️ Eye Makeup Preference Lab')
st.caption('CLIP事前計算済み特徴量を用いた美的選好探索デモ — FastAPI + Streamlit')


def api(method, path, **kwargs):
    try:
        r = requests.request(method, API + path, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f'バックエンド接続エラー: {exc}')
        st.stop()


health = api('GET', '/api/health')
if health['missing_images']:
    st.warning(f"元画像が {health['missing_images']} 枚不足しています。Eyemake4/ に配置してください。")
if 'session_id' not in st.session_state:
    st.session_state.session_id = api('POST', '/api/sessions')['session_id']
sid = st.session_state.session_id
info = api('GET', f'/api/sessions/{sid}')

with st.sidebar:
    st.header('セッション')
    st.metric('評価済みペア', info['count'])
    st.caption(f"画像 {health['images']} 枚 / 特徴量 {health['dimensions']} 次元 / {health['model']}")
    if st.button('新しいセッションを開始', use_container_width=True):
        st.session_state.session_id = api('POST', '/api/sessions')['session_id']
        st.rerun()
    st.download_button('評価履歴をCSV保存', data=requests.get(API + f'/api/sessions/{sid}/history.csv', timeout=30).content,
                       file_name='eye_makeup_votes.csv', mime='text/csv', use_container_width=True)
    st.caption('セッション情報はバックエンドのメモリ上に保持します。サーバーを再起動すると消えます。')

st.subheader('どちらのアイメイクが好き？')
pair = info['pair']
a, b = st.columns(2)
with a:
    st.image(API + pair['left']['url'], caption='A — ' + pair['left']['name'], use_container_width=True)
with b:
    st.image(API + pair['right']['url'], caption='B — ' + pair['right']['name'], use_container_width=True)

choices = [('Aがとても好き','left_strong'), ('Aが少し好き','left_slight'), ('Bが少し好き','right_slight'), ('Bがとても好き','right_strong'), ('スキップ','skip')]
for col, (label, choice) in zip(st.columns(5), choices):
    if col.button(label, key=choice, use_container_width=True):
        api('POST', f'/api/sessions/{sid}/votes', json={'choice': choice})
        st.rerun()

st.divider()
st.subheader('好みに近いアイメイク TOP 12')
rec = api('GET', f'/api/sessions/{sid}/recommendations?limit=12')
if not rec['trained']:
    st.info('まず画像ペアを評価してください。')
else:
    items = rec['items']
    for start in range(0, len(items), 4):
        for col, item in zip(st.columns(4), items[start:start+4]):
            with col:
                st.image(API + item['url'], use_container_width=True)
                st.caption(f"{item['name']} / スコア {item['score']:.3f}")
st.caption('スコアはCLIP空間における類似度であり、「好き」である確率ではありません。')
