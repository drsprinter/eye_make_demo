"""Run: streamlit run app.py. Uses previously computed Eyemake4/clip_embeddings.npz."""
from pathlib import Path
import csv
import io

import numpy as np
import streamlit as st
from preference import PreferenceSession

st.set_page_config(page_title="Eye Makeup Preference Lab", page_icon="👁️", layout="wide")
ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "Eyemake4"
NPZ_FILE = ASSET_DIR / "clip_embeddings.npz"


@st.cache_data(show_spinner="CLIP特徴量を読み込んでいます…")
def load_embeddings(file_path, mtime):
    with np.load(file_path, allow_pickle=False) as npz:
        if "embeddings" not in npz or "paths" not in npz:
            raise ValueError("NPZには embeddings と paths が必要です。")
        vectors = np.asarray(npz["embeddings"], dtype=np.float32)
        paths = [str(s) for s in npz["paths"]]
        model = str(npz["model"].item()) if "model" in npz else "不明"
    if vectors.ndim != 2 or len(paths) != vectors.shape[0] or len(paths) < 2:
        raise ValueError("paths数とembeddingsの行数が不一致、または画像数が2枚未満です。")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("特徴量にNaNまたは無限大が含まれています。")
    base = ASSET_DIR.resolve()
    resolved = []
    for p in paths:
        candidate = (base / p).resolve()
        # Do not allow NPZ to read arbitrary paths from the computer.
        if base not in candidate.parents:
            raise ValueError("画像パスがEyemake4の外を指しています: " + p)
        resolved.append(str(candidate))
    return paths, resolved, vectors, model


st.title("👁️ Eye Makeup Preference Lab")
st.caption("CLIPの事前計算済み特徴量を使って、自分の『好き』を探索するデモ")

if not NPZ_FILE.is_file():
    st.error("`Eyemake4/clip_embeddings.npz` が見つかりません。app.pyと同じ階層にEyemake4フォルダを配置してください。")
    st.code("eye_makeup_npz_app/\n  app.py\n  preference.py\n  Eyemake4/\n    clip_embeddings.npz\n    eye_001.jpg\n    ...")
    st.stop()

try:
    names, full_paths, vectors, model = load_embeddings(str(NPZ_FILE), NPZ_FILE.stat().st_mtime_ns)
except Exception as exc:
    st.error("NPZの読み込みに失敗しました: " + str(exc))
    st.stop()

missing = [p for p in full_paths if not Path(p).is_file()]
if missing:
    st.error(f"元画像が {len(missing)} 枚見つかりません。NPZには画像そのものは含まれません。")
    st.code("\n".join(missing[:12]) + ("\n..." if len(missing) > 12 else ""))
    st.stop()

if "preference_session" not in st.session_state or st.session_state.get("loaded_npz") != (str(NPZ_FILE), NPZ_FILE.stat().st_mtime_ns):
    st.session_state.preference_session = PreferenceSession(vectors)
    st.session_state.preference_session.select_pair()
    st.session_state.loaded_npz = (str(NPZ_FILE), NPZ_FILE.stat().st_mtime_ns)
session = st.session_state.preference_session

with st.sidebar:
    st.header("デモ設定")
    st.metric("評価済みペア", session.count)
    st.write(f"画像枚数: {len(names)}")
    st.caption(f"特徴量: {vectors.shape[1]}次元 / {model}")
    if st.button("最初からやり直す", use_container_width=True):
        seed = int(np.random.default_rng().integers(1, 2**31))
        st.session_state.preference_session = PreferenceSession(vectors, seed=seed)
        st.session_state.preference_session.select_pair()
        st.rerun()
    st.caption("履歴は現在のStreamlitセッション内に保持されます。必要に応じてCSVを書き出してください。")
    if session.history:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["round", "left", "right", "choice", "left_file", "right_file"])
        writer.writeheader()
        for h in session.history:
            writer.writerow({**h, "left_file": names[h["left"]], "right_file": names[h["right"]]})
        st.download_button("評価履歴をCSV保存", buf.getvalue().encode("utf-8-sig"), file_name="eye_makeup_votes.csv", mime="text/csv", use_container_width=True)

st.subheader("どちらのアイメイクが好き？")
st.write("左右の画像を比較して、好みに近い方とその強さを選んでください。")
left, right = session.last_pair
cols = st.columns(2)
with cols[0]:
    st.image(full_paths[left], caption="A", use_container_width=True)
with cols[1]:
    st.image(full_paths[right], caption="B", use_container_width=True)

choices = [
    ("A がとても好き", "left_strong"),
    ("A が少し好き", "left_slight"),
    ("B が少し好き", "right_slight"),
    ("B がとても好き", "right_strong"),
    ("スキップ", "skip"),
]
buttons = st.columns(5)
for col, (label, value) in zip(buttons, choices):
    if col.button(label, key=value, use_container_width=True):
        session.vote(value)
        session.select_pair()
        st.rerun()

st.divider()
st.subheader("今の好みに近いアイメイク TOP 12")
if not np.any(session.user_vector):
    st.info("まず画像ペアを1回以上評価してください。")
else:
    indices, scores = session.ranking(12)
    for offset in range(0, len(indices), 4):
        for col, i, score in zip(st.columns(4), indices[offset:offset + 4], scores[offset:offset + 4]):
            with col:
                st.image(full_paths[int(i)], use_container_width=True)
                st.caption(f"{names[int(i)]} — 類似度 {float(score):.3f}")
st.caption("類似度はCLIP特徴空間上のスコアであり、好きである確率ではありません。")
