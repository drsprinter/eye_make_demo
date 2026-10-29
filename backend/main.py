"""FastAPI backend: launch with uvicorn backend.main:app --reload --port 8000."""
import csv
import io
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from .preference import PreferenceSession

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (ROOT / 'Eyemake4').resolve()
NPZ_PATH = ASSETS / 'clip_embeddings.npz'
app = FastAPI(title='Eye Makeup Preference API', version='1.0.0')
lock = threading.RLock()
sessions = {}
WEB_DIR = ROOT / 'web'


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / 'index.html', media_type='text/html')


@app.get('/assets/{filename}', include_in_schema=False)
def web_asset(filename: str):
    if filename not in {'style.css', 'app.js'}:
        raise HTTPException(status_code=404, detail='Not Found')
    return FileResponse(WEB_DIR / filename)



def load_dataset():
    if not NPZ_PATH.is_file():
        raise RuntimeError(f'NPZ not found: {NPZ_PATH}')
    with np.load(NPZ_PATH, allow_pickle=False) as z:
        if not {'paths', 'embeddings'}.issubset(z.files):
            raise ValueError('NPZ needs paths and embeddings')
        names = [str(x) for x in z['paths']]
        vectors = np.asarray(z['embeddings'], dtype=np.float32)
        model = str(z['model'].item()) if 'model' in z else 'unknown'
    if vectors.ndim != 2 or vectors.shape[0] != len(names) or len(names) < 2 or not np.isfinite(vectors).all():
        raise ValueError('Invalid embeddings shape/values or paths count')
    paths = []
    for name in names:
        p = (ASSETS / name).resolve()
        if not p.is_relative_to(ASSETS) or p == ASSETS:
            raise ValueError(f'Unsafe image path: {name}')
        paths.append(p)
    return names, paths, vectors, model


class Vote(BaseModel):
    choice: str


def dataset():
    try:
        return load_dataset()
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def find_session(session_id):
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail='Session not found. Start a new session.')
    return state


def image_info(i, names):
    return {'index': int(i), 'name': names[int(i)], 'url': f'/images/{int(i)}'}


def pair_payload(state, names):
    a, b = state.last_pair
    return {'round': state.count + 1, 'left': image_info(a, names), 'right': image_info(b, names)}


@app.get('/api/health')
def health():
    names, paths, vectors, model = dataset()
    return {'status': 'ok', 'images': len(names), 'missing_images': sum(not p.is_file() for p in paths), 'dimensions': vectors.shape[1], 'model': model}


@app.post('/api/sessions')
def new_session():
    names, paths, vectors, model = dataset()
    session_id = uuid.uuid4().hex
    with lock:
        state = PreferenceSession(vectors, seed=int.from_bytes(os.urandom(4), 'little'))
        state.select_pair()
        sessions[session_id] = state
        return {'session_id': session_id, 'pair': pair_payload(state, names), 'total_images': len(names)}


@app.get('/api/sessions/{session_id}')
def session_details(session_id: str):
    names, _, _, _ = dataset()
    with lock:
        state = find_session(session_id)
        return {'session_id': session_id, 'count': state.count, 'pair': pair_payload(state, names), 'history': state.history}


@app.post('/api/sessions/{session_id}/votes')
def vote(session_id: str, payload: Vote):
    if payload.choice not in {'left_strong','left_slight','right_slight','right_strong','skip'}:
        raise HTTPException(status_code=422, detail='Invalid choice')
    names, _, _, _ = dataset()
    with lock:
        state = find_session(session_id)
        state.vote(payload.choice)
        state.select_pair()
        return {'count': state.count, 'pair': pair_payload(state, names)}


@app.get('/api/sessions/{session_id}/recommendations')
def recommendations(session_id: str, limit: int = 12):
    names, _, _, _ = dataset()
    with lock:
        state = find_session(session_id)
        indices, scores = state.ranking(min(max(limit, 1), 50))
        return {'trained': bool(np.linalg.norm(state.user_vector) > 1e-9), 'count': state.count,
                'items': [{**image_info(i, names), 'score': float(s)} for i, s in zip(indices, scores)]}


@app.get('/api/sessions/{session_id}/history.csv')
def export_history(session_id: str):
    names, _, _, _ = dataset()
    with lock:
        state = find_session(session_id)
        history = list(state.history)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=['round','left','right','choice','left_file','right_file'])
    writer.writeheader()
    for row in history:
        writer.writerow({**row, 'left_file': names[row['left']], 'right_file': names[row['right']]})
    contents = '\ufeff' + buf.getvalue()
    return StreamingResponse(iter([contents.encode('utf-8')]), media_type='text/csv; charset=utf-8',
                             headers={'Content-Disposition': 'attachment; filename="eye_makeup_votes.csv"'})


@app.get('/images/{index}')
def image(index: int):
    names, paths, _, _ = dataset()
    if index < 0 or index >= len(names):
        raise HTTPException(status_code=404, detail='Unknown image')
    if not paths[index].is_file():
        raise HTTPException(status_code=404, detail=f'Missing image: {names[index]}')
    return FileResponse(paths[index])
