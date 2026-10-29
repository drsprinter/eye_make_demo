"""Single-service FastAPI web demo for Render."""
import csv
import io
import os
import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from .preference import PreferenceSession, PAIR_ROUNDS, RATING_ROUNDS

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (ROOT / 'Eyemake4').resolve()
NPZ_PATH = ASSETS / 'clip_embeddings.npz'
WEB_DIR = ROOT / 'web'
app = FastAPI(title='Eye Makeup Two-Stage Preference Lab', version='2.0')
lock = threading.RLock()
sessions = {}

@app.get('/', include_in_schema=False)
def homepage():
    return FileResponse(WEB_DIR / 'index.html')

@app.get('/assets/{filename}', include_in_schema=False)
def assets(filename: str):
    if filename not in {'style.css', 'app.js'}:
        raise HTTPException(404, 'Not Found')
    return FileResponse(WEB_DIR / filename)


def load_dataset():
    if not NPZ_PATH.exists():
        raise RuntimeError(f'NPZ not found: {NPZ_PATH}')
    with np.load(NPZ_PATH, allow_pickle=False) as z:
        names = [str(s) for s in z['paths']]
        vectors = np.asarray(z['embeddings'], dtype=np.float32)
    if vectors.ndim != 2 or len(names) != len(vectors) or len(names) < 50 or not np.isfinite(vectors).all():
        raise ValueError('NPZ must have >= 50 valid vectors and matching paths')
    paths = []
    for name in names:
        path = (ASSETS / name).resolve()
        if not path.is_relative_to(ASSETS) or path == ASSETS:
            raise ValueError('Unsafe image path in NPZ')
        paths.append(path)
    return names, paths, vectors


def dataset():
    try:
        return load_dataset()
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


def session(sid):
    if sid not in sessions:
        raise HTTPException(404, 'Session not found; restart the questionnaire')
    return sessions[sid]


def image_info(i, names):
    return {'index': int(i), 'name': names[int(i)], 'url': f'/images/{int(i)}'}


def payload(s, names):
    d = {'phase': s.phase, 'pair_completed': s.pair_count, 'pair_total': PAIR_ROUNDS,
         'rating_completed': s.rating_count, 'rating_total': RATING_ROUNDS}
    if s.phase == 'pair':
        a, b = s.pair
        d['pair'] = {'left': image_info(a, names), 'right': image_info(b, names)}
    elif s.phase == 'rating':
        d['candidate'] = image_info(s.candidate, names)
    return d


class Vote(BaseModel):
    choice: str


class Rating(BaseModel):
    rating: int = Field(ge=1, le=5)


@app.get('/api/health')
def health():
    names, paths, vecs = dataset()
    return {'status': 'ok', 'images': len(names), 'dimensions': vecs.shape[1],
            'missing_images': sum(not p.is_file() for p in paths)}


@app.post('/api/sessions')
def create_session():
    names, _, vecs = dataset()
    sid = uuid.uuid4().hex
    with lock:
        s = PreferenceSession(vecs, seed=int.from_bytes(os.urandom(4), 'big'))
        sessions[sid] = s
        return {'session_id': sid, **payload(s, names)}


@app.get('/api/sessions/{sid}')
def status(sid: str):
    names, _, _ = dataset()
    with lock:
        return {'session_id': sid, **payload(session(sid), names)}


@app.post('/api/sessions/{sid}/votes')
def vote(sid: str, vote: Vote):
    names, _, _ = dataset()
    with lock:
        s = session(sid)
        try:
            s.vote(vote.choice)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return payload(s, names)


@app.post('/api/sessions/{sid}/ratings')
def rate(sid: str, entry: Rating):
    names, _, _ = dataset()
    with lock:
        s = session(sid)
        try:
            s.rate(entry.rating)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return payload(s, names)


@app.get('/api/sessions/{sid}/recommendations')
def recommendations(sid: str):
    names, _, _ = dataset()
    with lock:
        s = session(sid)
        if s.phase != 'done':
            raise HTTPException(409, 'Complete all 20 comparisons and 10 ratings first')
        ids, scores = s.ranking(10)
        return {'items': [{**image_info(i, names), 'score': float(sc)} for i, sc in zip(ids, scores)],
                'pair_completed': s.pair_count, 'rating_completed': s.rating_count}


@app.get('/api/sessions/{sid}/history.csv')
def history_csv(sid: str):
    names, _, _ = dataset()
    with lock:
        records = list(session(sid).history)
    f = io.StringIO()
    w = csv.writer(f)
    w.writerow(['phase','round','left_index','left_file','right_index','right_file','choice','image_index','image_file','rating'])
    for h in records:
        l, r, i = h['left'], h['right'], h['image']
        w.writerow([h['phase'], h['round'], l, names[l] if l != '' else '', r,
                    names[r] if r != '' else '', h['choice'], i, names[i] if i != '' else '', h['rating']])
    return StreamingResponse(iter([('\ufeff' + f.getvalue()).encode('utf-8')]), media_type='text/csv',
                             headers={'Content-Disposition': 'attachment; filename="eye_makeup_history.csv"'})


@app.get('/images/{index}')
def image(index: int):
    names, paths, _ = dataset()
    if not 0 <= index < len(names):
        raise HTTPException(404, 'Image not found')
    if not paths[index].is_file():
        raise HTTPException(404, f'Missing image file: {names[index]}')
    return FileResponse(paths[index])
