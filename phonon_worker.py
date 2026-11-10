"""Resident Phonon 31m speaking worker (CPU), driven over stdin/stdout JSON lines.

Run with a Python that has `ptts` (Phonon) installed:  python -I phonon_worker.py
Requests (one JSON per line):
  {"text": "...", "out": "path.wav"}                   synthesize to a wav file
  {"text": "...", "pcm": true, "id": 7}                stream: one {"id":7,"pcm":"<base64 s16le>"} line per chunk
  {"cancel": true}                                     stop the current request and drop every queued one
Replies: {"ok": true, "id":.., "first_chunk_s": .., "total_s": .., "audio_s": .., "out": ..} or
         {"ok": false, "id":.., "cancelled": true}
The model loads once at start; the first stdout line is {"ready": true, "load_s": .., "sample_rate": ..}.
"""
import base64
import json
import os
import queue
import sys
import threading
import time
import wave

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf"))

import numpy as np  # noqa: E402
import ptts  # noqa: E402

_lock = threading.Lock()


def emit(d):
    with _lock:
        sys.stdout.write(json.dumps(d) + "\n")
        sys.stdout.flush()


def write_wav(path, x, sr):
    pcm = (np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


t0 = time.perf_counter()
tts = ptts.TTS(lang="en")
sr = int(tts.sample_rate)
_ = tts.synth("Ready.")  # warm-up so the first real request is not a cold one
emit({"ready": True, "load_s": round(time.perf_counter() - t0, 3), "sample_rate": sr})

jobs = queue.Queue()
cancel = threading.Event()
gen = [0]  # bumps on every cancel; queued jobs from an older generation are dropped


def reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa: BLE001
            emit({"ok": False, "error": repr(e)[:200]})
            continue
        if req.get("cancel"):
            gen[0] += 1
            cancel.set()
            continue
        jobs.put((gen[0], req))
    jobs.put(None)


threading.Thread(target=reader, daemon=True).start()

while True:
    item = jobs.get()
    if item is None:
        break
    g, req = item
    rid = req.get("id")
    if g != gen[0]:
        emit({"ok": False, "id": rid, "cancelled": True})
        continue
    cancel.clear()
    try:
        kw = {"voice": req["voice"]} if req.get("voice") else {}
        t1 = time.perf_counter()
        first = None
        chunks = []
        cancelled = False
        it = tts.stream(req["text"], **kw)
        for c in it:
            if cancel.is_set() or g != gen[0]:
                cancelled = True
                break
            c = np.asarray(c, dtype=np.float32).reshape(-1)
            if first is None:
                first = time.perf_counter() - t1
            if req.get("pcm"):
                b = (np.clip(c, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                emit({"id": rid, "pcm": base64.b64encode(b).decode("ascii")})
            else:
                chunks.append(c)
        del it
        if cancelled:
            emit({"ok": False, "id": rid, "cancelled": True})
            continue
        total = time.perf_counter() - t1
        rep = {"ok": True, "id": rid, "first_chunk_s": round(first or total, 3), "total_s": round(total, 3)}
        if not req.get("pcm"):
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            write_wav(req["out"], audio, sr)
            rep.update(audio_s=round(len(audio) / sr, 3), out=req["out"])
        emit(rep)
    except Exception as e:  # noqa: BLE001
        emit({"ok": False, "id": rid, "error": repr(e)[:300]})
