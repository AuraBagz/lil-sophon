"""Local voice bridge: browser mic -> DuplexJev-4B-Para ears (end of turn + transcript) -> reply brain -> Phonon
voice -> browser speaker. Also takes recordings from the page (POST /rec).

Run:  python bridge.py [--port 8857] [--skip-gate]      (start-bridge.ps1 runs it detached on Windows)
Open: http://127.0.0.1:8857/   (serves DJV_UI_DIR, / -> /c.html, WebSocket at /ws). Binds 127.0.0.1 only.
Protocol: docs/PROTOCOL.md. Logs under DJV_LOG_DIR (default out/bridge/):
turns.jsonl (one line per turn with latency parts and mic levels), user.jsonl (transcripts), utt/*.wav (the
exact audio given to the model, only when DJV_SAVE_UTT=1; off by default).

Settings (env, defaults = the tuned live setting):
  DJV_BRAIN=bolt|duplexjev    reply writer: Bolt E4B over DJV_BRAIN_URL (text from the ears' transcript), or
                              DuplexJev-4B-Para audio-in. Bolt falls back to DuplexJev if the server is down.
  DJV_CHUNK=clause|sentence   first spoken piece cut at a clause (3+ words) or only at a full stop
  DJV_CHECKS=150,300,600      silence (ms) at which the end-of-turn check runs; thresholds 0.85 / 0.6 / 0.5
  DJV_SPEC=1|0                speculative start, audio held until DJV_HOLD_MS of silence, dropped on more speech
  DJV_HOLD_MS=250  DJV_FORCE_MS=1300  DJV_MAX_TOKENS=200  DJV_HISTORY=6  DJV_MIN_REPLY_MS=450
  DJV_BARGE_MS=280            sustained voice (Silero speech + energy over the playback leak) that cuts the agent
  DJV_BARGE_OVER_DB=8         how far above the measured playback leak (echo) the voice must be
"""
import asyncio
import base64
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from aiohttp import WSMsgType, web

import djv

E = os.environ.get
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else int(E("DJV_PORT", "8857"))
UI_DIR = E("DJV_UI_DIR", os.path.join(djv.ROOT, "ui"))
LOG_DIR = E("DJV_LOG_DIR", os.path.join(djv.ROOT, "out", "bridge"))
REC_DIR = E("DJV_REC_DIR", os.path.join(djv.ROOT, "out", "rec"))
UTT_DIR = os.path.join(LOG_DIR, "utt")
SAVE_UTT = E("DJV_SAVE_UTT", "0") == "1"
FFMPEG = E("DJV_FFMPEG") or shutil.which("ffmpeg")
for d in (LOG_DIR, REC_DIR, UTT_DIR):
    os.makedirs(d, exist_ok=True)

SR = 16000
BLOCK = 320                 # 20 ms VAD block
PREROLL_BLOCKS = 15         # 300 ms kept before speech start
START_BLOCKS = 3            # 60 ms of speech to start an utterance
MIN_VOICED_MS = 300         # utterances with less speech are dropped
BRAIN = E("DJV_BRAIN", "bolt")
HISTORY_MSGS = int(E("DJV_HISTORY", "6"))
MIN_REPLY_VOICED_MS = int(E("DJV_MIN_REPLY_MS", "450"))
CHECKS_MS = tuple(int(x) for x in E("DJV_CHECKS", "150,300,600").split(","))
CHECK_THRESH = {150: float(E("DJV_T150", "0.75")), 300: float(E("DJV_T300", "0.6"))}  # confidence needed to speak at 150 / 300 ms of silence (else 600)
SPEC = E("DJV_SPEC", "1") == "1"
HOLD_MS = int(E("DJV_HOLD_MS", "250"))
FORCE_MS = int(E("DJV_FORCE_MS", "1300"))
MAX_TOKENS = int(E("DJV_MAX_TOKENS", "200"))
BARGE_BLOCKS = int(E("DJV_BARGE_MS", "400")) // 20
BARGE_OVER_DB = float(E("DJV_BARGE_OVER_DB", "6"))
ECHO_TAIL_S = 0.3
BARGE_PROB = float(E("DJV_BARGE_PROB", "0.6"))        # Silero speech probability on every block of the window
BARGE_LEVEL_DB = float(E("DJV_BARGE_LEVEL_DB", "10"))  # within this many dB of the user's own speech level
BARGE_MIN_MS = int(E("DJV_BARGE_MIN_MS", "700"))      # sustained speech before she is actually cut
BARGE_QUIET_MS = int(E("DJV_BARGE_QUIET_MS", "300"))  # this much quiet first = it was a backchannel, resume
BARGE_MAX_BLOCKS = 60                                 # audio kept for the confirm check (1.2 s)
BARGE_P = float(E("DJV_BARGE_P", "0.65"))             # DuplexJev "trying to interrupt?" yes-probability to cut
ASR_AT_MS = int(E("DJV_ASR_AT_MS", "40"))            # early transcript start (Bolt brain)
NORM_TARGET_DB = float(E("DJV_NORM_DB", "-20"))
NORM_MAX_GAIN = 10.0        # +20 dB
SETTINGS = {"brain": BRAIN, "chunk": E("DJV_CHUNK", "clause"), "checks": CHECKS_MS, "spec": SPEC, "hold_ms": HOLD_MS,
            "force_ms": FORCE_MS, "max_tokens": MAX_TOKENS, "history": HISTORY_MSGS, "barge_ms": BARGE_BLOCKS * 20, "barge_prob": BARGE_PROB, "barge_level_db": BARGE_LEVEL_DB, "asr_at_ms": ASR_AT_MS,
            "barge_over_db": BARGE_OVER_DB, "vad": "silero+energy"}
REPLY_GEN = {"repetition_penalty": 1.1}


def norm_gain(x):
    """Speech RMS (loudest 30% of 20 ms blocks) to NORM_TARGET_DB, capped at +20 dB, peak kept under 0.95.
    Applied only to utterances the VAD already accepted as speech, so noise is never what decides a turn."""
    n = len(x) // BLOCK
    if n == 0:
        return 1.0
    e = np.mean(x[: n * BLOCK].reshape(n, BLOCK) ** 2, axis=1)
    top = np.sort(e)[-max(1, int(n * 0.3)):]
    rms = float(np.sqrt(np.mean(top))) + 1e-9
    g = min(NORM_MAX_GAIN, max(0.5, 10 ** (NORM_TARGET_DB / 20) / rms))
    peak = float(np.max(np.abs(x))) + 1e-9
    return round(min(g, 0.95 / peak) if g > 1 else g, 3)


class SpeechVAD:
    """Silero VAD (CPU, 32 ms windows) AND energy above a tracked noise floor. Falls back to energy only."""

    def __init__(self):
        self.floor = -60.0
        self.prob = 0.0
        self.active = False
        self.buf = np.zeros(0, np.float32)
        try:
            import torch
            from silero_vad import load_silero_vad
            self.torch = torch
            self.model = load_silero_vad()
        except Exception as e:  # noqa: BLE001
            print("silero unavailable, energy VAD only:", repr(e)[:200], flush=True)
            self.model = None

    def block(self, b):
        x = b.astype(np.float32) / 32768.0
        db = 10 * np.log10(float(np.mean(x * x)) + 1e-10)
        if self.model is not None:
            self.buf = np.concatenate([self.buf, x])
            while len(self.buf) >= 512:
                with self.torch.no_grad():
                    self.prob = float(self.model(self.torch.from_numpy(self.buf[:512].copy()), SR).item())
                self.buf = self.buf[512:]
            speechy = self.prob >= (0.35 if self.active else 0.5)
        else:
            speechy = db > self.floor + 12
        if (self.model is not None and self.prob < 0.3) or (self.model is None and db < self.floor + 6):
            self.floor = min(-20.0, max(-90.0, 0.95 * self.floor + 0.05 * db))
        v = speechy and db > self.floor + 6
        self.active = v
        return v, db


class Session:
    def __init__(self, app, ws, brain_mode=BRAIN):
        self.app, self.ws, self.loop = app, ws, asyncio.get_running_loop()
        self.brain_mode = brain_mode
        self.ears, self.voice, self.gpu, self.brain = app["ears"], app["voice"], app["gpu"], app["brain"]
        self.vad = SpeechVAD()
        self.residual = np.zeros(0, dtype=np.int16)
        self.preroll = []
        self.utt = []
        self.in_speech = False
        self.voiced_run = 0
        self.barge_run = 0
        self.voiced_total = 0
        self.last_voiced_t = None
        self.check_idx = 0
        self.check_busy = False
        self.turn_id = 0
        self.state = None
        self.history = []
        self.agent = None
        self.closed = False
        self.spec_dropped = 0
        self.prev_first = None
        self.echo_until = 0.0
        self.last_leak_db = -60.0
        self.speech_db = []        # dB of accepted user speech blocks this session (barge-in level reference)
        self.barge_buf = []
        self.barge_hist = []
        self.barge_watch = None
        self.barge_pending = False
        self.barge_hist = []
        self.barge_on = True
        self.asr_job = None

    # ---------------------------------------------------------------- send helpers
    async def send(self, d):
        if not self.closed and not self.ws.closed:
            try:
                await self.ws.send_str(json.dumps(d))
            except Exception:  # noqa: BLE001
                pass

    async def set_state(self, v):
        if v != self.state:
            self.state = v
            await self.send({"type": "state", "value": v})

    def send_threadsafe(self, d):
        asyncio.run_coroutine_threadsafe(self.send(d), self.loop)

    def out(self, a, kind, payload):
        """Agent output: held while the reply is speculative, sent once released."""
        if not a["released"]:
            a["held"].append((kind, payload))
            return
        self._emit(a, kind, payload)

    def _emit(self, a, kind, payload):
        if kind == "json":
            asyncio.ensure_future(self.send(payload))
            return
        now = time.perf_counter()
        if a["first_audio"] is None:
            a["first_audio"] = now
            a["playback_end"] = now
            ms = round((now - a["end"]) * 1000)
            a["latency_ms"] = ms
            ft = a.get("first_text_t", now)
            a["parts"].update(first_chunk_text=round((ft - a["end"]) * 1000),
                              voice_first_chunk=round((a.get("first_pcm_t", now) - ft) * 1000),
                              released_at=round((a["released_t"] - a["end"]) * 1000))
            asyncio.ensure_future(self.set_state("speaking"))
            asyncio.ensure_future(self.send({"type": "latency", "turn": a["turn"], "ms": ms, "parts": a["parts"]}))
        dur = len(payload) / 2 / self.voice.sample_rate
        a["audio_s"] += dur
        a["playback_end"] = max(a["playback_end"], now) + dur
        if not self.ws.closed:
            asyncio.ensure_future(self.ws.send_bytes(payload))

    # ---------------------------------------------------------------- mic in
    async def on_audio(self, data):
        a = np.frombuffer(data, dtype="<i2")
        buf = np.concatenate([self.residual, a])
        n = len(buf) // BLOCK
        self.residual = buf[n * BLOCK:]
        for i in range(n):
            await self.on_block(buf[i * BLOCK:(i + 1) * BLOCK].copy())

    async def on_block(self, b):
        now = time.perf_counter()
        v, db = self.vad.block(b)
        self.voiced_run = self.voiced_run + 1 if v else 0
        a = self.agent
        if a is not None and a["released"]:
            # the agent is talking (or its audio is still playing): the mic hears the speakers too.
            a["mic_db"].append(db)
            # echo reference: the 80th percentile of the mic over the last 2 s of her playback (the speakers'
            # level in the room), not the median (which the gaps between her sentences pull down)
            recent = a["mic_db"][-100:]
            leak = float(np.percentile(recent, 80)) if len(recent) >= 10 else self.vad.floor + 10
            lvl = self.speech_level()
            strong = (v and self.vad.prob >= BARGE_PROB and db > leak + BARGE_OVER_DB and db >= lvl - BARGE_LEVEL_DB)
            self.barge_hist = (self.barge_hist + [bool(strong)])[-BARGE_BLOCKS:]
            self.barge_buf = (self.barge_buf + [b])[-(BARGE_MAX_BLOCKS + PREROLL_BLOCKS):]
            self.preroll = (self.preroll + [b])[-PREROLL_BLOCKS:]
            w = self.barge_watch
            if w is None:
                # stage 1: 400 ms window with 80% qualifying blocks -> duck her and start watching
                if sum(self.barge_hist) >= int(BARGE_BLOCKS * 0.8) and not self.barge_pending:
                    info = {"db": round(db, 1), "leak_db": round(leak, 1), "speech_level_db": round(lvl, 1)}
                    if not self.barge_on:
                        a.setdefault("barge_ignored", []).append(dict(info, why="barge=off"))
                        self.barge_hist = []
                        return
                    self.barge_watch = {"voiced": sum(self.barge_hist), "quiet": 0, "info": info}
                    await self.send({"type": "duck", "on": True, "turn": a["turn"]})
                return
            # stage 2: a backchannel ("mm-hm", "yeah", a laugh) is short; a real interruption keeps going
            leak50 = float(np.percentile(recent, 50)) if len(recent) >= 10 else leak
            speechy = v and self.vad.prob >= 0.5 and db > leak50 + BARGE_OVER_DB and db >= lvl - BARGE_LEVEL_DB - 2
            if speechy:  # stage 2 is a little looser than the trigger: words dip between syllables
                w["voiced"] += 1
                w["quiet"] = 0
            else:
                w["quiet"] += 1
            if w["quiet"] * 20 >= BARGE_QUIET_MS:
                self.barge_watch, self.barge_hist = None, []
                a.setdefault("barge_ignored", []).append(dict(w["info"], why="short, backchannel", voiced_ms=w["voiced"] * 20))
                await self.send({"type": "duck", "on": False, "turn": a["turn"]})
            elif w["voiced"] * 20 >= BARGE_MIN_MS and not self.barge_pending:
                self.barge_pending = True
                info = dict(w["info"], voiced_ms=w["voiced"] * 20)
                asyncio.ensure_future(self.confirm_barge(a, list(self.barge_buf), info))
            return
        if not self.in_speech:
            self.preroll = (self.preroll + [b])[-PREROLL_BLOCKS:]
            if now < self.echo_until and not (v and db > self.last_leak_db + BARGE_OVER_DB):
                return  # tail of our own playback
            if self.voiced_run >= START_BLOCKS:
                self.start_utterance(now)
                await self.set_state("listening")
            return
        self.utt.append(b)
        if v:
            self.voiced_total += 1
            self.last_voiced_t = now
            self.check_idx = 0
            self.speech_db = (self.speech_db + [db])[-600:]
            self.asr_job = None  # an early transcript of a shorter utterance is stale now
            if a is not None and not a["released"] and self.voiced_run >= 2:
                await self.drop_speculation()
            return
        sil_ms = (now - self.last_voiced_t) * 1000
        if a is not None:
            if sil_ms >= a.get("release_ms", HOLD_MS):
                await self.release(a)
            return
        if (self.brain_mode == "bolt" and self.asr_job is None and sil_ms >= ASR_AT_MS
                and self.voiced_total * 20 >= MIN_REPLY_VOICED_MS):
            # start the transcript early, in parallel with the turn check; used if the turn ends here
            audio, dur = self.utt_audio(), len(self.utt) * BLOCK / SR
            self.asr_job = (self.last_voiced_t, self.app["asr_exec"].submit(
                lambda: (time.perf_counter(), djv.clean_transcript(djv.transcribe(self.ears, (audio, SR)), dur),
                         time.perf_counter())))
        if self.voiced_total * 20 < MIN_VOICED_MS and sil_ms > 600:
            self.save_utt(0, "dropped_short")
            self.reset_utterance()
            await self.set_state("idle")
            return
        if self.check_busy:
            return
        if sil_ms >= FORCE_MS:
            if self.voiced_total * 20 < MIN_REPLY_VOICED_MS:
                self.save_utt(0, "dropped_short")
                self.reset_utterance()
                await self.set_state("idle")
                return
            await self.start_reply(forced=True)
            return
        if self.check_idx < len(CHECKS_MS) and sil_ms >= CHECKS_MS[self.check_idx]:
            at = CHECKS_MS[self.check_idx]
            self.check_idx += 1
            self.check_busy = True
            asyncio.ensure_future(self.turn_check(at))

    def speech_level(self):
        return statistics.median(self.speech_db) if len(self.speech_db) >= 25 else -32.0

    async def confirm_barge(self, a, blocks, info):
        """Sustained speech at the user's own level while she talks: ask DuplexJev's barge-in question on that
        audio. A real interruption cuts her off; a laugh, a "mm-hm" or an aside only ducks her for a moment."""
        audio = np.concatenate(blocks).astype(np.float32) / 32768.0
        ctx = 'The assistant was saying: "' + " ".join(a["sentences"])[:300] + '"'
        try:
            res, dt = await self.loop.run_in_executor(
                self.app["turn_exec"], lambda: self.ears.turn((np.clip(audio * norm_gain(audio), -1, 1), SR),
                                                              context=ctx, barge=True, on_gpu_thread=False))
            pb = res.get("p_barge_in", 1.0)
        except Exception as e:  # noqa: BLE001
            pb, dt = 1.0, 0.0
            print("barge check failed", repr(e)[:160], flush=True)
        self.barge_pending = False
        self.barge_hist = []
        self.barge_watch = None
        self.barge_run = 0
        info.update(p_barge_in=round(pb, 3), check_ms=round(dt * 1000))
        if self.agent is not a:
            return
        if pb >= BARGE_P:
            a["barge"] = info
            await self.interrupt("user_speech")
            self.preroll = list(self.barge_buf)  # keep everything the user said since the barge began
            self.start_utterance(time.perf_counter())
        else:
            a.setdefault("barge_ignored", []).append(dict(info, why="not an interruption"))
            await self.send({"type": "duck", "on": False, "turn": a["turn"]})

    def start_utterance(self, now):
        self.in_speech = True
        self.utt = list(self.preroll)
        self.preroll = []
        self.asr_job = None
        self.voiced_total = self.voiced_run
        self.last_voiced_t = now
        self.check_idx = 0
        self.barge_run = 0

    def reset_utterance(self):
        self.in_speech, self.utt, self.voiced_total, self.check_idx = False, [], 0, 0

    def utt_audio(self):
        if not self.utt:
            return np.zeros(SR // 10, np.float32)
        x = np.concatenate(self.utt).astype(np.float32) / 32768.0
        g = norm_gain(x)
        return np.clip(x * g, -1.0, 1.0) if g != 1.0 else x

    def save_utt(self, tid, why):
        """Stats of the utterance (and the raw audio, DJV_SAVE_UTT=1) as handed to the model."""
        import wave
        x = np.concatenate(self.utt) if self.utt else np.zeros(0, np.int16)
        f = x.astype(np.float32) / 32768.0
        stats = {"file": None, "dur_ms": round(len(x) / SR * 1000), "voiced_ms": self.voiced_total * 20,
                 "rms_dbfs": round(10 * np.log10(float(np.mean(f * f)) + 1e-10), 1) if len(f) else None,
                 "peak": round(float(np.max(np.abs(f))), 3) if len(f) else None,
                 "noise_floor_db": round(self.vad.floor, 1), "why": why,
                 "norm_gain": norm_gain(f) if len(f) else None}
        if SAVE_UTT and len(x):
            name = f"{time.strftime('%H%M%S')}_turn{tid}_{why}.wav"
            try:
                with wave.open(os.path.join(UTT_DIR, name), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(x.astype("<i2").tobytes())
                stats["file"] = name
            except Exception as e:  # noqa: BLE001
                stats["file_error"] = repr(e)[:100]
        return stats

    def context(self):
        last = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), None)
        return f'The assistant had just said: "{last}"' if last else None

    # ---------------------------------------------------------------- end of turn
    async def turn_check(self, at_ms):
        t_start = time.perf_counter()
        audio, ctx, last_v = self.utt_audio(), self.context(), self.last_voiced_t
        try:
            res, dt = await self.loop.run_in_executor(self.app["turn_exec"], lambda: self.ears.turn((audio, SR), context=ctx, on_gpu_thread=False))
        finally:
            self.check_busy = False
        if self.last_voiced_t != last_v or not self.in_speech or self.agent is not None:
            return
        p = res["p_take_turn"]
        take = p >= 0.5 and self.voiced_total * 20 >= MIN_REPLY_VOICED_MS
        # start writing as soon as it leans finished; the silence needed before she speaks depends on confidence
        # (0.85+ -> 150 ms, 0.6+ -> 300 ms, else 600 ms), so a doubtful end of turn waits but the reply is ready
        release_ms = max(HOLD_MS, at_ms, 150 if p >= CHECK_THRESH[150] else 300 if p >= CHECK_THRESH[300] else 600)
        await self.send({"type": "turn", "turn": self.turn_id + 1, "at_ms": at_ms, "decision": "reply" if take else "wait",
                         "p_take_turn": p, "class": res["class"], "release_ms": release_ms if take else None})
        if take:
            await self.start_reply(turn_ms=dt * 1000, check_start=t_start, at_ms=at_ms, release_ms=release_ms, p=p)

    async def start_reply(self, forced=False, turn_ms=None, check_start=None, at_ms=None, release_ms=None, p=None):
        self.turn_id += 1
        tid = self.turn_id
        audio = self.utt_audio()
        stop = threading.Event()
        now = time.perf_counter()
        stats = self.save_utt(tid, "reply")
        a = {"turn": tid, "stop": stop, "end": self.last_voiced_t, "utt": stats, "sentences": [], "spoken": [],
             "pending": 0, "text_done": False, "first_audio": None, "audio_s": 0.0, "playback_end": None, "held": [],
             "released": False, "released_t": None, "mic_db": [], "dur_s": len(audio) / SR,
             "parts": {"check_at_ms": at_ms, "check_started": round(((check_start or now) - self.last_voiced_t) * 1000),
                       "turn": round(turn_ms) if turn_ms else None, "forced": forced, "spec": SPEC, "brain": self.brain_mode,
                       "p_take_turn": p, "release_ms": release_ms}}
        a["release_ms"] = release_ms or HOLD_MS
        self.barge_hist, self.barge_run, self.barge_watch = [], 0, None
        asr = self.asr_job[1] if (self.asr_job and self.asr_job[0] == self.last_voiced_t) else None
        a["parts"]["asr_early"] = asr is not None
        self.agent = a
        await self.set_state("thinking")
        hist = [{"role": m["role"], "content": m["content"]} for m in self.history[-HISTORY_MSGS:]]
        self.loop.run_in_executor(self.gpu, self.gen_worker, tid, audio, hist, stop, a["dur_s"], asr)
        if not SPEC or (now - self.last_voiced_t) * 1000 >= a["release_ms"]:
            await self.release(a)

    async def release(self, a):
        if a["released"]:
            return
        a["released"] = True
        a["released_t"] = time.perf_counter()
        self.reset_utterance()
        held, a["held"] = a["held"], []
        for kind, payload in held:
            self._emit(a, kind, payload)
        if a.get("no_turn"):
            self.end_silently(a)
        elif a["text_done"]:
            self.maybe_finish()

    async def drop_speculation(self):
        a = self.agent
        a["stop"].set()
        self.voice.cancel()
        self.agent = None
        self.turn_id -= 1
        self.spec_dropped += 1
        await self.send({"type": "turn", "turn": a["turn"], "decision": "speculation_dropped"})
        await self.set_state("listening")

    def gen_worker(self, tid, audio, hist, stop, dur_s, asr=None):
        """GPU thread: write the reply and hand pieces to the voice.
        bolt: transcript first (it is the brain's input and the user caption), then Bolt streams the reply.
        duplexjev: the ears' model replies from the audio, the transcript runs after (caption + history)."""
        t0 = time.perf_counter()
        said, used = None, self.brain_mode
        if self.brain_mode == "bolt":
            if asr is not None:  # the early transcript started at ASR_AT_MS of silence, usually done already
                t_a, said, t_b = asr.result()
                asr_ms = round((t_b - t_a) * 1000)
            else:
                said = djv.clean_transcript(djv.transcribe(self.ears, (audio, SR)), dur_s)
                asr_ms = round((time.perf_counter() - t0) * 1000)
            self.loop.call_soon_threadsafe(self.on_user_text, tid, said, asr_ms)
            if not said:
                self.loop.call_soon_threadsafe(self.on_no_turn, tid)
                return
        ch, full, first_tok = djv.Chunker(), "", None
        try:
            if self.brain_mode == "bolt":
                try:
                    stream = self.brain.stream(said, history=hist, max_tokens=MAX_TOKENS, stop_event=stop)
                    first = next(stream)
                    stream = _chain([first], stream)
                except Exception as e:  # noqa: BLE001
                    print("bolt unavailable, DuplexJev replies:", repr(e)[:160], flush=True)
                    used = "duplexjev(fallback)"
                    stream = self.ears.reply_stream((audio, SR), history=hist, stop_event=stop, system=djv.SYSTEM_V3,
                                                    max_new_tokens=MAX_TOKENS, gen_kw=REPLY_GEN)
            else:
                stream = self.ears.reply_stream((audio, SR), history=hist, stop_event=stop, system=djv.SYSTEM_V3,
                                                max_new_tokens=MAX_TOKENS, gen_kw=REPLY_GEN)
            for piece, _ in stream:
                if stop.is_set():
                    break
                if first_tok is None:
                    first_tok = time.perf_counter()
                full += piece
                for s in ch.feed(piece):
                    self.loop.call_soon_threadsafe(self.on_sentence, tid, s)
            if not stop.is_set():
                for s in ch.flush():
                    self.loop.call_soon_threadsafe(self.on_sentence, tid, s)
        except Exception as e:  # noqa: BLE001
            self.send_threadsafe({"type": "error", "text": f"reply failed: {e!r}"[:300]})
        if stop.is_set():
            return
        self.loop.call_soon_threadsafe(self.on_text_done, tid, djv.clean_reply(full), first_tok, used)
        if self.brain_mode != "bolt":
            try:
                t1 = time.perf_counter()
                said = djv.clean_transcript(djv.transcribe(self.ears, (audio, SR)), dur_s)
            except Exception as e:  # noqa: BLE001
                said = ""
                print("transcript failed", repr(e)[:160], flush=True)
            self.loop.call_soon_threadsafe(self.on_user_text, tid, said, round((time.perf_counter() - t1) * 1000))

    def cur(self, tid):
        a = self.agent
        return a if a and a["turn"] == tid and not a["stop"].is_set() else None

    def on_no_turn(self, tid):
        """The transcript says there was no real speech (noise, a cough): no reply at all."""
        a = self.cur(tid)
        if not a:
            return
        a["no_turn"] = True
        if a["released"]:
            self.end_silently(a)

    def end_silently(self, a):
        self.log_turn(a, interrupted=None, note="no_turn")
        self.agent = None
        asyncio.ensure_future(self.send({"type": "turn", "turn": a["turn"], "decision": "no_speech"}))
        asyncio.ensure_future(self.set_state("listening" if self.in_speech else "idle"))

    @staticmethod
    def _norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    def on_sentence(self, tid, s):
        a = self.cur(tid)
        if not a or not any(ch.isalnum() for ch in s):  # never speak "..." or bare punctuation
            return
        idx = len(a["sentences"])
        if idx == 0:
            if self.prev_first and self._norm(s) == self.prev_first:
                a["parts"]["skipped_repeat_opening"] = True
                return  # same opening as the last reply: skip it, the rest still plays
            a["first_text_t"] = time.perf_counter()
            a["first_norm"] = self._norm(s)
        a["sentences"].append(s)
        a["pending"] += 1
        self.voice.stream_async(s, f"{tid}:{idx}")
        self.out(a, "json", {"type": "caption", "who": "agent", "turn": tid, "text": s, "final": False})

    def on_text_done(self, tid, reply, first_tok, used):
        a = self.cur(tid)
        if not a:
            return
        a["text_done"] = True
        a["reply"] = reply
        a["parts"]["brain_used"] = used
        if first_tok:
            a["parts"]["ttft"] = round((first_tok - a["end"]) * 1000)
        if a.get("first_norm"):
            self.prev_first = a["first_norm"]
        self.out(a, "json", {"type": "caption", "who": "agent", "turn": tid, "text": " ".join(a["sentences"]),
                             "final": True})
        if a["released"]:
            self.maybe_finish()

    def on_user_text(self, tid, said, asr_ms):
        with open(os.path.join(LOG_DIR, "user.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "turn": tid, "said": said, "asr_ms": asr_ms}) + "\n")
        a = self.agent
        if a and a["turn"] == tid:
            a["parts"]["asr_ms"] = asr_ms
        if not said:
            return
        asyncio.ensure_future(self.send({"type": "caption", "who": "user", "turn": tid, "text": said, "final": True}))
        msg = {"role": "user", "content": said, "turn": tid}
        for i, m in enumerate(self.history):
            if m.get("turn") == tid and m["role"] == "assistant":
                self.history.insert(i, msg)
                return
        self.history.append(msg)

    # ---------------------------------------------------------------- voice out (Phonon reader thread)
    def on_voice_line(self, d):
        self.loop.call_soon_threadsafe(self._on_voice_line, d)

    def _on_voice_line(self, d):
        rid = str(d.get("id") or "")
        a = self.agent
        if not a or not rid.startswith(f"{a['turn']}:") or a["stop"].is_set():
            return
        if "pcm" in d:
            if "first_pcm_t" not in a:
                a["first_pcm_t"] = time.perf_counter()
            self.out(a, "pcm", base64.b64decode(d["pcm"]))
            return
        a["pending"] -= 1
        if d.get("ok"):
            a["spoken"].append(a["sentences"][int(rid.split(":")[1])])
        if a["released"]:
            self.maybe_finish()

    def maybe_finish(self):
        a = self.agent
        if a and a["released"] and a["text_done"] and a["pending"] <= 0 and not a.get("ending"):
            a["ending"] = True
            asyncio.ensure_future(self.send({"type": "audio_end", "turn": a["turn"]}))
            wait = max(0.0, (a["playback_end"] or time.perf_counter()) - time.perf_counter())
            tid = a["turn"]
            self.loop.call_later(wait, lambda: self.finish_turn(tid))

    def finish_turn(self, tid):
        a = self.agent
        if not a or a["turn"] != tid:
            return
        self.log_turn(a, interrupted=None)
        self.history.append({"role": "assistant", "content": " ".join(a["spoken"]) or a.get("reply", ""), "turn": tid})
        if a["mic_db"]:
            self.last_leak_db = statistics.median(a["mic_db"])
        self.echo_until = time.perf_counter() + ECHO_TAIL_S
        self.agent = None
        asyncio.ensure_future(self.set_state("listening" if self.in_speech else "idle"))

    async def interrupt(self, by):
        a = self.agent
        if not a:
            return
        a["stop"].set()
        self.voice.cancel()
        self.agent = None
        if not a["released"]:
            self.turn_id -= 1
            await self.set_state("listening")
            return
        heard = a["spoken"][:-1] if a["playback_end"] and a["playback_end"] > time.perf_counter() else a["spoken"]
        if heard:
            self.history.append({"role": "assistant", "content": " ".join(heard) + " ...", "turn": a["turn"]})
        self.log_turn(a, interrupted=by)
        await self.send({"type": "interrupted", "turn": a["turn"], "by": by})
        await self.set_state("listening")

    def log_turn(self, a, interrupted, note=None):
        m = a["mic_db"]
        mic = ({"median_db": round(statistics.median(m), 1), "p90_db": round(float(np.percentile(m, 90)), 1),
                "max_db": round(max(m), 1), "blocks": len(m)} if m else None)
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": SETTINGS, "brain": self.brain_mode, "turn": a["turn"],
               "latency_ms": a.get("latency_ms"), "parts": a["parts"], "utt": a.get("utt"), "reply": a.get("reply"),
               "audio_s": round(a["audio_s"], 2), "chars": len(a.get("reply") or ""), "interrupted": interrupted,
               "barge": a.get("barge"), "barge_ignored": a.get("barge_ignored"),
               "mic_during_playback": mic, "note": note, "spec_dropped_so_far": self.spec_dropped}
        with open(os.path.join(LOG_DIR, "turns.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _chain(first, rest):
    yield from first
    yield from rest


# -------------------------------------------------------------------- http + ws
async def ws_handler(request):
    app = request.app
    ws = web.WebSocketResponse(max_msg_size=4 * 2**20, heartbeat=20)
    await ws.prepare(request)
    old = app["session"]
    if old is not None:  # newest connection takes over
        old.closed = True
        if old.agent:
            old.agent["stop"].set()
            app["voice"].cancel()
            old.agent = None
        try:
            await old.ws.send_str(json.dumps({"type": "error", "text": "replaced by a newer connection"}))
            await old.ws.close(code=4000, message=b"replaced")
        except Exception:  # noqa: BLE001
            pass
        print("session replaced by a newer connection", flush=True)
    q = request.query.get("brain", BRAIN).lower()
    brain_mode = "duplexjev" if q.startswith("duplex") else "bolt"
    s = Session(app, ws, brain_mode)
    s.barge_on = request.query.get("barge", "on").lower() not in ("off", "0", "false", "no")
    print("session open", request.remote, "brain", brain_mode, flush=True)
    app["session"] = s
    app["voice"].on_line = s.on_voice_line
    hello = {"type": "ready", "audio_in": {"rate": 16000, "format": "s16le", "channels": 1},
             "audio_out": {"rate": app["voice"].sample_rate, "format": "s16le", "channels": 1},
             "model": "DuplexJev-4B-Para", "voice": "Phonon 31m", "brain": brain_mode, "barge": "on" if s.barge_on else "off",
             "brain_label": "Bolt E4B (text from DuplexJev transcript)" if brain_mode == "bolt" else "DuplexJev-4B-Para (audio in)",
             "settings": SETTINGS}
    await ws.send_str(json.dumps(hello))
    await s.set_state("idle")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await s.on_audio(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    d = json.loads(msg.data)
                except Exception:  # noqa: BLE001
                    continue
                t = d.get("type")
                if t in ("interrupt", "stop"):
                    if s.agent:
                        await s.interrupt("client")
                    else:
                        await s.send({"type": "interrupted", "turn": s.turn_id, "by": "client"})
                        await s.set_state("listening" if s.in_speech else "idle")
                elif t == "reset":
                    s.history = []
                elif t == "hello":
                    await ws.send_str(json.dumps(hello))
    except Exception as e:  # noqa: BLE001
        print("session error", repr(e)[:300], flush=True)
    finally:
        s.closed = True
        if app["session"] is s:
            if s.agent:
                s.agent["stop"].set()
                app["voice"].cancel()
            app["voice"].on_line = None
            app["session"] = None
            print("session closed", flush=True)
    return ws


async def rec_handler(request):
    """POST /rec: raw body streamed to disk, then always H.264 + AAC 48 kHz mp4 (yuv420p, constant 60 fps from the
    page's variable 49 to 55 fps, faststart, CRF 16); the original upload is kept as <name>.orig.<ext>."""
    name = re.sub(r"[^A-Za-z0-9_-]", "", request.headers.get("X-Rec-Name", ""))[:80] or time.strftime("take-%Y%m%d-%H%M%S")
    ctype = request.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = {"video/mp4": "mp4", "video/webm": "webm", "audio/webm": "webm", "video/x-matroska": "mkv"}.get(ctype, "bin")
    orig = os.path.join(REC_DIR, f"{name}.orig.{ext}")
    mp4 = os.path.join(REC_DIR, f"{name}.mp4")
    n = 0
    with open(orig, "wb") as f:
        async for chunk in request.content.iter_chunked(1 << 20):
            f.write(chunk)
            n += len(chunk)
    if not FFMPEG:
        return web.json_response({"ok": False, "error": "ffmpeg not found (set DJV_FFMPEG)", "path_orig": orig, "bytes": n})
    t0 = time.perf_counter()
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", orig, "-c:v", "libx264", "-preset", "medium",
            "-crf", "16", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", "60", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", mp4]
    kw = {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kw)
    _, err = await p.communicate()
    ok = p.returncode == 0 and os.path.exists(mp4)
    print(f"rec {name}: {n} bytes {ctype}, transcode {'ok' if ok else 'FAILED'} in {time.perf_counter() - t0:.1f} s", flush=True)
    return web.json_response({"ok": ok, "path_mp4": mp4 if ok else None, "path_orig": orig, "bytes": n,
                              **({} if ok else {"error": err.decode("utf-8", "ignore")[-300:]})})


async def root(request):
    raise web.HTTPFound("/c.html")


async def static(request):
    rel = request.match_info["path"]
    base = os.path.realpath(UI_DIR)
    p = os.path.realpath(os.path.join(base, rel))
    if not p.startswith(base + os.sep) or not os.path.isfile(p):
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-store"})


def main():
    if "--skip-gate" not in sys.argv:
        ok, info = djv.gate()
        print("gate", ok, info, flush=True)
        if not ok:
            sys.exit(3)
    t0 = time.perf_counter()
    voice = djv.Voice()
    ears = djv.Ears()
    print(f"loaded voice {voice.load_s:.1f} s, ears {ears.load_s:.1f} s, {ears.vram()}", flush=True)
    t1 = time.perf_counter()
    warm = os.path.join(djv.ROOT, "clips", "q3_moon.wav")
    ears.warm(warm)
    djv.transcribe(ears, warm)
    brain = djv.Brain()
    if True:  # always warm the Bolt brain (sessions pick their brain per connection)
        try:
            for _ in brain.stream("Say hi.", max_tokens=4):
                pass
            print("brain: Bolt at", djv.BRAIN_URL, flush=True)
        except Exception as e:  # noqa: BLE001
            print("brain: Bolt NOT reachable, replies fall back to DuplexJev:", repr(e)[:160], flush=True)
    print(f"warm {time.perf_counter() - t1:.1f} s; total startup {time.perf_counter() - t0:.1f} s", flush=True)
    print("settings", json.dumps(SETTINGS), flush=True)
    app = web.Application(client_max_size=2 * 2**30)
    app["ears"], app["voice"], app["brain"], app["session"] = ears, voice, brain, None
    app["gpu"] = ThreadPoolExecutor(max_workers=1)        # reply writer (consumes the stream)
    app["turn_exec"] = ThreadPoolExecutor(max_workers=1)  # eager turn / barge checks, concurrent with generation
    app["asr_exec"] = ThreadPoolExecutor(max_workers=1)   # early transcripts
    app.router.add_get("/", root)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/rec", rec_handler)
    app.router.add_get("/{path:.+}", static)
    if not os.path.isdir(UI_DIR):
        print(f"note: {UI_DIR} not there; /ws works, pages appear once it exists", flush=True)
    print(f"bridge on http://127.0.0.1:{PORT}/  ws://127.0.0.1:{PORT}/ws  rec POST /rec -> {REC_DIR}", flush=True)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None, access_log=None)


if __name__ == "__main__":
    main()
