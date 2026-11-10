"""Shared helpers: cache env, GPU gate, DuplexJev-4B-Para load, turn decision, streamed text reply, Phonon worker.

Every path is relative to this folder (the repo root) unless overridden by an environment variable or by
config.local.json next to this file (machine-specific, git-ignored). All caches live under the repo.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:  # machine-specific overrides, e.g. {"DJV_PHONON_PY": "...", "DJV_FFMPEG": "..."}
    for _k, _v in json.load(open(os.path.join(ROOT, "config.local.json"), encoding="utf-8")).items():
        os.environ.setdefault(_k, str(_v))
except FileNotFoundError:
    pass
for k, v in {
    "HF_HOME": os.path.join(ROOT, "hf"),
    "HUGGINGFACE_HUB_CACHE": os.path.join(ROOT, "hf", "hub"),
    "HF_XET_CACHE": os.path.join(ROOT, "hf", "xet"),
    "TORCH_HOME": os.path.join(ROOT, "cache", "torch"),
    "TRITON_CACHE_DIR": os.path.join(ROOT, "cache", "triton"),
    "TMP": os.path.join(ROOT, "tmp"),
    "TEMP": os.path.join(ROOT, "tmp"),
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(k, v)
# force our own compile caches (a global TORCHINDUCTOR_CACHE_DIR on this box pointed at C:)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.environ.get("DJV_INDUCTOR_CACHE", os.path.join(ROOT, "cache", "inductor"))

import ctypes  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

MODEL_DIR = os.environ.get("DJV_MODEL_DIR", os.path.join(ROOT, "hf", "models", "DuplexJev-4B-Para"))
PHONON_PY = os.environ.get("DJV_PHONON_PY", sys.executable)  # a Python with `ptts` installed
PHONON_HF_HOME = os.environ.get("DJV_PHONON_HF_HOME", os.path.join(ROOT, "hf"))
PHONON_WORKER = os.path.join(ROOT, "phonon_worker.py")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console windows flashing on Windows
BRAIN_URL = os.environ.get("DJV_BRAIN_URL", "http://127.0.0.1:8858/v1/chat/completions")

TURN_Q = {"id": "turn", "text": "Has the user finished speaking?",
          "options": ["finished, the assistant can reply", "not finished, still talking",
                      "just a backchannel, not taking the turn", "hesitating or asking to wait"], "lang": "en"}
BARGE_Q = {"id": "barge_in", "text": "Is the user trying to interrupt or take over the conversation?",
           "options": ["yes", "no"], "lang": "en"}
TURN_NAMES = ["finished", "not finished", "backchannel", "hesitating"]

# Kept short on purpose: the 2026-11-09 live session showed a longer self-describing prompt being parroted back
# ("I'm Sophi, a friendly voice assistant ... I can talk out loud"). No name rule lives in the prompt AND in
# clean_reply (the checkpoint's trained identity is "Sophon").
SYSTEM_PLAIN = ("Reply to what the user just said in one or two short sentences. Plain spoken words, no lists. "
                "If asked your name, say you don't have one.")
SYSTEM_SHORT = SYSTEM_PLAIN + " Start with a short phrase, then answer."
# v3 (after the live parroting, 2026-11-09): answer the question, some personality, no self-intro, no name.
SYSTEM_V3 = ("You are a voice assistant running locally on this PC. Answer exactly what the user just said, in one or "
             "two short spoken sentences. Be warm, quick and a little witty. Do not introduce yourself unless asked. "
             "You have no name; if asked who you are, say you are a voice assistant running locally on this PC. "
             "Never repeat an earlier answer. If asked for a joke or a story, actually tell it, complete, in under "
             "thirty seconds of speech, and if told it was bad, tell a different one. For arithmetic, multiply and divide "
             "before adding and subtracting (2 + 2 x 3 = 8). No lists, no markdown.")

_NOISE_WORDS = {"noise", "static", "silence", "music", "inaudible", "unintelligible", "background", "breathing",
                "sniff", "sniffs", "cough", "coughs", "coughing", "sigh", "sighs", "laughter", "laughs", "clicking"}


def clean_transcript(t, dur_s=None):
    """Make the ears' transcript safe to show and to put in history. Returns "" when it is not a real user turn:
    leaked reasoning about the instruction, noise tags, or more words than the audio could hold."""
    t = (t or "").split("</think>")[-1].strip().strip('"').strip()
    if not t:
        return ""
    if re.search(r"(?i)\b(the user|transcri\w*|verbatim|the speech above|instruction)\b", t):
        return ""
    words = re.findall(r"[A-Za-z']+", t)
    if not words or re.fullmatch(r"\W*\(\(?.*\)\)?\W*", t) or all(w.lower() in _NOISE_WORDS for w in words):
        return ""
    if dur_s and len(words) / max(dur_s, 0.3) > 6.0:
        return ""
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:  # ALL CAPS from the ASR: sentence case
        t = t.lower()
        t = re.sub(r"\bi\b", "I", t)
        t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t[:1].upper() + t[1:]


class Brain:
    """Bolt E4B (llama-server, OpenAI-compatible) as the reply writer; DuplexJev stays the ears."""

    def __init__(self, url=None):
        self.url = url or BRAIN_URL

    def stream(self, text, history=None, system=None, max_tokens=200, stop_event=None):
        import http.client
        import urllib.parse
        u = urllib.parse.urlparse(self.url)
        msgs = [{"role": "system", "content": system or SYSTEM_V3}] + list(history or []) + [{"role": "user", "content": text}]
        body = json.dumps({"messages": msgs, "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
                           "top_p": 0.95, "cache_prompt": True,
                           "chat_template_kwargs": {"enable_thinking": False}})
        t0 = time.perf_counter()
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        c.request("POST", u.path, body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        try:
            for raw in r:
                if stop_event is not None and stop_event.is_set():
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    piece = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except Exception:  # noqa: BLE001
                    continue
                if piece:
                    yield piece, time.perf_counter() - t0
        finally:
            c.close()
SYSTEM = SYSTEM_SHORT if os.environ.get("DJV_PROMPT", "short") == "short" else SYSTEM_PLAIN


# ------------------------------------------------------------------------------------------------ gate
def free_ram_gb():
    if os.name != "nt":
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
        return 0.0
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2**30


def vram_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, creationflags=NOWIN).stdout
    return int(out.strip().splitlines()[0])


def busy_procs():
    if os.name != "nt":
        return []
    ps = ("Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*' -and $_.CommandLine -like '*main.py*') "
          "-or $_.Name -like 'VALORANT*' } | ForEach-Object { $_.Name }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60, creationflags=NOWIN).stdout
    return [x for x in out.split() if x]


def gate(max_vram=None, min_ram=None):
    """Refuse to load when another GPU job runs: VRAM in use, free RAM, a ComfyUI main.py or a game.
    Thresholds: DJV_GATE_VRAM_MIB (6000), DJV_GATE_RAM_GB (12); DJV_GATE=0 skips it."""
    if os.environ.get("DJV_GATE", "1") == "0":
        return True, {"skipped": True}
    max_vram = max_vram or int(os.environ.get("DJV_GATE_VRAM_MIB", "6000"))
    min_ram = min_ram or float(os.environ.get("DJV_GATE_RAM_GB", "12"))
    v, r, b = vram_used_mib(), free_ram_gb(), busy_procs()
    ok = v < max_vram and r >= min_ram and not b
    return ok, {"vram_used_mib": v, "free_ram_gb": round(r, 1), "busy": b}


# ------------------------------------------------------------------------------------------------ model
class GpuThread:
    """One long-lived thread that runs EVERY model call (load, compile, decide, generate). torch.compile's CUDA-graph
    trees keep state in the thread-local storage of the thread that imported them, so calling the compiled model
    from other threads fails (AssertionError in cudagraph_trees.get_obj). Callers submit work and wait."""

    def __init__(self):
        from concurrent.futures import Future
        self.Future = Future
        self.q = queue.Queue()
        self.ident = None
        threading.Thread(target=self._run, daemon=True, name="djv-gpu").start()

    def _run(self):
        self.ident = threading.get_ident()
        while True:
            fn, fut = self.q.get()
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)

    def submit(self, fn):
        fut = self.Future()
        self.q.put((fn, fut))
        return fut

    def call(self, fn):
        if threading.get_ident() == self.ident:
            return fn()
        return self.submit(fn).result()


class Ears:
    def __init__(self, model_dir=MODEL_DIR):
        import torch
        self.torch = torch
        self.gpu = GpuThread()
        t0 = time.perf_counter()

        def load():
            import torch._inductor.cudagraph_trees  # noqa: F401  (imported on the GPU thread on purpose)
            from duplexjev import Decider
            d = Decider.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            return d
        self.d = self.gpu.call(load)
        self.load_s = time.perf_counter() - t0
        self.model, self.proc, self.tok = self.d.model, self.d.processor, self.d.tok

    def vram(self):
        t = self.torch
        return {"allocated_mib": round(t.cuda.memory_allocated() / 2**20),
                "peak_allocated_mib": round(t.cuda.max_memory_allocated() / 2**20),
                "reserved_mib": round(t.cuda.memory_reserved() / 2**20)}

    def turn(self, audio, context=None, barge=False, on_gpu_thread=True):
        """One forward pass: turn state (+ optional barge-in). Returns (dict, seconds).
        on_gpu_thread=False runs it in the caller's thread: decide() is eager (not compiled), so it can overlap a
        transcript or reply being generated on the GPU thread instead of queueing behind it."""
        qs = [TURN_Q] + ([BARGE_Q] if barge else [])

        def run():
            t0 = time.perf_counter()
            r = self.d.decide(audio, qs, context=context)
            return r, time.perf_counter() - t0
        with self.torch.inference_mode():
            r, dt = self.gpu.call(run) if on_gpu_thread else run()
        probs = r["turn"]["probs"]
        p = [float(probs.get(o, 0.0)) for o in TURN_Q["options"]]
        out = {"class": TURN_NAMES[max(range(4), key=lambda i: p[i])],
               "p": dict(zip(TURN_NAMES, [round(x, 3) for x in p])),
               "p_take_turn": round(p[0] + p[2], 3)}
        if barge:
            out["p_barge_in"] = round(float(r["barge_in"]["probs"].get("yes", 0.0)), 3)
        return out, dt

    def warm(self, audio, fast=True):
        """Compile the decode step (static cache + torch.compile, about 40 s once) and allocate a 1k-token cache so
        later calls reuse it without recompiling. Without this, decode runs ~20 tok/s (Windows launch-bound)."""
        self.fast = fast
        self.turn(audio)
        for _ in range(2):
            for _ in self.reply_stream(audio, max_new_tokens=900 if _ == 0 else 8):
                pass

    def reply_stream(self, audio, history=None, max_new_tokens=160, stop_event=None, system=None, user_text=None,
                     gen_kw=None, text_only=None):
        """Yield (text_piece, t_since_call) as the reply streams. history: list of {role, content} before this turn.
        stop_event: threading.Event; when set, generation stops at the next token."""
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        from duplexjev.decider import load_audio, SAMPLE_RATE
        torch = self.torch
        t0 = time.perf_counter()
        if text_only is not None:
            user = text_only
        else:
            user = "<|audio|>" if not user_text else user_text + "\n\n<|audio|>"
        msgs = [{"role": "system", "content": system or SYSTEM}] + list(history or []) + [{"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        gen = self.model.generate
        if text_only is not None:  # transcript text in, no audio: the checkpoint's Qwen3-4B (merged LoRAs) alone
            inputs = dict(self.tok(prompt, return_tensors="pt").to(self.d.device))
            gen = self.model.language_model.generate
        else:
            a = self.d._align(load_audio(audio))
            p = self.proc(text=prompt, audio=a, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {}
            for k, v in p.items():
                if hasattr(v, "to"):
                    v = v.to(self.d.device)
                    if k == "audio_values":
                        v = v.to(torch.bfloat16)
                inputs[k] = v
            if "audio_batch_size" not in inputs:
                inputs["audio_batch_size"] = torch.tensor([inputs["audio_values"].shape[0]], device=self.d.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer,
                  pad_token_id=self.tok.eos_token_id)
        if getattr(self, "fast", False):
            kw["cache_implementation"] = "static"
        if stop_event is not None:
            class _Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **k):
                    return stop_event.is_set()
            kw["stopping_criteria"] = StoppingCriteriaList([_Stop()])
        if gen_kw:
            kw.update(gen_kw)
        fut = self.gpu.submit(lambda: gen(**kw))  # generate on the GPU thread, stream it from here
        for piece in streamer:
            if piece:
                yield piece, time.perf_counter() - t0
        fut.result()


def transcribe(ears, audio, max_new_tokens=80):
    """User-turn caption. Prompt tested 2026-11-09: no system message, audio first, then the instruction (a system
    message or 'The user said:' framing made it answer the question or return nothing)."""
    import torch
    from duplexjev.decider import load_audio, SAMPLE_RATE
    msgs = [{"role": "user", "content": "<|audio|>\n\nTranscribe the speech above verbatim."}]
    prompt = ears.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p = ears.proc(text=prompt, audio=ears.d._align(load_audio(audio)), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inp = {}
    for k, v in p.items():
        if hasattr(v, "to"):
            v = v.to(ears.d.device)
            inp[k] = v.to(torch.bfloat16) if k == "audio_values" else v
    if "audio_batch_size" not in inp:
        inp["audio_batch_size"] = torch.tensor([inp["audio_values"].shape[0]], device=ears.d.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=ears.tok.eos_token_id)
    if getattr(ears, "fast", False):
        kw["cache_implementation"] = "static"
    out = ears.gpu.call(lambda: ears.model.generate(**inp, **kw))  # on the GPU thread (see GpuThread)
    txt = ears.tok.decode(out[0], skip_special_tokens=True)
    txt = txt.split("</think>")[-1]
    return clean_reply(txt).strip().strip('"')


class Chunker:
    """Cut streamed reply text into speakable pieces. The first piece is cut early (a clause of 4+ words at a comma,
    or 9 words at any space) so the voice can start sooner; later pieces are whole sentences."""
    END = re.compile(r"[.!?](\s|$)")
    CLAUSE = re.compile(r"[,;:](\s)")

    def __init__(self, mode=None):
        self.buf, self.n = "", 0
        self.mode = mode or os.environ.get("DJV_CHUNK", "clause")  # "clause" (early first piece) or "sentence"

    def feed(self, piece):
        self.buf += piece
        out = []
        while True:
            m = self.END.search(self.buf)
            if self.n == 0 and not m and self.mode == "clause":
                c = self.CLAUSE.search(self.buf)
                if c and len(self.buf[:c.end()].split()) >= 3:
                    m = c
                elif len(self.buf.split()) >= 8 and self.buf.endswith(" "):
                    out.append(self._take(len(self.buf))); continue
            if not m:
                break
            out.append(self._take(m.end()))
        return [s for s in out if s]

    def _take(self, i):
        s, self.buf = self.buf[:i].strip(), self.buf[i:]
        self.n += 1
        return clean_reply(s, cap=self.n == 1)

    def flush(self):
        s = clean_reply(self.buf, cap=self.n == 0); self.buf = ""
        return [s] if s else []


def clean_reply(text, cap=True):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("\u2014", ", ").replace("\u2013", ", ").replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub("[\U0001F000-\U0001FAFF\u2600-\u27bf\ufe0f\u200d]", "", text)  # emojis are not speech
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text).replace("\u2026", "...")         # markdown emphasis
    text = re.sub(r"\s+,", ",", re.sub(r"[ \t]+", " ", text)).strip()
    text = re.sub(r"\b(I'?m|I am|my name is|call me)\s+Soph\w*\s*[.!]\s*", "", text, flags=re.I)
    text = re.sub(r"\b(I'?m|I am)\s+Soph\w*,?\s*(?=(a|an|the|your|here|just)\b)", "I'm ", text, flags=re.I)
    text = re.sub(r"\b(I'?m|I am)\s+Soph\w*,?\s*", "", text, flags=re.I)
    text = re.sub(r"\bSoph(on|i|ia)\b[,]?\s*", "", text)
    return text[:1].upper() + text[1:] if (text and cap) else text


# ------------------------------------------------------------------------------------------------ voice
class Voice:
    """Resident Phonon 31m worker (CPU) in its own venv; requests are handled in order."""

    def __init__(self, on_line=None):
        self.on_line = None
        env = dict(os.environ)
        env["HF_HOME"] = PHONON_HF_HOME
        env.pop("HF_HUB_OFFLINE", None)
        t0 = time.perf_counter()
        self.p = subprocess.Popen([PHONON_PY, "-I", PHONON_WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1, env=env, creationflags=NOWIN | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        ready = self.q.get(timeout=180)
        self.load_s = time.perf_counter() - t0
        self.sample_rate = ready.get("sample_rate", 24000)
        self.on_line = on_line

    def _read(self):
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                cb = self.on_line
                if cb is not None:
                    cb(d)
                else:
                    self.q.put(d)

    def say_async(self, text, out):
        self.p.stdin.write(json.dumps({"text": text, "out": out}) + "\n")
        self.p.stdin.flush()

    def stream_async(self, text, rid):
        self.p.stdin.write(json.dumps({"text": text, "pcm": True, "id": rid}) + "\n")
        self.p.stdin.flush()

    def cancel(self):
        self.p.stdin.write(json.dumps({"cancel": True}) + "\n")
        self.p.stdin.flush()

    def result(self, timeout=120):
        return self.q.get(timeout=timeout)

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()
