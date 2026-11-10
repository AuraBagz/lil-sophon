# AGENTS.md

Instructions for an AI coding agent asked to install, run or modify lil-sophon. Follow the steps in order. Each step has a check. Do not skip a check, and stop and report if one fails.

## What this is

A local real-time voice agent. A browser page streams microphone audio to `bridge.py` on 127.0.0.1. The bridge uses three models:

| Role | Model | Exact source | Runs on | Required |
|---|---|---|---|---|
| Ears (end of turn + transcript) | DuplexJev-4B-Para v1.4 | Hugging Face repo `adventists-ai/DuplexJev-4B-Para` | GPU, about 9 GB | yes |
| Brain (reply text) | Nemotron 3.5 Bolt E4B, file `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` | Hugging Face repo `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` | GPU, about 4 GB, served by llama.cpp `llama-server` | no (falls back to the ears model) |
| Voice | Phonon 31m, Python package `ptts==0.4.0` | weights `gradium/phonon-31m-en` and `gradium/phonon-voices`, fetched by `ptts` | CPU | yes |
| Speech detection | Silero VAD, package `silero-vad==6.2.3` | bundled in the package | CPU | yes |

Do not substitute other models unless the user asks. Do not use DuplexJev-4B-Para-LiveTurn, DuplexJev-4B, 4B-Turn or 4B-Stress: the code expects 4B-Para. Sophon Realtime (32B) is a different model and is not used here.

## Licence facts you must tell the user

- DuplexJev-4B-Para weights are CC BY-NC 4.0: non-commercial only. If the user wants commercial use, say so before installing and stop.
- Never commit, upload or redistribute model weights. `hf/` is git-ignored. Keep it that way.

## Requirements to check first

1. OS: Windows 11 is the only tested system. On Linux or macOS, tell the user it is untested before continuing. Skip `triton-windows` there (the requirement is already marked Windows only) and start with `python bridge.py`, not `start-bridge.ps1`.
2. GPU: `nvidia-smi` must work and show at least 16 GB total memory. With less than 14 GB free, use the ears-only brain (`DJV_BRAIN=duplexjev`) and do not start Bolt.
3. Python 3.12: `py -3.12 --version` (Windows) or `python3.12 --version`.
4. Disk: about 15 GB free on the drive holding the repo. All caches and weights stay inside the repo folder (`hf/`, `cache/`, `tmp/`).

## Install

Working directory is the repo root for every command.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps "duplexjev @ git+https://github.com/adventists-ai/duplexjev"
```

Check: `python -c "import torch, duplexjev, ptts, silero_vad, aiohttp; print(torch.cuda.is_available())"` prints `True`.

The cu128 wheel index is for RTX 50 series cards. For older cards pick the CUDA build that matches the installed driver.

## Download models

Ears (about 8 GB). The folder name and location matter:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('adventists-ai/DuplexJev-4B-Para', local_dir='hf/models/DuplexJev-4B-Para')"
```

Check: `hf/models/DuplexJev-4B-Para/config.json` exists.

Voice. The bridge runs with `HF_HUB_OFFLINE=1`, so the voice weights must be fetched before the first run, into the repo's `hf` folder:

```powershell
$env:HF_HOME = "$PWD\hf"
python -c "import ptts; ptts.TTS(lang='en').synth('Ready.')"
```

Check: the command exits without error.

Brain (optional). Download the one GGUF file, then start the server and leave it running:

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB', 'Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf', local_dir='hf/models/bolt'))"
llama-server -m hf\models\bolt\Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --host 127.0.0.1 --port 8858 -ngl 999 -c 8192 -np 1 --jinja
```

`llama-server` comes from llama.cpp (github.com/ggml-org/llama.cpp releases, the CUDA build). If it is not installed, ask the user before installing it, or skip the brain.

Check: `curl http://127.0.0.1:8858/v1/models` returns JSON.

The bridge sends `chat_template_kwargs: {"enable_thinking": false}` with each request. If you point `DJV_BRAIN_URL` at a different OpenAI-compatible server, it must stream and should answer in well under 200 ms to first token, or the latency numbers in the README will not hold.

## Run

```powershell
python bridge.py --skip-gate
```

`--skip-gate` is needed when the Bolt server is already holding GPU memory. Without it the bridge refuses to start if more than 6 GB of GPU memory is in use or less than 12 GB of RAM is free.

Check 1: the log prints `bridge on http://127.0.0.1:8857/`. It also prints `brain: Bolt at ...` or `brain: Bolt NOT reachable, replies fall back to DuplexJev`.

Check 2, a full spoken turn with no microphone:

```powershell
python smoke.py 8857 bolt
python smoke.py 8857 duplex
```

Each must print a `{"type": "latency", ...}` event and `agent audio bytes` greater than 0. If `agent audio bytes 0`, read the bridge output for a voice worker error (usually the voice weights were not downloaded: redo the voice step).

Then tell the user to open http://127.0.0.1:8857/ in Chrome, click the microphone and talk. The page must be opened from that address, not from the file, so the browser grants the microphone.

## Wiring summary

- `bridge.py`: HTTP and WebSocket server on 127.0.0.1:8857. Serves `ui/`, handles `/ws` and `POST /rec`. Owns turn detection, barge-in, chunking and logs.
- `djv.py`: model loading, the end-of-turn question, transcript, reply generation, the brain client and the voice worker client. All paths and settings are read here from environment variables or `config.local.json`.
- `phonon_worker.py`: the voice, a child process driven over stdin and stdout.
- `ui/c.html`: the whole page. All protocol handling is in the `Bridge` object; visuals only receive `setState`, `setLevel`, `pushCaption`, `setLatency`.
- `docs/PROTOCOL.md`: the WebSocket message format. Read it before changing either side.
- `smoke.py`, `test_client.py`: drive the bridge with the synthetic clips in `clips/`. `python test_client.py --turns20` runs a 20-turn session and reports latency.
- `out/`: logs, transcripts and recordings. Git-ignored. May contain the user's voice. Never commit it.

## Rules for changes

- Keep the server bound to 127.0.0.1. Do not add a public listener or a tunnel.
- Every model call must go through the single GPU thread in `djv.py` (`GpuThread`). Calling the compiled model from another thread crashes.
- After editing Python: `python -m py_compile bridge.py djv.py`, then start the bridge on a spare port (`python bridge.py --port 8859 --skip-gate`) and run `python smoke.py 8859 bolt` before touching a running instance.
- When you change latency-related behaviour, report end of speech to first audio as median and worst over at least 20 turns, measured from the last voiced frame. Do not quote a number you did not measure.
- Run browser tests headless with a fake media device. Do not capture the user's real microphone in automated tests.

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Page says "mic only, no agent" | Bridge not running or still starting | Start it, wait for the `bridge on` line, reload |
| Agent cuts itself off | It hears its own voice from the speakers | Headphones, or `?barge=off` |
| Replies are slow from the first turn | Bolt server not reachable, or running on CPU | Check `curl http://127.0.0.1:8858/v1/models` and that `-ngl 999` was passed |
| First reply after start takes many seconds | One-time compile | Wait; later starts use the cache in `cache/` |
| Bridge exits with code 3 at start | GPU or RAM gate | `--skip-gate`, or free memory |
| Recording does not save | ffmpeg missing | Install ffmpeg or set `DJV_FFMPEG` |
