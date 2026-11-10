# lil-sophon

Real-time voice AI on one consumer GPU. You talk, it answers out loud, typically a bit over half a second after you stop. Everything runs on your own PC. Nothing goes to the cloud.

[Sophon Realtime](https://github.com/adventists-ai/sophon-realtime) is a 32B voice model that needs an 80 GB GPU (two for its end-to-end build). lil-sophon is not that model made smaller. It is a small pipeline built around the same team's 4B hearing model, [DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para), with a separate brain and voice, and it fits on a single RTX 5090.

**If you are an AI agent setting this up for someone, read [AGENTS.md](AGENTS.md). It has the exact models, commands and checks.**

## How it works

```
your mic (browser)
   |  16 kHz audio over a local WebSocket
   v
ears    DuplexJev-4B-Para (GPU)   decides when you have finished talking, and writes down what you said
brain   Nemotron 3.5 Bolt E4B (GPU, via llama.cpp)   writes the reply
        or DuplexJev-4B-Para itself, straight from the audio (faster, less witty)
voice   Phonon 31m (CPU)          speaks the reply clause by clause as it is written
   |  24 kHz audio back over the same socket
   v
your speakers (browser)
```

The page is a single HTML file. It shows your voice and the agent's voice on a scrolling tape, and the measured gap between them as the big number: milliseconds from the end of your speech to the first sound of the answer.

## Measured numbers

All on one machine: RTX 5090 (32 GB), Windows 11, 31 GB RAM, Chrome, 2026-11-09. Time is from the last voiced frame of the user's speech to the first audio chunk of the reply leaving the server.

| Brain | Session | Turns | Median | Fastest | Slowest |
|---|---|---|---|---|---|
| DuplexJev-4B-Para (audio in) | live, one person | 25 | 523 ms | 465 ms | 1524 ms |
| Bolt E4B | live, one person | 33 | 586 ms | 441 ms | 1449 ms |
| Bolt E4B | scripted, 20 turns | 20 | 570 ms | | 876 ms |

The slow outliers are turns where the speaker trailed off mid-sentence and the system waited to be sure they had finished.

GPU memory: about 9 GB for DuplexJev-4B-Para plus about 4 GB for Bolt E4B. The voice runs on the CPU. Startup takes about 15 seconds (about 40 to 45 the first time, while it compiles).

These numbers come from one machine and one speaker. Yours will differ.

## What you need

- Windows 11 (the only system this has been run on; see Limitations)
- An NVIDIA GPU with 16 GB or more of memory. Tested on an RTX 5090.
- Python 3.12
- Chrome
- About 15 GB of disk for models
- Optional: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` for the Bolt brain, and ffmpeg for the built-in recorder

## Setup

Run these in PowerShell from the folder where you cloned the repo.

**1. Python environment**

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps "duplexjev @ git+https://github.com/adventists-ai/duplexjev"
```

**2. The hearing model (about 8 GB)**

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('adventists-ai/DuplexJev-4B-Para', local_dir='hf/models/DuplexJev-4B-Para')"
```

**3. The voice (small, downloads once)**

```powershell
$env:HF_HOME = "$PWD\hf"
python -c "import ptts; ptts.TTS(lang='en').synth('Ready.')"
```

**4. The brain (optional, recommended)**

Download `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` from [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) and serve it with llama.cpp in a second terminal:

```powershell
llama-server -m path\to\Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --host 127.0.0.1 --port 8858 -ngl 999 -c 8192 -np 1 --jinja
```

If you skip this step, lil-sophon uses DuplexJev for the replies as well.

**5. Run it**

```powershell
python bridge.py --skip-gate
```

Wait for the line `bridge on http://127.0.0.1:8857/`, then open http://127.0.0.1:8857/ in Chrome, click the microphone, allow mic access, and talk.

To check it without a microphone, in another terminal:

```powershell
python smoke.py 8857 bolt
```

It plays a test question into the server and prints the events. You should see a `latency` line and a non-zero `agent audio bytes`.

## Using the page

| Do this | To |
|---|---|
| Click the mic, or press Space | Start or end a session |
| Just talk | It answers when you finish |
| Talk over it, or press Esc | Interrupt it |
| Press R | Start or stop a recording of the screen with both voices |

Options go on the address, for example `http://127.0.0.1:8857/?brain=duplex&barge=off`:

| Option | Effect |
|---|---|
| `brain=bolt` (default) | Bolt E4B writes the replies |
| `brain=duplex` | DuplexJev writes the replies from the audio. Faster, plainer answers |
| `barge=off` | It always finishes its sentence. Useful when recording |
| `codec=webm` | Record as WebM. More reliable than the default MP4 capture; the server converts it to MP4 |
| `rec=vertical` | Record at 1080x1920 |
| `demo=1` | Play a simulated conversation with no server |

Recordings are saved to `out/rec/`. The recorder needs ffmpeg on your PATH (or set `DJV_FFMPEG`).

## Models and licences

The code in this repository is MIT licensed. **The models are not part of this repository and have their own licences.** You download them yourself.

| Piece | Licence | Note |
|---|---|---|
| [DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) weights | CC BY-NC 4.0 | **Non-commercial use only.** Commercial use needs a licence from its authors. |
| [duplexjev](https://github.com/adventists-ai/duplexjev) Python package | Apache-2.0 | Installed with pip, not copied here |
| [Nemotron 3.5 Bolt E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) | OpenMDW-1.1 | Optional brain |
| Phonon 31m (`ptts`) | Apache-2.0 weights, MIT or Apache-2.0 package | Voice |
| Silero VAD | MIT | Speech detection |

Because the hearing model is non-commercial, lil-sophon as shipped is for personal and research use.

## Limitations

- Run and measured on one Windows 11 machine with one GPU and one speaker.
- Install check, 2026-11-10: setup steps 1, 2, 3 and 5 were followed exactly as written from a fresh clone into an empty folder with a new Python environment and fresh model downloads, and both `smoke.py` runs passed (Bolt 518 and 695 ms, DuplexJev 464 and 482 ms). That was on the same PC, so the GPU driver, Python 3.12 and git were already installed, and an already running Bolt server was reused: step 4 (downloading the GGUF and starting `llama-server`) was only checked as far as the file being available. It has not been installed on a second machine.
- Not tested on Linux or macOS. `start-bridge.ps1` is Windows only; `python bridge.py` is the portable way to start it.
- English only in practice.
- Use headphones. On loud speakers the agent can hear itself, and in testing it then missed a real interruption.
- It gets arithmetic wrong sometimes, and it is a small model: expect short, light answers, not expertise.
- No tool calling.
- One conversation at a time. The newest browser tab takes over.
- The server listens on 127.0.0.1 only. Do not expose it to a network.
- Transcripts of each turn are written to `out/bridge/user.jsonl` on your own disk. Audio of your turns is only saved if you set `DJV_SAVE_UTT=1`.

## Settings

Defaults work. To change them, set environment variables or copy `config.example.json` to `config.local.json`. The full list is at the top of `bridge.py`. The ones you are most likely to need:

| Variable | Default | Meaning |
|---|---|---|
| `DJV_BRAIN` | `bolt` | `bolt` or `duplexjev` |
| `DJV_BRAIN_URL` | `http://127.0.0.1:8858/v1/chat/completions` | Any OpenAI-compatible chat endpoint |
| `DJV_MODEL_DIR` | `hf/models/DuplexJev-4B-Para` | Where the hearing model lives |
| `DJV_PHONON_PY` | the current Python | A Python that has `ptts` installed, if you keep the voice in its own environment |
| `DJV_FFMPEG` | `ffmpeg` on PATH | Used only for recordings |
| `DJV_GATE` | `1` | Set `0` (or pass `--skip-gate`) to start even when the GPU is already in use |

## Credits

The hearing model and the idea come from the [Sophon Realtime and DuplexJev](https://github.com/adventists-ai) team. lil-sophon is an independent project and is not affiliated with or endorsed by them. The page protocol is documented in [docs/PROTOCOL.md](docs/PROTOCOL.md).
