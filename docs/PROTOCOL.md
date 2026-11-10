# Bridge protocol

Server: `bridge.py`, bound to 127.0.0.1 only, port 8857 by default.

- `GET /` redirects to `/c.html`. Static files are served from `ui/`.
- WebSocket at `ws://127.0.0.1:8857/ws`. One live session at a time: the newest connection takes over, and the older socket receives `{"type":"error","text":"replaced by a newer connection"}` and is closed.
- `POST /rec`: raw recording body, header `X-Rec-Name` (letters, digits, `_` and `-`), `Content-Type` `video/mp4` or `video/webm`. Saved under `out/rec/` and converted to H.264 + AAC MP4. Responds with JSON `{ok, path_mp4}`. Needs ffmpeg.

## Query parameters on `/ws`

| Parameter | Values | Meaning |
|---|---|---|
| `brain` | `bolt` (default), `duplex` | Which model writes the reply |
| `barge` | `on` (default), `off` | `off`: the agent always finishes speaking |

The `ready` event reports both.

## Client to server

**Binary frame: microphone audio.** Raw PCM, signed 16-bit little-endian, mono, 16,000 Hz, no header. Any frame length works; 20 ms (320 samples, 640 bytes) is recommended. Send continuously while the mic is open, including while the agent speaks, or barge-in cannot work. Enable the browser's `echoCancellation`, `noiseSuppression` and `autoGainControl`.

**Text frame: JSON control.**

- `{"type":"interrupt"}` (alias `stop`): stop the agent now. The server answers `interrupted`, then a `state`.
- `{"type":"reset"}`: forget the conversation history.
- `{"type":"hello"}`: the server re-sends `ready`.

## Server to client

**Binary frame: agent audio.** Raw PCM, signed 16-bit little-endian, mono, 24,000 Hz, no header. Chunks arrive faster than real time. Queue them back to back for gapless playback.

**Text frame: JSON event**, always with a `type`. Turn-related events carry an integer `turn`.

| Event | Fields | Meaning |
|---|---|---|
| `ready` | `audio_in`, `audio_out`, `model`, `voice`, `brain`, `barge`, `settings` | Sent on connect |
| `state` | `value`: `idle`, `listening`, `thinking`, `speaking` | idle: no speech yet. listening: the user is talking, or the end of turn is being decided. thinking: reply being written. speaking: agent audio is being sent |
| `caption` | `who`: `user` or `agent`, `text`, `final` | Agent text arrives one spoken piece at a time; the last has `final: true` and the whole reply. The user caption is the transcript of the turn |
| `latency` | `ms`, `parts` | Milliseconds from the user's last voiced frame (server clock) to the first agent audio chunk sent. Includes the silence waited before deciding the turn. Excludes browser playback buffering |
| `turn` | `decision`, `p_take_turn`, `class` | Debug: every end-of-turn check |
| `duck` | `on` | The user may be interrupting. `on: true`: lower playback volume now. `on: false`: it was a laugh or a short aside, restore the volume |
| `interrupted` | `by`: `user_speech` or `client` | Stop playback now and drop queued audio |
| `audio_end` | | The last audio chunk of a reply has been sent |
| `error` | `text` | |

A page should not rely on `state: listening` to know the user is speaking: the server confirms speech conservatively to reject noise. Draw the user's level from the local microphone.
