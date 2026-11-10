"""Drive bridge.py like a browser would: stream clips as real-time 16 kHz Int16 mic frames, keep the mic open
(low noise between turns), collect events, measure end of speech to first audio, and test one barge-in.

Run: python test_client.py [--port 8857] [--echo 0.2] [--out result.json]
Writes out/bridge/test_client.json (or --out)
"""
import asyncio
import os
import json
import statistics
import sys
import time

import aiohttp
import numpy as np
from duplexjev.decider import load_audio

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8857
ECHO = float(sys.argv[sys.argv.index("--echo") + 1]) if "--echo" in sys.argv else 0.0  # loop agent audio back into the mic at this gain
BRAINQ = sys.argv[sys.argv.index("--brain") + 1] if "--brain" in sys.argv else "bolt"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
CL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips") + os.sep
TURNS = ["q1_cook", "q2_coveredcall", "q3_moon", "q4_stress", "q5_australia", "t1_pause_full", "b1_coffee",
         "b2_sleep", "b3_python", "b4_joke", "b5_rain", "b6_plants", "b7_long"]
if "--bargeonly" in sys.argv:
    TURNS = ["q1_cook", "q3_moon", "q4_stress"]
elif "--turns20" in sys.argv:  # 20-turn session: add 7 follow-up turns from clips/conv
    TURNS += ["conv/v%02d" % i for i in range(1, 8)]
FRAME = 320
rng = np.random.default_rng(0)


def pcm(x):
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def noise(n):
    return rng.normal(0, 0.0015, n).astype(np.float32)  # about -56 dBFS room tone


async def main():
    events, audio_bytes = [], [0]
    echo_q = [np.zeros(0, np.float32)]
    state = {"v": None}
    async with aiohttp.ClientSession() as cs:
        async with cs.ws_connect(f"ws://127.0.0.1:{PORT}/ws?brain={BRAINQ}", max_msg_size=8 * 2**20) as ws:
            async def rx():
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.BINARY:
                        audio_bytes[0] += len(m.data)
                        if ECHO:
                            y = np.frombuffer(m.data, dtype='<i2').astype(np.float32) / 32768.0
                            y = np.interp(np.arange(0, len(y), 1.5), np.arange(len(y)), y).astype(np.float32)
                            echo_q[0] = np.concatenate([echo_q[0], y * ECHO])
                    elif m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data); d["_t"] = time.perf_counter(); events.append(d)
                        if d["type"] == "state":
                            state["v"] = d["value"]
                        if d["type"] in ("latency", "caption", "turn", "interrupted", "error", "duck"):
                            print(json.dumps({k: v for k, v in d.items() if k != "_t"}), flush=True)
            rxt = asyncio.ensure_future(rx())
            t_next = time.perf_counter()

            async def send(x):
                nonlocal t_next
                for i in range(0, len(x), FRAME):
                    fr = x[i:i + FRAME].copy()
                    if ECHO and len(echo_q[0]):
                        e = echo_q[0][:len(fr)]; echo_q[0] = echo_q[0][len(fr):]
                        fr[:len(e)] += e
                    await ws.send_bytes(pcm(fr))
                    t_next += FRAME / 16000
                    await asyncio.sleep(max(0.0, t_next - time.perf_counter()))

            await send(noise(16000))
            results = []
            for name in TURNS:
                x = load_audio(CL + name + ".wav")
                n0 = len(events)
                await send(x)
                # keep the mic open with room tone until the agent finished speaking (or 25 s)
                t_end = time.perf_counter() + 25
                spoke = False
                while time.perf_counter() < t_end:
                    await send(noise(FRAME * 5))
                    if state["v"] == "speaking":
                        spoke = True
                    if spoke and state["v"] in ("idle", "listening"):
                        break
                await send(noise(8000))
                lat = [e for e in events[n0:] if e["type"] == "latency"]
                cap = [e for e in events[n0:] if e["type"] == "caption"]
                results.append({"clip": name, "latency_ms": lat[0]["ms"] if lat else None,
                                "parts": lat[0]["parts"] if lat else None,
                                "agent": next((c["text"] for c in cap if c["who"] == "agent" and c["final"]), None),
                                "user": next((c["text"] for c in cap if c["who"] == "user"), None)})
            # backchannel while she talks: "Mm-hmm, yeah." 1.0 s into her reply should NOT cut her off
            n0 = len(events)
            await send(load_audio(CL + "q2_coveredcall.wav"))
            t_end = time.perf_counter() + 15
            while state["v"] != "speaking" and time.perf_counter() < t_end:
                await send(noise(FRAME * 5))
            await send(noise(16000))
            await send(load_audio(CL + "t4_backchannel.wav"))
            t_end = time.perf_counter() + 20
            while state["v"] == "speaking" and time.perf_counter() < t_end:
                await send(noise(FRAME * 5))
            await send(noise(8000))
            seg = events[n0:]
            backchannel = {"interrupted": any(e["type"] == "interrupted" for e in seg),
                           "ducked": any(e["type"] == "duck" for e in seg)}
            # barge-in: ask a long question, start talking 1.2 s after the agent starts speaking
            n0 = len(events)
            await send(load_audio(CL + "b3_python.wav"))
            t_end = time.perf_counter() + 15
            while state["v"] != "speaking" and time.perf_counter() < t_end:
                await send(noise(FRAME * 5))
            t_sp = time.perf_counter()
            await send(noise(int(16000 * 1.2)))
            t_barge = time.perf_counter()
            await send(load_audio(CL + "q5_australia.wav"))
            await send(noise(16000 * 6))
            seg = events[n0:]
            intr = next((e for e in seg if e["type"] == "interrupted"), None)
            barge = {"interrupted": bool(intr), "by": intr and intr.get("by"),
                     "cut_ms_after_user_start": round((intr["_t"] - t_barge) * 1000) if intr else None,
                     "next_reply": [c["text"] for c in seg if c["type"] == "caption" and c["who"] == "agent" and c["final"]]}
            await ws.close()
            rxt.cancel()
    lats = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    summary = {"turns": len(results), "replied": len(lats), "median_ms": statistics.median(lats) if lats else None,
               "worst_ms": max(lats) if lats else None, "best_ms": min(lats) if lats else None,
               "mean_ms": round(statistics.mean(lats)) if lats else None, "barge_in": barge, "backchannel": backchannel,
               "agent_audio_bytes": audio_bytes[0]}
    json.dump({"summary": summary, "results": results}, open(OUT or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bridge", "test_client.json"), "w"), indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


asyncio.run(main())
