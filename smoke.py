"""One-turn smoke test against a running bridge: python smoke.py [port] [brain] [clip]"""
import asyncio, json, os, sys, time
import aiohttp, numpy as np
from duplexjev.decider import load_audio
port = sys.argv[1] if len(sys.argv) > 1 else "8857"
brain = sys.argv[2] if len(sys.argv) > 2 else "bolt"
clip = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips", "b2_sleep.wav")
async def main():
    async with aiohttp.ClientSession() as cs, cs.ws_connect(f"ws://127.0.0.1:{port}/ws?brain={brain}") as ws:
        x = np.concatenate([np.zeros(8000, np.float32), load_audio(clip), np.random.default_rng(0).normal(0, 0.0015, 16000 * 6).astype(np.float32)])
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        async def tx():
            t = time.perf_counter()
            for i in range(0, len(pcm), 320):
                await ws.send_bytes(pcm[i:i + 320].tobytes()); t += 0.02
                await asyncio.sleep(max(0, t - time.perf_counter()))
        task = asyncio.ensure_future(tx()); nbytes = 0; got = {}
        async for m in ws:
            if m.type == aiohttp.WSMsgType.BINARY:
                nbytes += len(m.data); continue
            d = json.loads(m.data)
            if d["type"] in ("ready", "latency", "caption", "error", "audio_end"):
                print(json.dumps(d)[:300], flush=True)
            if d["type"] == "audio_end":
                break
        task.cancel(); print("agent audio bytes", nbytes)
asyncio.run(main())
