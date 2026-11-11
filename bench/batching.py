"""Continuous against static batching: the same requests sent two ways, with fixed and with varied output lengths.

uv run --env-file .env modal run -m bench.batching --env A10G [--users 16 --num-prompts 96]

continuous: a closed loop of `users` (bench/load.py); a finished request's slot is refilled at once.
static:     send `users` requests, wait for all of them, then send the next group. vLLM still batches
            continuously, but it never gets a new request while a group runs, so a group lasts as
            long as its longest request: what a static batcher does.
Fixed output lengths are the control (the two should match); varied lengths show the gap.
The client runs in a Modal CPU container, as for M2, so the network adds little.
"""

import datetime
import json
import os
import random
import time

import modal

from server.vllm_modal import vllm_image

BASE_URL = os.environ.get("VLLM_BASE_URL", "https://unidoctor--vllm-serving-lab-server.us-east.modal.direct")
SHORTEST, LONGEST = 16, 400  # varied lengths: uniform, mean 208, close to the sweep's fixed 200
ARMS = (("fixed", "continuous"), ("fixed", "static"), ("varied", "continuous"), ("varied", "static"))

image = vllm_image.add_local_python_source("bench", "client", "results", "server")
app = modal.App("vllm-serving-lab-batching")


def output_lengths(dist: str, n: int, seed: int) -> list[int]:
    """n output lengths: all 200, or uniform in [SHORTEST, LONGEST] with a fixed seed."""
    if dist == "fixed":
        return [200] * n
    rng = random.Random(seed)
    return [rng.randint(SHORTEST, LONGEST) for _ in range(n)]


def predicted_ratio(lens: list[int], users: int) -> float:
    """Static over continuous run time, counting decode steps only.

    A static group takes as many steps as its longest request; continuous batching keeps all
    `users` slots busy, so it takes about sum(lens) / users steps.
    """
    static = sum(max(lens[i:i + users]) for i in range(0, len(lens), users))
    return static / (sum(lens) / users)


def run_mode(client, mode: str, users: int, lens: list[int], seed: int):
    """One run: a closed loop over all requests, or groups of `users` with a wait after each."""
    from bench.load import BenchResult, run_load

    if mode == "continuous":
        return run_load(client, users, len(lens), seed=seed, output_lens=lens)
    start = time.perf_counter()
    timings, failed = [], 0
    for i in range(0, len(lens), users):
        group = lens[i:i + users]
        r = run_load(client, users, len(group), seed=seed + i, output_lens=group)  # returns when all are done
        timings += r.timings
        failed += r.failed
    return BenchResult(users, len(lens), failed, time.perf_counter() - start, timings)


@app.function(image=image, cpu=4, secrets=[modal.Secret.from_name("vllm-api-key")], timeout=30 * 60)
def compare(base_url: str, users: int, num_prompts: int, seed: int) -> list[dict]:
    """Run the four arms one after another; return one summary per arm."""
    from openai import OpenAI

    from client.timing import make_prompt, stream_and_time, summarize

    client = OpenAI(base_url=f"{base_url}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=600, max_retries=0)
    stream_and_time(client, make_prompt(500, seed=-1), 16, ignore_eos=True)  # warm-up, discarded
    out = []
    for k, (dist, mode) in enumerate(ARMS):
        lens = output_lengths(dist, num_prompts, seed)  # both modes of a dist get the same lengths
        # distinct prompts per arm, so prefix caching can't reuse an earlier arm's prompts
        r = run_mode(client, mode, users, lens, seed=seed * 1_000_000 + 900_000 + k * 10_000)
        s = summarize(r.timings, r.duration_s)
        out.append({"dist": dist, "mode": mode, "failed": r.failed, "duration_s": r.duration_s,
                    "slot_busy": sum(t.e2e_s for t in r.timings) / (users * r.duration_s),
                    "predicted_ratio": predicted_ratio(lens, users), **s})
        print("ARM " + json.dumps({"seed": seed, **out[-1]}), flush=True)  # kept in the app logs if the local client drops
    return out


@app.local_entrypoint()
def main(env: str, users: int = 16, num_prompts: int = 96, seed: int = 0) -> None:
    import httpx

    from client.chat import wait_until_healthy
    from client.timing import MODEL
    from results.record import append_result

    wait_until_healthy()
    vllm_version = httpx.get(f"{BASE_URL}/version", timeout=30).json()["version"]
    arms = compare.remote(BASE_URL, users, num_prompts, seed)
    command = f"modal run -m bench.batching --env {env} --users {users} --num-prompts {num_prompts}"
    for a in arms:
        append_result({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "env": env, "model": MODEL,
            "setting": f"{a['mode']} batching, {a['dist']} output", "users": users, "num_prompts": num_prompts,
            "failed": a["failed"], **{k: round(a[k], 1) for k in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
                                                                   "output_tok_s")},
            "tool": "batching.py", "vllm_version": vllm_version, "command": command,
        })
        print(f"{a['dist']:6s} {a['mode']:10s} | {a['failed']} failed, {a['duration_s']:.1f} s | "
              f"{a['output_tok_s']:.1f} tok/s | slots busy {a['slot_busy']:.0%} | "
              f"TTFT p50 {a['ttft_p50_ms']:.0f} p95 {a['ttft_p95_ms']:.0f} ms | TPOT p50 {a['tpot_p50_ms']:.1f} ms | "
              f"E2E p50 {a['e2e_p50_ms'] / 1000:.2f} s")
    by = {(a["dist"], a["mode"]): a for a in arms}
    for dist in ("fixed", "varied"):
        c, s = by[(dist, "continuous")], by[(dist, "static")]
        print(f"{dist}: continuous / static throughput = {c['output_tok_s'] / s['output_tok_s']:.2f}, "
              f"decode-step model predicts {s['predicted_ratio']:.2f}")
