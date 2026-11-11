"""M2: run `vllm bench serve` and bench/load.py from the same CPU container on Modal, then compare.

uv run --env-file .env modal run -m bench.vllm_bench_modal --env A10G [--concurrency 1 --num-prompts 20]

vLLM is not installed on the Mac. Running both tools in one container gives them the same
network path to the server, so their TTFT and TPOT can be compared directly.
--env names the GPU the server runs on (check: modal container exec <id> -- nvidia-smi).
"""

import datetime
import json
import os
import subprocess

import modal

from server.vllm_modal import MODEL_NAME, vllm_image

BASE_URL = os.environ.get("VLLM_BASE_URL", "https://unidoctor--vllm-serving-lab-server.us-east.modal.direct")
RESULT_PATH = "/tmp/vllm_bench.json"

image = vllm_image.add_local_python_source("bench", "client", "results", "server")
app = modal.App("vllm-serving-lab-bench")


def vllm_bench_args(base_url: str, concurrency: int, num_prompts: int, input_len: int,
                    output_len: int, seed: int, out: str = RESULT_PATH) -> list[str]:
    """The `vllm bench serve` command matching one load.py run; the API key goes by OPENAI_API_KEY, never a flag."""
    from client.timing import NO_THINKING, TEMPLATE_TOKENS

    directory, filename = os.path.split(out)
    return [
        "vllm", "bench", "serve", "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
        "--base-url", base_url, "--model", MODEL_NAME, "--dataset-name", "random",
        # random text of n tokens; the chat template adds TEMPLATE_TOKENS, as with make_prompt
        "--random-input-len", str(input_len - TEMPLATE_TOKENS), "--random-output-len", str(output_len),
        "--random-range-ratio", "0", "--ignore-eos", "--extra-body", json.dumps(NO_THINKING),
        "--max-concurrency", str(concurrency), "--num-prompts", str(num_prompts), "--seed", str(seed),
        "--num-warmups", "1", "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
        "--save-result", "--result-dir", directory, "--result-filename", filename,
    ]


def vllm_row(result: dict) -> dict:
    """The results.csv metric columns from a `vllm bench serve` result JSON."""
    return {
        "failed": result["failed"],
        "ttft_p50_ms": round(result["p50_ttft_ms"], 1),
        "ttft_p95_ms": round(result["p95_ttft_ms"], 1),
        "tpot_p50_ms": round(result["p50_tpot_ms"], 1),
        "output_tok_s": round(result["output_throughput"], 1),
    }


@app.function(image=image, cpu=4, secrets=[modal.Secret.from_name("vllm-api-key")], timeout=30 * 60)
def compare(base_url: str, concurrency: int, num_prompts: int, input_len: int, output_len: int,
            seed: int) -> dict:
    """Run both tools against base_url, one after the other; return both summaries."""
    from openai import OpenAI

    from bench.load import run_load
    from client.timing import summarize

    env = {**os.environ, "OPENAI_API_KEY": os.environ["VLLM_API_KEY"]}  # vllm bench reads this name
    subprocess.run(vllm_bench_args(base_url, concurrency, num_prompts, input_len, output_len, seed),
                   check=True, env=env)
    with open(RESULT_PATH) as f:
        vllm = json.load(f)

    client = OpenAI(base_url=f"{base_url}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=600, max_retries=0)
    load = run_load(client, concurrency, num_prompts, input_len, output_len, seed=seed * 1_000_000 + 999_000)
    s = summarize(load.timings, load.duration_s)
    return {
        "vllm": {**vllm_row(vllm), "prompt_tokens": vllm["total_input_tokens"] / max(vllm["completed"], 1)},
        "load": {"failed": load.failed, **{k: round(v, 1) for k, v in s.items() if k != "e2e_p50_ms"},
                 "prompt_tokens": sum(t.prompt_tokens for t in load.timings) / max(len(load.timings), 1)},
    }


@app.local_entrypoint()
def main(env: str, concurrency: int = 1, num_prompts: int = 20, input_len: int = 500,
         output_len: int = 200, seed: int = 0) -> None:
    import httpx

    from client.chat import wait_until_healthy
    from client.timing import MODEL
    from results.record import append_result

    wait_until_healthy()
    vllm_version = httpx.get(f"{BASE_URL}/version", timeout=30).json()["version"]
    out = compare.remote(BASE_URL, concurrency, num_prompts, input_len, output_len, seed)
    command = (f"modal run -m bench.vllm_bench_modal --env {env} --concurrency {concurrency} "
               f"--num-prompts {num_prompts}")
    for tool, key in (("vllm bench serve", "vllm"), ("load.py", "load")):
        r = {k: v for k, v in out[key].items() if k != "prompt_tokens"}
        append_result({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "env": env, "model": MODEL,
            "setting": "baseline", "users": concurrency, "num_prompts": num_prompts, **r,
            "tool": tool, "vllm_version": vllm_version, "command": command,
        })
        print(f"{tool:16s} | {r['failed']} failed | prompt {out[key]['prompt_tokens']:.0f} tok | "
              f"TTFT p50 {r['ttft_p50_ms']:.1f} ms p95 {r['ttft_p95_ms']:.1f} ms | "
              f"TPOT p50 {r['tpot_p50_ms']:.1f} ms | {r['output_tok_s']:.1f} tok/s")
    for metric in ("ttft_p50_ms", "tpot_p50_ms"):
        a, b = out["load"][metric], out["vllm"][metric]
        gap = abs(a - b) / b
        print(f"M2 {metric}: load.py {a:.1f} vs vllm {b:.1f} -> {gap:.1%} {'PASS' if gap <= 0.10 else 'FAIL'}")
