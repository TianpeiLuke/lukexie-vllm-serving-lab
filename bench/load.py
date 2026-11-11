"""Load the vLLM server at a fixed concurrency and append one results row per run.

uv run --env-file .env python -m bench.load --env L4 --concurrency 1 4 16 64 [--setting baseline]

--env names the GPU the server runs on (check: modal container exec <id> -- nvidia-smi).
"""

import argparse
import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from openai import OpenAI

from client.timing import MODEL, RequestTiming, make_prompt, stream_and_time, summarize
from results.record import append_result

BASE_URL = os.environ.get("VLLM_BASE_URL", "https://unidoctor--vllm-serving-lab-server.us-east.modal.direct")


@dataclass
class BenchResult:
    concurrency: int
    num_prompts: int
    failed: int
    duration_s: float      # first request sent -> last request finished
    timings: list[RequestTiming]


def run_load(client: OpenAI, concurrency: int, num_prompts: int, input_len: int = 500,
             output_len: int = 200, seed: int = 0, output_lens: list[int] | None = None) -> BenchResult:
    """Send num_prompts requests with at most concurrency in flight; count failures instead of raising.

    output_lens, if given, sets each request's output length and overrides output_len.
    """
    def one(i):
        n_out = output_lens[i] if output_lens else output_len
        try:
            return stream_and_time(client, make_prompt(input_len, seed + i), n_out, ignore_eos=True)
        except Exception as err:  # a failed request is a result (M7), not a crash
            print(f"request {i} failed: {type(err).__name__}: {err}", file=sys.stderr)
            return None

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:  # the pool size caps requests in flight
        results = list(pool.map(one, range(num_prompts)))
    duration = time.perf_counter() - start
    timings = [r for r in results if r is not None]
    return BenchResult(concurrency, num_prompts, num_prompts - len(timings), duration, timings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", required=True, help="GPU the server runs on, e.g. L4 or A10G")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--num-prompts", type=int, help="default: 20 at 1 user, else 100")
    parser.add_argument("--input-len", type=int, default=500)
    parser.add_argument("--output-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setting", default="baseline")
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args(argv)

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=600, max_retries=0)
    vllm_version = httpx.get(f"{BASE_URL}/version", timeout=30).json()["version"]
    stream_and_time(client, make_prompt(args.input_len, seed=-1), 16, ignore_eos=True)  # warm-up, discarded
    command = "python -m bench.load " + " ".join(argv if argv is not None else sys.argv[1:])

    for c in args.concurrency:
        n = args.num_prompts or (20 if c == 1 else 100)
        # distinct prompts per run, so prefix caching can't reuse an earlier run's prompts
        result = run_load(client, c, n, args.input_len, args.output_len, seed=args.seed * 1_000_000 + c * 1_000)
        s = summarize(result.timings, result.duration_s)
        append_result({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "env": args.env, "model": MODEL,
            "setting": args.setting, "users": c, "num_prompts": n, "failed": result.failed,
            **{k: round(v, 1) for k, v in s.items() if k != "e2e_p50_ms"},
            "tool": "load.py", "vllm_version": vllm_version, "command": command,
        }, args.out)
        print(f"users {c:3d} | {n} prompts, {result.failed} failed, {result.duration_s:.1f} s | "
              f"TTFT p50 {s['ttft_p50_ms']:.0f} ms p95 {s['ttft_p95_ms']:.0f} ms | "
              f"TPOT p50 {s['tpot_p50_ms']:.1f} ms | {s['output_tok_s']:.1f} output tok/s", flush=True)


if __name__ == "__main__":
    main()
