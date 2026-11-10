# P1 Specification: vLLM Serving Lab

Status as of 2026-11-10. The plan and the session history live in the Obsidian vault (`Interview_Preparation/Career_Coaching/AI_Project/P1_vLLM_Serving/`); this file is the contract for the code in this repo, and [`PLAN.md`](PLAN.md) is the step-by-step order.

## 1. Goal

**Serve an LLM with vLLM and explain every number it produces.** By the end of P1:

- one command starts a vLLM server for `Qwen/Qwen3-8B` on one Modal L4 GPU, behind an OpenAI-compatible API with an API key;
- my own client and `vllm bench serve` measure time to first token (TTFT), time per output token (TPOT) and throughput, and agree with each other;
- a results table shows how the numbers change with concurrency (1, 4, 16, 64 users) and with one engine setting at a time;
- each change has a one-sentence cause, explained with the KV cache, batching or memory bandwidth.

Out of scope: multi-GPU, LoRA serving, speculative decoding, Kubernetes, building vLLM from source.

**MVP: start the server and chat with it** (metric M1). Reached 2026-11-10 and repeated the same day:

```bash
uv run modal deploy server/vllm_modal.py
uv run python -c "import modal; modal.Server.from_name('vllm-serving-lab', 'Server').update_autoscaler(min_containers=1)"
uv run --env-file .env python client/chat.py
uv run modal app stop vllm-serving-lab
```

The second line forces a container to start now. On 2026-11-10 no container started for minutes after two deploys because Modal had no free L4 (the app log said "waiting to be scheduled on a GPU_L4 worker"). With `GPU = ["L4", "A10"]`, a container started on demand and landed on an A10G. The `target_concurrency=0.0` printed by the deprecated `update_autoscaler()` is how Modal stores "unset"; `server.autoscaler.settings()` returns `None`. Everything after the MVP measures and explains the server.

## 2. Success Metrics

P1 is done when every row passes. Measure on the fixed baseline workload (section 5) unless the row says otherwise.

| # | Metric | Pass when | Status 2026-11-10 |
| --- | --- | --- | --- |
| M1 | Server answers | `/health` is 200, `/v1/models` lists `Qwen/Qwen3-8B` with the key, and a streamed chat returns text | **pass**: `client/chat.py`, health 200 at 14:05 |
| M2 | Client agrees with the benchmark tool | at 1 user, TTFT p50 and TPOT p50 from `bench/load.py` are within 10% of `vllm bench serve` | open |
| M3 | Batching gain | output tokens/s at 16 users ≥ 4 × the 1-user number | open |
| M4 | KV cache explained | the estimate is within 20% of the logged `GPU KV cache size`, and the gap is explained | **logged 20,336 tokens**; first estimate 25,000 (23% high, GB vs GiB); the corrected estimate of 22,900 is 13% high: pass after the correction |
| M5 | Where it queues | during the 64-user run, `/metrics` shows `num_requests_waiting` > 0 and the KV cache usage, explained in one sentence | open |
| M6 | One-setting experiments | 3 experiments, each with a baseline row, a changed row and a one-line cause that matches the numbers | open |
| M7 | Reliability | every benchmark row has 0 failed requests | open |
| M8 | Reproducible | someone else can rerun any row from the README and the command stored in `results/results.csv` | open |
| M9 | Cost | GPU time and dollars logged per session; P1 total under \$10 | S1 about 6 L4 minutes; S2 container up since 14:01 (with `min_containers=1`, it bills until stopped) |
| M10 | Secrets | the key appears in no file in git, no log and no command line; `grep -rIlwE "[0-9a-f]{48}" --exclude-dir=.venv --exclude-dir=.git --exclude=.env .` finds nothing (`-w`, so the 64-hex hashes in `uv.lock` do not match) | **pass** |
| M11 | Code checks | `uv run pytest` passes; pure functions have unit tests that need no server | open |

## 3. What to Implement

| # | File | What | Used for |
| --- | --- | --- | --- |
| I1 | `server/vllm_modal.py` | experiments selected by name, without editing the arguments by hand | M6 |
| I2 | `client/timing.py` | time one streamed request; make prompts of a given length; summarize timings | M2, M3 |
| I3 | `bench/load.py` | my own load generator at a fixed concurrency (AsyncOpenAI) | M2, M3, M7 |
| I4 | `bench/vllm_bench_modal.py` | run `vllm bench serve` in a CPU container on Modal, since vLLM is not installed on the Mac | M2 |
| I5 | `bench/metrics.py` | scrape `/metrics` during a run | M5 |
| I6 | `results/record.py` | append one row per run to `results/results.csv` | M8 |
| I7 | `tests/` | unit tests for the pure functions in I2, I5 and I6 | M11 |
| I8 | `README.md` | how to run each step, the results table, findings, cost | M8 |

## 4. Signatures

### What We Have

`server/vllm_modal.py` (deployed 2026-11-10):

```python
MODEL_NAME: str = "Qwen/Qwen3-8B"
MODEL_REVISION: str                      # pinned commit of the weights
GPU: list[str] = ["L4", "A10"]           # first free type; record which one ran (L4 300 GB/s, A10G 600 GB/s)
VLLM_PORT: int = 8000
ENGINE_ARGS: list[str]                   # ["--max-model-len", "8192", "--gpu-memory-utilization", "0.90"]

vllm_image: modal.Image                  # CUDA 12.9 + vllm==0.21.0
hf_cache_vol, vllm_cache_vol: modal.Volume   # weights and compile cache
app = modal.App("vllm-serving-lab")

@app.server(gpu=GPU, port=VLLM_PORT, max_containers=1, scaledown_window=900, startup_timeout=1200, ...)
class Server:
    @modal.enter()
    def start(self) -> None: ...         # Popen(["vllm", "serve", ...]); key only via VLLM_API_KEY
    @modal.exit()
    def stop(self) -> None: ...

@app.local_entrypoint()
def smoke_test() -> None: ...            # modal run: wait for /health, list models
```

`client/chat.py`:

```python
BASE_URL: str                            # VLLM_BASE_URL or the deployed Modal URL
MODEL: str = "Qwen/Qwen3-8B"
client: OpenAI

def ask(messages: list[dict], max_tokens: int = 512) -> str: ...
    # stream one answer, print it, then TTFT, tokens, TPOT, prompt tokens; return the text
def wait_until_healthy(timeout_s: float = 1200, poll_s: float = 10) -> None: ...
    # 503 and transport errors mean not ready; raise TimeoutError at the deadline
def read_message(prompt: str = "\nyou> ") -> str: ...
    # one typed line, or a whole pasted block (blank lines included) as one message;
    # the live chat quits on /quit or Ctrl-D, and skips empty messages
```

### What We Want

**I1. `server/vllm_modal.py`: experiments by name**

```python
BASE_ARGS: list[str]                     # today's ENGINE_ARGS
EXPERIMENTS: dict[str, list[str]] = {    # extra or overriding args, one setting each
    "baseline": [],
    "e1_max_num_seqs_16": ["--max-num-seqs", "16"],
    "e2_no_prefix_caching": ["--no-enable-prefix-caching"],
    "e3_max_model_len_2048": ["--max-model-len", "2048"],
    "e4_gpu_mem_0.85": ["--gpu-memory-utilization", "0.85"],   # 0.6 would not fit the 15.27 GiB of weights
}
EXPERIMENT: str = os.environ.get("P1_EXPERIMENT", "baseline")
ENGINE_ARGS: list[str] = BASE_ARGS + EXPERIMENTS[EXPERIMENT]   # vLLM takes the last value of a repeated flag
```

- `P1_EXPERIMENT` must also be set inside the container, for example with `vllm_image.env({"P1_EXPERIMENT": EXPERIMENT})`, so both imports agree. Unknown names fail at deploy with a `KeyError`.
- Usage: `P1_EXPERIMENT=e1_max_num_seqs_16 uv run modal deploy server/vllm_modal.py`.
- To check before relying on it: the flag spellings against `vllm serve --help` for 0.21.0, and whether a repeated flag takes the last value.

**I2. `client/timing.py`**

```python
@dataclass
class RequestTiming:
    ttft_s: float          # request sent → first content token
    tpot_s: float          # (last token time − first token time) / (output_tokens − 1)
    e2e_s: float           # request sent → stream finished
    prompt_tokens: int     # from the server's usage, not estimated
    output_tokens: int

def ttft_tpot(start: float, token_times: list[float]) -> tuple[float, float]: ...
    # pure; one token → tpot 0.0; no tokens → ValueError

def stream_and_time(client: OpenAI, messages: list[dict], max_tokens: int,
                    model: str = MODEL, ignore_eos: bool = False) -> RequestTiming: ...
    # stream=True, stream_options={"include_usage": True}; thinking off;
    # ignore_eos=True (vLLM extra_body) forces exactly max_tokens output for benchmarks

def make_prompt(n_tokens: int, seed: int = 0) -> list[dict]: ...
    # one user message whose rendered prompt is n_tokens ± 5% (checked against usage.prompt_tokens);
    # deterministic for a given seed

def percentile(values: list[float], q: float) -> float: ...
    # q in [0, 100], linear interpolation like numpy.percentile; empty → ValueError

def summarize(timings: list[RequestTiming], duration_s: float) -> dict[str, float]: ...
    # keys: ttft_p50_ms, ttft_p95_ms, tpot_p50_ms, e2e_p50_ms, output_tok_s (= total output tokens / duration_s)
```

`client/chat.py` keeps its signatures. `ask` may call `stream_and_time` internally.

**I3. `bench/load.py`**

```python
@dataclass
class BenchResult:
    concurrency: int
    num_prompts: int
    failed: int
    duration_s: float      # first request sent → last request finished
    timings: list[RequestTiming]

async def run_load(base_url: str, api_key: str, concurrency: int, num_prompts: int,
                   input_len: int = 500, output_len: int = 200, seed: int = 0) -> BenchResult: ...
    # AsyncOpenAI; an asyncio.Semaphore(concurrency) caps requests in flight;
    # prompts from make_prompt(input_len, seed + i); a failed request is counted, not raised

def main(argv: list[str] | None = None) -> None: ...
    # uv run --env-file .env python bench/load.py --concurrency 1 4 16 64 [--setting baseline]
    # one warm-up request first (discarded), then one run per concurrency; appends rows with record.append_result
```

**I4. `bench/vllm_bench_modal.py`**

```python
@app.function(image=vllm_image, cpu=4, secrets=[modal.Secret.from_name("vllm-api-key")], timeout=1800)
def run_vllm_bench(base_url: str, concurrency: int, num_prompts: int,
                   input_len: int = 500, output_len: int = 200, seed: int = 0) -> dict: ...
    # runs: vllm bench serve --backend openai-chat --endpoint /v1/chat/completions
    #   --base-url <base_url> --model Qwen/Qwen3-8B --dataset-name random
    #   --random-input-len <input_len> --random-output-len <output_len>
    #   --max-concurrency <concurrency> --num-prompts <num_prompts> --seed <seed>
    #   --save-result --result-filename /tmp/result.json
    # returns the parsed JSON (ttft, tpot, output throughput, failed)

@app.local_entrypoint()
def main(concurrency: int = 1, num_prompts: int = 20) -> None: ...
```

- To check before relying on it: how `vllm bench serve` 0.21.0 sends the API key (an environment variable or a flag) and that it runs on a CPU-only container. Never pass the key as a flag; it would be logged.

**I5. `bench/metrics.py`**

```python
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total")   # names checked on the live server 2026-11-10

def parse_metrics(text: str, names: tuple[str, ...] = GAUGES) -> dict[str, float]: ...
    # pure; Prometheus text format; sums a metric over its label sets; a missing name → 0.0

def scrape(base_url: str) -> dict[str, float]: ...        # GET /metrics (no key needed)

async def poll(base_url: str, interval_s: float, stop: asyncio.Event) -> list[dict[str, float]]: ...
    # one sample per interval with a "t" key (seconds since the start), until stop is set
```

**I6. `results/record.py`**

```python
COLUMNS = ("timestamp", "env", "model", "setting", "users", "num_prompts", "failed",
           "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "output_tok_s",
           "tool", "vllm_version", "command")
           # tool: "load.py" or "vllm bench serve"

def append_result(row: dict, path: str = "results/results.csv") -> None: ...
    # writes the header if the file is new; a missing or unknown column → ValueError
```

**I7. `tests/`**

| Test file | Checks, no server needed |
| --- | --- |
| `tests/test_timing.py` | `ttft_tpot` on hand-made times (one token, many tokens, none); `percentile` against `numpy.percentile`; `summarize` on three fake timings |
| `tests/test_metrics.py` | `parse_metrics` on a saved `/metrics` sample with two label sets and a missing name |
| `tests/test_record.py` | `append_result` writes the header once, appends rows, rejects bad columns (with `tmp_path`) |

Live checks (need the server, not run by `pytest`): `make_prompt(500)` gives 475–525 prompt tokens; `stream_and_time(..., max_tokens=200, ignore_eos=True)` returns exactly 200 output tokens.

## 5. Baseline Workload

Random prompts, **500 input tokens, 200 output tokens** (`ignore_eos`), 100 prompts per run (20 at 1 user), all sent at once with the concurrency capping requests in flight, seed 0, thinking off. Only the setting under test changes between rows. Discard one warm-up run after each start.

## 6. Run Rules

- Before each GPU session: the launch checklist in the vault (`Checklist - Launch an LLM Server on a Cloud GPU with vLLM and Modal`).
- After each session: `uv run modal app stop vllm-serving-lab`, then `uv run modal container list` shows nothing; log the minutes and cost.
- `.env` and `results/logs/*.log` stay out of git; `results/results.csv` is committed.
