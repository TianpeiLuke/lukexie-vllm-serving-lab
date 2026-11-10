# P1 Implementation Plan: vLLM Serving Lab

Step-by-step plan derived from the success metrics in [`SPEC.md`](SPEC.md). Names in *italics* are notes in the Obsidian vault.

## Summary

> **How this plan is built**
> **Work backwards from the success metrics.** Each of the metrics M1–M11 in the repo's [`SPEC.md`](SPEC.md) needs certain code and certain GPU runs. The steps below are ordered so that:
> 1. code that needs no GPU is written and tested first, with the app stopped;
> 2. each GPU step produces a row that some metric checks;
> 3. every step ends with a **done when** check, and the GPU is on only for steps 3–8.

> **Schedule**
> | Session | Steps | GPU |
> | --- | --- | --- |
> | **S2 rest of today, Tue Nov 10, until 3:00** | 0, 1, 2 | stop the app first; none needed |
> | **S3 Thu Nov 12, 1:00–3:00** | 3, 4, 5, 6 | one start, about 80 min on |
> | **S4 Fri Nov 13, 1:00–3:00** | 7, 8, 9 | three restarts, about 60 min on |
> | spare / next mock slot | 10 | none |
>
> Expected GPU total about 2.5 h on an L4 ≈ \$2, well inside M9's \$10.

## From Metrics to Steps

| Metric | Needs | Step |
| --- | --- | --- |
| M1 server answers | deployed server, chat client | 0 (passed today) |
| M2 client agrees with `vllm bench serve` | `timing.py`, `load.py`, a way to run `vllm bench serve`, both at 1 user | 2, 3, 4, 5 |
| M3 batching gain at 16 users | concurrency sweep with `load.py` | 6 |
| M4 KV cache explained | start-up log line, estimate | done (passes after the GiB correction); restate in the README in step 9 |
| M5 where it queues | `metrics.py`, a 64-user run with polling | 7 |
| M6 three experiments | experiments by name in the server, three runs at 16 users | 8 |
| M7 no failed requests | `failed` counted in every run | 4–8 |
| M8 reproducible | `record.py` with the command per row, README | 1, 9 |
| M9 cost | cost log per session | every GPU step; total in 9 |
| M10 no secrets | key only in `.env` and the Modal secret | checked at every commit |
| M11 tests pass | unit tests for the pure functions | 1, 2, 7 |

## Steps

### Step 0. Freeze Today's Working State (S2, 10 min, no GPU)

- **Do:** stop the app (`uv run modal app stop vllm-serving-lab`, then `uv run modal container list` is empty). Log today's GPU minutes in the cost table of *P1 vLLM Serving - Tasks and Experiments*. Commit `SPEC.md`, `client/chat.py` and the `server/vllm_modal.py` change on a branch `s2-live-endpoint`.
- **Decide:** keep `min_containers=1` out of the code and set it with the override only during sessions. A deploy then never leaves a GPU running by accident.
- **Done when:** zero containers; the M10 grep (in `SPEC.md`) finds nothing; the commit is pushed.

### Step 1. Results File: `results/record.py` and Its Test (S2, 20 min, no GPU)

- **Do:** `append_result(row, path)` with the fixed `COLUMNS`; header on a new file; `ValueError` on a missing or unknown column.
- **Test:** `tests/test_record.py` with `tmp_path`: the header written once, two rows appended, a bad row rejected.
- **Done when:** `uv run pytest tests/test_record.py` passes. Add `pytest` and `numpy` as dev dependencies (`uv add --dev pytest numpy`).
- **Why first:** every later step writes rows; M8 needs the command stored with each one.

### Step 2. Timing Math: `client/timing.py` and Its Test (S2, 30 min, no GPU)

- **Do:** the pure functions first: `ttft_tpot`, `percentile`, `summarize`, and the `RequestTiming` dataclass. Then `make_prompt` (deterministic for a seed) and `stream_and_time`, which can only be checked live.
- **Test:** `tests/test_timing.py`: hand-made token times (none, one, many), `percentile` against `numpy.percentile` at q = 50 and 95, `summarize` on three fake timings with a known throughput.
- **Done when:** `uv run pytest` passes. `client/chat.py` still runs; it may reuse `stream_and_time`.
- **Derive first:** with $n$ output tokens, TPOT = (last token time − first token time)/(n − 1). Why n − 1 and not n?

### Step 3. Live Checks of the Client (S3, 15 min, GPU on)

- **Do:** start the server (launch checklist: deploy, `min_containers=1`, the four live checks). Then run two live checks:
  - `make_prompt(500)` gives 475–525 prompt tokens by the server's `usage`;
  - `stream_and_time(..., max_tokens=200, ignore_eos=True)` returns exactly 200 output tokens.
- **Done when:** both checks pass. If the prompt length is off, calibrate words per token for Qwen's tokenizer once and fix `make_prompt`.
- **Cost note:** each start costs about 1–7 min of waiting for a GPU plus about 4 min of loading. Do all GPU steps of a session in one start.

### Step 4. Own Load Generator at 1 User: `bench/load.py` (S3, 25 min, GPU on)

- **Do:** `run_load` with `AsyncOpenAI` and an `asyncio.Semaphore(concurrency)`; a failed request is counted, not raised; one warm-up request is discarded; `main` appends one row per concurrency with `append_result`.
- **Run:** `--concurrency 1`, 20 prompts.
- **Done when:** the row has 0 failed requests (M7), and TPOT p50 is close to the prediction below.

> **Prediction to check: TPOT at 1 user is bandwidth-bound**
> Each decode step reads all weights once: $16.4\ \text{GB} / 300\ \text{GB/s}$ (L4 memory bandwidth) $\approx 55\ \text{ms}$. Expect a TPOT p50 of about 55–65 ms. A 1-user run then takes about 20 × 200 × 0.06 s ≈ 4 min.

### Step 5. Benchmark Tool and the M2 Comparison (S3, 30 min, GPU on)

- **Do:** run `vllm bench serve` against the endpoint, then compare it with step 4 at 1 user. vLLM is not installed on the Mac, so try these in order:
  1. **`bench/vllm_bench_modal.py`** (I4 in `SPEC.md`): a CPU Modal function in the vLLM image.
  2. **Fallback, no new code:** `uv run modal container exec <container id> vllm bench serve ...` inside the server container.
- **Decide before comparing:** run both tools **from the same place**. The Mac adds network time to every TTFT, and a tool inside Modal does not. So either run `load.py` inside the same Modal function, or compare TPOT (barely affected by the network) and TTFT after subtracting the measured round trip to `/health`.
- **Check first:** how `vllm bench serve` 0.21.0 sends the API key (environment variable, not a flag), and that it runs without a GPU.
- **Done when:** TTFT p50 and TPOT p50 agree within 10%, both rows are in `results/results.csv`, and the session log says which placement was used (**M2**).

### Step 6. Concurrency Sweep (S3, 20 min, GPU on)

- **Run:** `bench/load.py --concurrency 1 4 16 64` on the baseline (100 prompts, 20 at 1 user).
- **Done when:** four rows with 0 failed requests; output tokens/s at 16 users is at least 4 × the 1-user number (**M3**); stop the app; log the cost.

> **Prediction to check: where batching stops helping**
> One request holds about 700 tokens (500 in + 200 out) of the 20,336-token KV cache, so about 29 requests fit at once.
> - **At 16 users,** all requests fit, and each decode step still reads the weights once, now for 16 tokens. The KV reads add about 16 × 700 × 144 KiB ≈ 1.65 GB per step, so a step takes about (16.4 + 1.65) / 300 ≈ 60 ms. The upper bound is about 16 / 0.060 ≈ 270 tokens/s, about 15× the 1-user number; prefill steps and scheduling make the real gain smaller. The pass line is 4×.
> - **At 64 users,** only about 29 run at a time and the rest wait. Throughput stops growing and TTFT jumps. Step 7 measures this.

### Step 7. Where the 64-User Run Queues: `bench/metrics.py` (S4, 30 min; code without GPU, then GPU on)

- **Do:** write `parse_metrics` and its test against a saved `/metrics` sample before starting the GPU. Then `poll` every second during a 64-user run.
- **Done when:**
  - `tests/test_metrics.py` passes;
  - the run shows `vllm:num_requests_waiting` > 0, together with `vllm:num_requests_running` and `vllm:kv_cache_usage_perc`;
  - one sentence explains it with the 29-request estimate (**M5**). Note any `vllm:num_preemptions_total` increase.

### Step 8. Three One-Setting Experiments (S4, 60 min, GPU on)

- **Do:** add `EXPERIMENTS` and `P1_EXPERIMENT` to `server/vllm_modal.py` (I1) before the session. Check the flag spellings first with `vllm serve --help` inside the container.
- **Run:** each experiment is a deploy with `P1_EXPERIMENT=<name>`, a restart (about 5 min), then `load.py` at 16 users, plus 64 where the expected effect is there.

| Order | Experiment | Expected effect | Why this one |
| --- | --- | --- | --- |
| 1 | E1 `--max-num-seqs 16` | at 64 users the throughput stays at the 16-user level; more requests wait | the batch-size limit |
| 2 | E3 `--max-model-len 2048` | the same KV tokens at start-up, but the logged maximum concurrency rises from 2.48× to about 9.9× | KV capacity vs context length |
| 3 | E4 `--gpu-memory-utilization 0.85` | KV memory about 0.85 × 22.1 − 15.27 − 1.85 ≈ 1.67 GiB ≈ 12,200 tokens (from 20,336), so about 17 requests of 700 tokens fit and queueing starts above about 17 users | memory headroom |

- **Not 0.6, as the task list first said:** 0.6 × 22.1 ≈ 13.3 GiB is less than the 15.27 GiB of weights, so the server would not start.
- **Spare (only if time is left):** E2, prefix caching, needs a shared-prefix workload that `make_prompt` does not make yet.
- **Done when:** three rows each with a baseline row and a one-line cause that matches the numbers (**M6**); the app is stopped and the cost logged.

### Step 9. Write-Up (S4, 30 min, no GPU)

- **Do:** README: how to run each step, the results table from `results/results.csv`, the three findings, the KV cache estimate vs the log (M4), cost (M9), what broke (the 503 incident, the key leak in S1).
- **Done when:** someone else could rerun any row from the README alone (**M8**); `uv run pytest` passes (**M11**); the M10 grep finds nothing; the work is merged to `main`.

### Step 10. Interview Prep (spare slot, 40 min, no GPU)

- **Do:** answer the six questions in *P1 vLLM Serving - Project Plan* aloud, each with a number from the results table; record the 2-minute story.
- **Done when:** each answer names a number; the story covers the goal, the setup, two findings and one failure.

## What Is Left Out

- **The Mac (vllm-metal) rows, T2–T6 of the task list,** are not needed for any metric. They are optional after P1, as a comparison.
- **E5 (FP8) and E6 (long prompts)** are not planned: three experiments are enough for M6.
- **Pass rule:** a metric that fails is recorded with its cause, not hidden. P1 can close with a documented miss, as M4 first did.

## Risks

| Risk | Effect | Plan |
| --- | --- | --- |
| no container starts after a deploy (2026-11-10) | lost GPU-session minutes | `min_containers=1` override at the start of each session; check `modal container list` |
| `vllm bench serve` does not run on a CPU container | M2 blocked | fallback: `modal container exec` inside the server container |
| network time distorts the M2 comparison | false miss | same placement for both tools, or compare TPOT and TTFT minus the round trip |
| an experiment flag is spelled wrong for 0.21.0 | the server fails to start, wasting a restart | check `vllm serve --help` in the container before S4 |
| forgetting to stop the app | the L4 bills about \$0.80/h | the stop and an empty container list are the last lines of every GPU step |

## Related Notes (vault)

- *P1 vLLM Serving - Project Plan*
- *P1 vLLM Serving - Tasks and Experiments*
- *P1 vLLM Serving - Session Log*
- *Checklist - Launch an LLM Server on a Cloud GPU with vLLM and Modal*
- *OpenAI-Compatible API as the Chat Interface for Open-Weight Models like Qwen*
- *KV Cache Memory Budget and Capacity Estimation in vLLM*
- *vLLM Call the OpenAI-Compatible Server with openai and httpx*
