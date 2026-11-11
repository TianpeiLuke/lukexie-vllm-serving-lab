# P1 Implementation Plan: vLLM Serving Lab

Step-by-step plan derived from the success metrics in [`SPEC.md`](SPEC.md). Names in *italics* are notes in the Obsidian vault.

## Status (2026-11-10, 16:05)

| Step | State |
| --- | --- |
| MVP (M1): deploy and chat | **done**: Qwen3-8B answered on an L4 (14:05) and on an A10G (14:43); `client/chat.py` takes multi-line pastes |
| 0. Freeze the working state | **partly**: the app scales to zero by itself (`min_containers` unset); the commit of today's work is still to do |
| 1. `results/record.py` | **done**: 3 tests pass |
| 2. `client/timing.py` | **done**: 13 tests pass (20 in total) |
| 3. Live checks of the client | **done early** on the A10G: exact prompt and output lengths |
| 4. `bench/load.py` at 1 user | **done** on the A10G: 0 failed, TPOT p50 34.5 ms |
| 5. `vllm bench serve` and M2 | **done early** on the A10G (15:57–16:01): M2 passes, TTFT p50 within 7.6%, TPOT p50 within 0.9% |
| 6. Concurrency sweep | **done early** on the A10G (15:15–15:21): 4, 16 and 64 users, 0 failed; M3 passes (10.5×) |
| 7–10 | to do; S3 starts at Step 7 (`bench/metrics.py`). The 64-user engine log already previews M5 |

## Summary

> **How this plan is built**
> **Work backwards from the success metrics.** Each of the metrics M1–M11 in the repo's [`SPEC.md`](SPEC.md) needs certain code and certain GPU runs. The steps below are ordered so that:
> 1. code that needs no GPU is written and tested first, with the app stopped;
> 2. each GPU step produces a row that some metric checks;
> 3. every step ends with a **done when** check, and the GPU is on only for steps 3–8.

> **Schedule**
> | Session | Steps | GPU |
> | --- | --- | --- |
> | **S2 Tue Nov 10** (done, ran to about 4:05) | 1, 2, 4, and 3, 5 and 6 early; 0 partly (commit pending) | about 1.7 GPU hours in all: L4 about 20 min, A10G about 60 min (sweep) and about 23 min (M2) |
> | **S3 Thu Nov 12, 1:00–3:00** | 7, then 8 (and the L4 rows of 4–6 if the session gets an L4) | one start, about 60 min on |
> | **S4 Fri Nov 13, 1:00–3:00** | 7, 8, 9 | three restarts, about 60 min on |
> | spare / next mock slot | 10 | none |
>
> Expected GPU total about 2.5 h on an L4 ≈ \$2, well inside M9's \$10.

## From Metrics to Steps

| Metric | Needs | Step |
| --- | --- | --- |
| M1 server answers | deployed server, chat client | **passed 2026-11-10** (the MVP) |
| M2 client agrees with `vllm bench serve` | `timing.py`, `load.py`, a way to run `vllm bench serve`, both at 1 user | **passed 2026-11-10** (step 5, A10G): TTFT p50 194.5 vs 180.8 ms (7.6%), TPOT p50 33.7 vs 33.4 ms (0.9%) |
| M3 batching gain at 16 users | concurrency sweep with `load.py` | **passed 2026-11-10** (step 6, A10G): 287.3 ÷ 27.3 = 10.5× output tokens/s, pass line 4× |
| M4 KV cache explained | start-up log line, estimate | done (passes after the GiB correction); restate in the README in step 9 |
| M5 where it queues | `metrics.py`, a 64-user run with polling | 7 to do. Preview from the engine log: Waiting up to 36, KV usage 99.8%, 29 preemptions, TTFT p50 13.8 s |
| M6 three experiments | experiments by name in the server, three runs at 16 users | 8 |
| M7 no failed requests | `failed` counted in every run | so far: 0 failed in all 6 rows (steps 4, 5 and 6); 7, 8 to do |
| M8 reproducible | `record.py` with the command per row, README | 1 done (every row has its command); README in 9 |
| M9 cost | cost log per session | S2 logged (L4 about \$0.25, A10G about \$0.75, sweep included); total in 9 |
| M10 no secrets | key only in `.env` and the Modal secret | checked at every commit; key grep clean on 2026-11-10 15:40 (before the next commit) |
| M11 tests pass | unit tests for the pure functions | 1, 2, 4 and 5 done (20 pass); 7 still to add |

## Steps

### Step 0. Freeze Today's Working State (S2, 10 min, no GPU)

- **Do:** stop the app (`uv run modal app stop vllm-serving-lab`, then `uv run modal container list` is empty). Log today's GPU minutes in the cost table of *P1 vLLM Serving - Tasks and Experiments*. Commit `SPEC.md`, `client/chat.py` and the `server/vllm_modal.py` change on a branch `s2-live-endpoint`.
- **Decide:** keep `min_containers=1` out of the code and set it with the override only during sessions. A deploy then never leaves a GPU running by accident.
- **Learned 2026-11-10:** the deploys that started no container were waiting for a free L4 (the app log said "waiting to be scheduled on a GPU_L4 worker"), not blocked by autoscaling. `GPU = ["L4", "A10"]` now falls back to an A10G; with `min_containers` unset, a request starts a container on demand.
- **State:** the app scales to zero after 15 idle minutes. **Still to do:** commit Steps 1–3, `client/chat.py`, the GPU list and the plan and spec edits.
- **Done when:** zero containers; the M10 grep (in `SPEC.md`) finds nothing; the commit is pushed.

### Step 1. Results File: `results/record.py` and Its Test (S2, 20 min, no GPU)

**Done 2026-11-10 15:00:** `uv run pytest tests/test_record.py` gives 3 passed. The `env` column records the GPU that ran (`L4` or `A10G`), because the server can land on either.

- **Do:** `append_result(row, path)` with the fixed `COLUMNS`; header on a new file; `ValueError` on a missing or unknown column.
- **Test:** `tests/test_record.py` with `tmp_path`: the header written once, two rows appended, a bad row rejected.
- **Done when:** `uv run pytest tests/test_record.py` passes. Add `pytest` and `numpy` as dev dependencies (`uv add --dev pytest numpy`).
- **Why first:** every later step writes rows; M8 needs the command stored with each one.

### Step 2. Timing Math: `client/timing.py` and Its Test (S2, 30 min, no GPU)

**Done 2026-11-10 15:06:** `uv run pytest` gives 16 passed (13 in `tests/test_timing.py`, including `stream_and_time` on replayed vLLM events through a mock transport). One timestamp is taken per generated token: each text chunk, plus the finish chunk of the end-of-turn token, which has no text but counts in `completion_tokens`.

- **Do:** the pure functions first: `ttft_tpot`, `percentile`, `summarize`, and the `RequestTiming` dataclass. Then `make_prompt` (deterministic for a seed) and `stream_and_time`, which can only be checked live.
- **Test:** `tests/test_timing.py`: hand-made token times (none, one, many), `percentile` against `numpy.percentile` at q = 50 and 95, `summarize` on three fake timings with a known throughput.
- **Done when:** `uv run pytest` passes. `client/chat.py` still runs; it may reuse `stream_and_time`.
- **Derive first:** with $n$ output tokens, TPOT = (last token time − first token time)/(n − 1). Why n − 1 and not n?

### Step 3. Live Checks of the Client (S3, 15 min, GPU on)

**Both checks passed early, 2026-11-10 15:06, on the A10G that was still up:**

| `make_prompt(n)` | prompt tokens | output tokens (`max_tokens=200`, `ignore_eos`) | TTFT | TPOT |
| --- | --- | --- | --- | --- |
| 500 | 500 | 200 | 757 ms | 34.3 ms |
| 2000 | 2000 | 200 | 1238 ms | 34.8 ms |

The longer prompt raised TTFT but not TPOT: prefill grows with the prompt, while decode does not (a first look at E6). These are one-off A10G numbers, not results-table rows.

- **Do:** start the server (launch checklist: deploy, `min_containers=1`, the four live checks). Then run two live checks:
  - `make_prompt(500)` gives 475–525 prompt tokens by the server's `usage`;
  - `stream_and_time(..., max_tokens=200, ignore_eos=True)` returns exactly 200 output tokens.
- **Done when:** both checks pass. If the prompt length is off, calibrate words per token for Qwen's tokenizer once and fix `make_prompt`.
- **Cost note:** each start costs about 1–7 min of waiting for a GPU plus about 4 min of loading. Do all GPU steps of a session in one start.

### GPU for the Measured Runs (decide at the start of S3)

The server can now land on an L4 or an A10G, and their numbers can't share a table (decode speed follows memory bandwidth: 300 vs 600 GB/s).
- **Prefer the L4,** which the predictions and S1 used. Check with `uv run modal container exec <id> -- nvidia-smi --query-gpu=name --format=csv,noheader`.
- **If the session lands on an A10G,** run the whole sweep there and write `A10G` in the `env` column. Rows within one GPU are comparable; rows across GPUs are not.
- **To force the L4,** set `GPU = "L4"` for the session, and accept the wait if none is free.

### Step 4. Own Load Generator at 1 User: `bench/load.py` (S3, 25 min, GPU on)

**Done 2026-11-10 15:14, on the A10G:** `uv run --env-file .env python -m bench.load --env A10G --concurrency 1`. 20 prompts, **0 failed**, 146.7 s; TTFT p50 491 ms, p95 522 ms; **TPOT p50 34.5 ms**; 27.3 output tokens/s. It is row 1 of `results/results.csv`. `tests/test_load.py` (2 tests) checks the cap on requests in flight and the failure count; 18 tests pass in total. The A10G bound is 27 ms, so TPOT is 28% above it; the L4 rows (prediction 55–65 ms) are still to run.

- **Do:** `run_load` with a `ThreadPoolExecutor(max_workers=concurrency)` (the pool size caps requests in flight); a failed request is counted, not raised; one warm-up request is discarded; `main` appends one row per concurrency with `append_result`.
- **Run:** `--concurrency 1`, 20 prompts.
- **Done when:** the row has 0 failed requests (M7), and TPOT p50 is close to the prediction below.

> **Prediction to check: TPOT at 1 user is bandwidth-bound**
> Each decode step reads all weights once: $16.4\ \text{GB} / 300\ \text{GB/s}$ (L4 memory bandwidth) $\approx 55\ \text{ms}$. Expect a TPOT p50 of about 55–65 ms. A 1-user run then takes about 20 × 200 × 0.06 s ≈ 4 min.
> On an A10G the same bound is 16.4 / 600 ≈ 27 ms. Measured from the Mac on 2026-11-10: **34.3 ms**, so about 7 ms per token above the bound. The gap is something to explain (A10G bandwidth below the A10 spec, kernel overhead, the network).

### Step 5. Benchmark Tool and the M2 Comparison (S3, 30 min, GPU on)

**Done 2026-11-10 15:57–16:01, on the A10G:** `uv run --env-file .env modal run -m bench.vllm_bench_modal --env A10G` (option 1). Both tools ran **from the same Modal CPU container**, one after the other, 20 prompts each, 0 failed; both saw 500 prompt tokens (the server's `usage`).

| Tool | TTFT p50 | TTFT p95 | TPOT p50 | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| `vllm bench serve` | 180.8 ms | 295.2 ms | 33.4 ms | 29.2 |
| `load.py` | 194.5 ms | 208.5 ms | 33.7 ms | 29.1 |
| gap | 7.6% | | 0.9% | |

M2 passes. Checked in the vLLM 0.21.0 source first: `vllm bench serve` reads the key from `OPENAI_API_KEY` (no flag), and it ran on a CPU-only container. The same `load.py` from the Mac measured TTFT p50 491 ms, so the Mac adds about 300 ms of network to TTFT and about 1 ms to TPOT. Not explained: `load.py` TTFT is 14 ms higher (the two clients may stamp the first token at different chunks), and `vllm bench serve` has the higher p95 (one slow request out of 20).

- **Do:** run `vllm bench serve` against the endpoint, then compare it with step 4 at 1 user. vLLM is not installed on the Mac, so try these in order:
  1. **`bench/vllm_bench_modal.py`** (I4 in `SPEC.md`): a CPU Modal function in the vLLM image.
  2. **Fallback, no new code:** `uv run modal container exec <container id> vllm bench serve ...` inside the server container.
- **Decide before comparing:** run both tools **from the same place**. The Mac adds network time to every TTFT, and a tool inside Modal does not. So either run `load.py` inside the same Modal function, or compare TPOT (barely affected by the network) and TTFT after subtracting the measured round trip to `/health`.
- **Check first:** how `vllm bench serve` 0.21.0 sends the API key (environment variable, not a flag), and that it runs without a GPU.
- **Done when:** TTFT p50 and TPOT p50 agree within 10%, both rows are in `results/results.csv`, and the session log says which placement was used (**M2**).

### Step 6. Concurrency Sweep (S3, 20 min, GPU on)

**Done 2026-11-10 15:15–15:21, on the A10G:** `uv run --env-file .env python -m bench.load --env A10G --concurrency 4 16 64`, 100 prompts each, **0 failed**.

| Users | TTFT p50 | TTFT p95 | TPOT p50 | Output tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 491 ms | 522 ms | 34.5 ms | 27.3 |
| 4 | 678 ms | 871 ms | 37.6 ms | 98.2 |
| 16 | 1,408 ms | 2,253 ms | 44.1 ms | 287.3 |
| 64 | 13,786 ms | 17,988 ms | 63.9 ms | 418.8 |

M3 passes: 10.5× at 16 users. At 64 users vLLM admitted 38 requests (their prompts fit), not the 29 predicted below; as they grew, the cache hit 99.8% and it preempted back to about 28. The engine log is read second by second in the vault note *P1 Learning Session 1 - 2026-11-10 - Reading the vLLM Engine Log Prefill Decode and KV Cache Usage*.

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
- **E5 (FP8) and E6 (long prompts)** are not planned: three experiments are enough for M6. E5 needs an L4 (the A10G has no FP8). A first look at E6 came from Step 3: TTFT 757 → 1238 ms from 500 to 2000 prompt tokens, with TPOT unchanged.
- **Pass rule:** a metric that fails is recorded with its cause, not hidden. P1 can close with a documented miss, as M4 first did.

## Risks

| Risk | Effect | Plan |
| --- | --- | --- |
| no free L4 on Modal (2026-11-10: no container for 8 min, twice) | lost GPU-session minutes | `GPU = ["L4", "A10"]` fallback; read `modal app logs` for "waiting to be scheduled"; `min_containers=1` to start at once |
| the GPU type changes between sessions | rows that look comparable but are not | check `nvidia-smi` in the container; write the GPU in `env`; never mix GPUs in one comparison |
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
