# lukexie-vllm-serving-lab

My personal practice of LLM serving with [vLLM](https://github.com/vllm-project/vllm): start a server, call it through the OpenAI-compatible API, load-test it, change one engine setting at a time, and explain every number with the KV cache and batching.

> **Status (Nov 5, 2026):** the GPU server runs: `server/vllm_modal.py` starts vLLM 0.21.0 with Qwen3-8B on a Modal L4 and the smoke test passes (`/health` 200, `/v1/models` 200). The Mac path, the client and the benchmarks are not built yet.

## Goals

1. Serve an open model with `vllm serve` and query it with curl and a streaming Python client.
2. Measure time to first token (TTFT), time per output token (TPOT) and throughput at 1, 4, 16 and 64 concurrent users.
3. Estimate the KV cache capacity by hand and compare it with what vLLM reports at start-up.
4. Run three experiments that each change one setting, and explain why the numbers move.

## Environments

| | Mac | CUDA GPU |
| --- | --- | --- |
| Hardware | Apple M4 Pro, 24 GB unified memory, macOS 15 | one NVIDIA L4 (24 GB) rented on [Modal](https://modal.com) |
| vLLM | [vllm-metal](https://github.com/vllm-project/vllm-metal) plugin (MLX backend) | official vLLM |
| Model | `Qwen/Qwen3-0.6B`, then `mlx-community/Qwen3-8B-4bit` | `Qwen/Qwen3-8B` (bf16) |
| Used for | learning the API, the client and the benchmark tool | the real engine and the reported numbers |

vllm-metal is a community plugin, so Mac throughput is not comparable with CUDA vLLM; the results table keeps the two apart.

## Milestones

- [ ] **M1, Mac server:** install vllm-metal, serve Qwen3-0.6B and Qwen3-8B 4-bit, streaming client, single-user TTFT and TPOT, `vllm bench serve` at 1 and 4 users.
- [ ] **M2, GPU server:** ~~serve vLLM on a Modal L4~~ (done), ~~read the KV cache size from the start-up log~~ (done: 20,336 tokens), baseline benchmark at 1, 4, 16, 64 users.
- [ ] **M3, experiments and write-up:** three one-setting experiments, `/metrics` during a 64-user run, results and findings below.

## Layout

```text
server/vllm_modal.py   # vLLM on a Modal L4, with a smoke test (done)
client/                # streaming chat client that reports TTFT and TPOT (planned)
bench/                 # benchmark commands and workloads (planned)
results/               # benchmark output and summary tables; *.log files stay local (gitignored)
```

## Quick Start

**Mac** (not run yet; [vllm-metal install](https://github.com/vllm-project/vllm-metal), needs macOS 15+):

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --max-model-len 8192
curl http://127.0.0.1:8000/v1/models
```

**GPU** (`server/vllm_modal.py`, adapted from [Modal's vLLM example](https://modal.com/docs/examples/vllm_inference): one L4, `Qwen/Qwen3-8B`, `--max-model-len 8192`, one container, API key required):

```bash
uv sync
uv run modal setup                                   # once: log in to Modal
printf 'VLLM_API_KEY=%s\n' "$(openssl rand -hex 24)" > .env   # once; .env is gitignored
chmod 600 .env
uv run --with python-dotenv modal secret create vllm-api-key --from-dotenv .env   # add --force to rotate
uv run --env-file .env modal run server/vllm_modal.py      # temporary server + smoke test
uv run modal deploy server/vllm_modal.py                   # persistent endpoint
uv run modal app logs vllm-serving-lab                     # start-up log (kept 1 day on the Starter plan)
uv run modal app stop vllm-serving-lab                     # stop after each session
```

- The server reads the key from the `VLLM_API_KEY` environment variable set by the Modal secret. Don't pass `--api-key`: vLLM logs its non-default arguments, so the key would end up in the log.
- The URL answers **503** until vLLM is ready, about 4 minutes from container start (weights cached on a Volume). Clients should retry 503 and network errors.
- With `modal run`, an error in the local smoke test stops the GPU container too.

## Benchmark Workload

The same workload for every run so that only the setting under test changes: random prompts, 500 input tokens, 200 output tokens, fixed seed, load set by `--max-concurrency`.

```bash
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model <model> --base-url <server url> \
  --dataset-name random --random-input-len 500 --random-output-len 200 \
  --max-concurrency 16 --num-prompts 100 --seed 0
```

## KV Cache Estimate

For `Qwen/Qwen3-8B` (36 layers, 8 KV heads, head size 128, bf16, from its `config.json`):

```text
KV bytes per token = 2 (K and V) x 36 layers x 8 heads x 128 x 2 bytes = 147,456 B, about 144 KiB
L4 memory for KV  ~ 24 GB x 0.9 - 16.4 GB weights - ~1.5 GB overhead ~ 3.7 GB
KV cache capacity ~ 3.7e9 / 147,456 ~ 25,000 tokens
```

**Measured (Nov 5, vLLM 0.21.0, L4):** 2.79 GiB of KV cache memory = **20,336 tokens**, maximum concurrency **2.48×** at 8,192 tokens per request. The estimate was 23% high, mainly because the L4's "24 GB" is about 22.1 GiB usable and I mixed GB with GiB; the overhead was 1.85 GiB (0.52 GiB CUDA graphs + about 1.33 GiB activations), not 1.5.

## Experiments (three to be chosen)

| # | Change | Question |
| --- | --- | --- |
| E1 | `--max-num-seqs` 256 → 16 | how does the batch limit trade throughput for queueing? |
| E2 | prefix caching on vs off, shared 400-token prefix | when does prefix caching cut TTFT? |
| E3 | `--max-model-len` 8192 → 2048 | how does context length change maximum concurrency? |
| E4 | `--gpu-memory-utilization` 0.9 → 0.6 | what happens when KV memory runs out? |
| E5 | bf16 → FP8 weights | memory and bandwidth vs output quality |
| E6 | prompt 500 → 2000 tokens | prefill (TTFT) vs decode (TPOT) |

## Results

Start-up on a Modal L4, Qwen3-8B bf16, `--max-model-len 8192` (Nov 5, second run, weights already on the Volume):

| Step | Time |
| --- | --- |
| load weights (15.27 GiB) | 14.2 s |
| torch.compile | 44.4 s |
| whole engine start-up (profile, KV cache, CUDA graphs) | 181.8 s |
| container start → server ready | about 4 min |

Benchmarks:

| Env | Model | Setting | Users | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | Output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## References

- [vLLM documentation](https://docs.vllm.ai): [quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html), [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/), [benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli.html), [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html), [optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html), [metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)
- Kwon et al. (2023), [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vllm-metal](https://github.com/vllm-project/vllm-metal) and its [supported models](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md)
- [Modal: vLLM inference example](https://modal.com/docs/examples/vllm_inference)
