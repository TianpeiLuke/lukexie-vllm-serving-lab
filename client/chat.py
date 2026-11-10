"""Chat with the vLLM server and print TTFT, output tokens and TPOT per answer.

One prompt:  uv run --env-file .env python client/chat.py "Explain the KV cache."
Live chat:   uv run --env-file .env python client/chat.py      (empty line to quit)

VLLM_API_KEY comes from .env; VLLM_BASE_URL defaults to the deployed Modal endpoint.
"""

import os
import sys
import time

import httpx
from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "https://unidoctor--vllm-serving-lab-server.us-east.modal.direct")
MODEL = "Qwen/Qwen3-8B"

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key=os.environ["VLLM_API_KEY"])


def ask(messages, max_tokens=512):
    """Stream one answer; return its text and print the timings."""
    start = time.perf_counter()
    first_token_at = usage = None
    parts = []
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},  # last chunk carries token counts
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3: skip the <think> block
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            parts.append(chunk.choices[0].delta.content)
            print(parts[-1], end="", flush=True)
    end = time.perf_counter()
    n = usage.completion_tokens
    ttft = first_token_at - start
    tpot = (end - first_token_at) / max(n - 1, 1)  # decode time per token after the first
    print(f"\n[TTFT {ttft * 1000:.0f} ms | {n} tokens | TPOT {tpot * 1000:.1f} ms | prompt {usage.prompt_tokens} tokens]")
    return "".join(parts)


def wait_until_healthy(timeout_s=20 * 60, poll_s=10):
    """503 or a network error means the container is queued for a GPU or vLLM is still loading."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/health", timeout=30).status_code == 200:
                return
        except httpx.TransportError:
            pass
        print("waiting for the server ...", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"{BASE_URL} did not become healthy in {timeout_s} s")


if __name__ == "__main__":
    wait_until_healthy()
    if len(sys.argv) > 1:
        ask([{"role": "user", "content": " ".join(sys.argv[1:])}])
        sys.exit()
    messages = []
    while prompt := input("\nyou> ").strip():
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": ask(messages)})
