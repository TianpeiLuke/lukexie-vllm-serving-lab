"""Time streamed chat requests: TTFT, TPOT, end-to-end latency, and summaries over many requests."""

import math
import random
import time
from dataclasses import dataclass

from openai import OpenAI

MODEL = "Qwen/Qwen3-8B"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}  # Qwen3: answer without a <think> block
TEMPLATE_TOKENS = 12  # chat-template tokens around one user message with thinking off (measured: 6 words -> 18)
clock = time.perf_counter  # module-level so tests can replace it without touching the time module
WORDS = ("data model token cache memory request server batch layer value time queue user query answer "
         "system signal network window number result graph table index score field state block page "
         "file image text route price order event agent task plan goal rule test code bug fix step").split()


@dataclass
class RequestTiming:
    ttft_s: float          # request sent -> first generated token
    tpot_s: float          # (last token time - first token time) / (tokens - 1)
    e2e_s: float           # request sent -> stream finished
    prompt_tokens: int     # from the server's usage, not estimated
    output_tokens: int


def ttft_tpot(start: float, token_times: list[float]) -> tuple[float, float]:
    """TTFT and TPOT from the request start and one arrival time per generated token."""
    if not token_times:
        raise ValueError("no tokens arrived")
    ttft = token_times[0] - start
    if len(token_times) == 1:
        return ttft, 0.0
    return ttft, (token_times[-1] - token_times[0]) / (len(token_times) - 1)  # n tokens, n - 1 gaps


def stream_and_time(client: OpenAI, messages: list[dict], max_tokens: int,
                    model: str = MODEL, ignore_eos: bool = False) -> RequestTiming:
    """Stream one chat request and time it; ignore_eos=True forces exactly max_tokens output tokens."""
    start = clock()
    token_times = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},  # last chunk carries token counts
        extra_body={**NO_THINKING, "ignore_eos": ignore_eos},  # ignore_eos is a vLLM field
    )
    with stream:
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices:
                choice = chunk.choices[0]
                # one chunk per generated token: text, or the finish chunk of the end-of-turn token;
                # the first chunk (role only, no text, no finish) carries no token
                if choice.delta.content or choice.finish_reason:
                    token_times.append(clock())
    end = clock()
    ttft, tpot = ttft_tpot(start, token_times)
    return RequestTiming(ttft, tpot, end - start, usage.prompt_tokens, usage.completion_tokens)


def make_prompt(n_tokens: int, seed: int = 0) -> list[dict]:
    """One user message whose rendered prompt is about n_tokens tokens; the same seed gives the same text."""
    rng = random.Random(seed)
    n_words = max(n_tokens - TEMPLATE_TOKENS, 1)  # one Qwen token per word: 500 -> 500 and 2000 -> 2000 prompt tokens, live 2026-11-10
    return [{"role": "user", "content": " ".join(rng.choice(WORDS) for _ in range(n_words))}]


def percentile(values: list[float], q: float) -> float:
    """q-th percentile, q in [0, 100], with linear interpolation (numpy.percentile's default)."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize(timings: list[RequestTiming], duration_s: float) -> dict[str, float]:
    """Percentiles in ms over the requests, and output throughput over the whole run."""
    ttft = [t.ttft_s * 1000 for t in timings]
    return {
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "tpot_p50_ms": percentile([t.tpot_s * 1000 for t in timings], 50),
        "e2e_p50_ms": percentile([t.e2e_s * 1000 for t in timings], 50),
        "output_tok_s": sum(t.output_tokens for t in timings) / duration_s,
    }
