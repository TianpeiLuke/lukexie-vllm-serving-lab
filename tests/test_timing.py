import json
import random

import httpx2
import numpy as np
import pytest
from openai import OpenAI

from client.timing import (RequestTiming, make_prompt, percentile, stream_and_time, summarize,
                           ttft_tpot)


def test_ttft_tpot_many_tokens():
    # start 10.0, tokens at 10.5, 10.6, 10.7, 10.8: 4 tokens, 3 gaps of 0.1 s
    ttft, tpot = ttft_tpot(10.0, [10.5, 10.6, 10.7, 10.8])
    assert ttft == pytest.approx(0.5)
    assert tpot == pytest.approx(0.1)


def test_ttft_tpot_one_token():
    assert ttft_tpot(1.0, [1.25]) == (pytest.approx(0.25), 0.0)


def test_ttft_tpot_no_tokens():
    with pytest.raises(ValueError):
        ttft_tpot(1.0, [])


@pytest.mark.parametrize("n", [1, 2, 7, 20])
def test_percentile_matches_numpy(n):
    values = [random.Random(i).uniform(0, 100) for i in range(n)]
    for q in (0, 25, 50, 90, 95, 99, 100):
        assert percentile(values, q) == pytest.approx(np.percentile(values, q))


@pytest.mark.parametrize("values, q", [([], 50), ([1.0], -1), ([1.0], 101)])
def test_percentile_bad_input(values, q):
    with pytest.raises(ValueError):
        percentile(values, q)


def test_summarize_three_requests():
    timings = [RequestTiming(0.1, 0.03, 1.0, 500, 200),
               RequestTiming(0.2, 0.04, 2.0, 500, 200),
               RequestTiming(0.3, 0.05, 3.0, 500, 200)]
    s = summarize(timings, duration_s=4.0)
    assert s["ttft_p50_ms"] == pytest.approx(200)
    assert s["ttft_p95_ms"] == pytest.approx(290)  # 200 + 0.9 * (300 - 200)
    assert s["tpot_p50_ms"] == pytest.approx(40)
    assert s["e2e_p50_ms"] == pytest.approx(2000)
    assert s["output_tok_s"] == pytest.approx(150)  # 600 tokens / 4 s


def test_make_prompt_deterministic_and_sized():
    assert make_prompt(500, seed=3) == make_prompt(500, seed=3)
    assert make_prompt(500, seed=3) != make_prompt(500, seed=4)
    (message,) = make_prompt(500)
    assert message["role"] == "user"
    assert len(message["content"].split()) == 500 - 12


def sse_client(events):
    """An OpenAI client whose server replays these SSE events (the shape vLLM 0.21 sends)."""
    def handler(request):
        sent = json.loads(request.content)
        assert sent["stream"] and sent["stream_options"] == {"include_usage": True}
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    return OpenAI(base_url="http://test/v1", api_key="test",
                  http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def chunk(delta, finish=None):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "Qwen/Qwen3-8B",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def test_stream_and_time_counts_one_time_per_token(monkeypatch):
    clock = iter(range(1, 100))  # perf_counter returns 1, 2, 3, ...: one tick per call
    monkeypatch.setattr("client.timing.clock", lambda: next(clock))
    # as captured live: role chunk, 6 text chunks, finish chunk (end-of-turn token), usage chunk
    events = [chunk({"role": "assistant", "content": ""})]
    events += [chunk({"content": t}) for t in ["Hello", ",", " how", " are", " you", "?"]]
    events += [chunk({"content": ""}, "stop"),
               {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "Qwen/Qwen3-8B",
                "choices": [], "usage": {"prompt_tokens": 18, "completion_tokens": 7, "total_tokens": 25}}]
    t = stream_and_time(sse_client(events), [{"role": "user", "content": "hi"}], max_tokens=10)
    assert (t.prompt_tokens, t.output_tokens) == (18, 7)
    # calls: start = 1, one per token (2..8), end = 9; so 7 token times were recorded, one per output token
    assert t.e2e_s == 8 and t.ttft_s == 1 and t.tpot_s == 1
