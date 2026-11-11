import json
import threading
import time

import httpx2
from openai import OpenAI

from bench.load import run_load


def fake_server(fail_on=None, delay_s=0.05):
    """An OpenAI client on a mock vLLM: 3 output tokens per request; tracks the most requests in flight."""
    state = {"now": 0, "peak": 0, "prompts": [], "max_tokens": [], "log": []}
    lock = threading.Lock()

    def handler(request):
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        with lock:
            state["prompts"].append(content)
            state["max_tokens"].append(body["max_tokens"])
            state["log"].append(("start", content))
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            index = len(state["prompts"]) - 1
        time.sleep(delay_s)  # keep the request in flight so overlap is visible
        with lock:
            state["now"] -= 1
            state["log"].append(("end", content))
        if index == fail_on:
            return httpx2.Response(500, json={"error": {"message": "boom"}})
        base = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "Qwen/Qwen3-8B"}
        events = [{**base, "choices": [{"index": 0, "delta": {"content": w}, "finish_reason": f}]}
                  for w, f in [("a", None), ("b", None), ("c", "length")]]
        events.append({**base, "choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 3,
                                                        "total_tokens": 503}})
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = OpenAI(base_url="http://test/v1", api_key="test", max_retries=0,
                    http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    return client, state


def test_concurrency_caps_requests_in_flight():
    client, state = fake_server()
    result = run_load(client, concurrency=4, num_prompts=12, input_len=50, output_len=3)
    assert (result.failed, len(result.timings)) == (0, 12)
    assert state["peak"] == 4
    assert len(set(state["prompts"])) == 12  # every request has its own prompt


def test_failed_request_counted_not_raised():
    client, _ = fake_server(fail_on=2)
    result = run_load(client, concurrency=2, num_prompts=5, input_len=50, output_len=3)
    assert (result.num_prompts, result.failed, len(result.timings)) == (5, 1, 4)
    assert all(t.output_tokens == 3 for t in result.timings)


def test_output_lens_set_each_request():
    client, state = fake_server()
    run_load(client, concurrency=1, num_prompts=3, input_len=50, output_lens=[5, 7, 9])
    assert state["max_tokens"] == [5, 7, 9]
