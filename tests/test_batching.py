import pytest

from bench.batching import LONGEST, SHORTEST, output_lengths, predicted_ratio, run_mode
from tests.test_load import fake_server


def test_output_lengths():
    assert output_lengths("fixed", 3, seed=0) == [200, 200, 200]
    varied = output_lengths("varied", 500, seed=1)
    assert varied == output_lengths("varied", 500, seed=1)  # the same seed gives the same lengths
    assert SHORTEST <= min(varied) and max(varied) <= LONGEST


def test_predicted_ratio():
    assert predicted_ratio([200] * 32, users=16) == 1.0
    # static: groups (10, 100) and (10, 100) take 100 + 100 steps; continuous: 220 / 2 = 110
    assert predicted_ratio([10, 100, 10, 100], users=2) == pytest.approx(200 / 110)


def test_static_waits_for_each_group():
    client, state = fake_server()
    result = run_mode(client, "static", users=4, lens=[3] * 10, seed=0)
    assert (result.failed, len(result.timings)) == (0, 10)
    starts = [i for i, (event, _) in enumerate(state["log"]) if event == "start"]
    ends = [i for i, (event, _) in enumerate(state["log"]) if event == "end"]
    assert starts[4] > ends[3] and starts[8] > ends[7]  # each group starts after the last one ended


def test_continuous_refills_slots():
    client, state = fake_server()
    result = run_mode(client, "continuous", users=4, lens=[3] * 10, seed=0)
    assert (result.failed, len(result.timings), state["peak"]) == (0, 10, 4)
    assert state["max_tokens"] == [3] * 10
