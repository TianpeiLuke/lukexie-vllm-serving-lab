from bench.vllm_bench_modal import vllm_bench_args, vllm_row


def test_args_match_load_py_and_carry_no_key():
    args = vllm_bench_args("http://server", concurrency=1, num_prompts=20, input_len=500, output_len=200, seed=0,
                           out="/tmp/r.json")
    flag = {a: args[i + 1] for i, a in enumerate(args[:-1]) if a.startswith("--")}
    assert flag["--random-input-len"] == "488"  # 488 random tokens + 12 template tokens = 500, like make_prompt
    assert flag["--random-output-len"] == "200" and "--ignore-eos" in args
    assert flag["--extra-body"] == '{"chat_template_kwargs": {"enable_thinking": false}}'
    assert (flag["--result-dir"], flag["--result-filename"]) == ("/tmp", "r.json")
    assert not any("key" in a.lower() for a in args)  # the key travels in OPENAI_API_KEY only


def test_vllm_row_maps_result_json():
    result = {"failed": 0, "p50_ttft_ms": 480.04, "p95_ttft_ms": 510.0, "p50_tpot_ms": 34.21,
              "output_throughput": 27.66, "completed": 20}
    assert vllm_row(result) == {"failed": 0, "ttft_p50_ms": 480.0, "ttft_p95_ms": 510.0,
                                "tpot_p50_ms": 34.2, "output_tok_s": 27.7}
