"""Append one benchmark row to results/results.csv with a fixed set of columns."""

import csv
import os

COLUMNS = ("timestamp", "env", "model", "setting", "users", "num_prompts", "failed",
           "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "output_tok_s",
           "tool", "vllm_version", "command")  # tool: "load.py" or "vllm bench serve"


def append_result(row: dict, path: str = "results/results.csv") -> None:
    """Append row to the CSV at path; write the header first if the file is new or empty."""
    missing = [c for c in COLUMNS if c not in row]
    unknown = [c for c in row if c not in COLUMNS]
    if missing or unknown:
        raise ValueError(f"missing columns {missing}, unknown columns {unknown}")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
