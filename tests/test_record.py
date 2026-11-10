import csv

import pytest

from results.record import COLUMNS, append_result


def make_row(**overrides):
    row = {c: "" for c in COLUMNS}
    row.update(env="L4", model="Qwen/Qwen3-8B", setting="baseline", users=1,
               command='vllm bench serve --max-concurrency 1, "quoted"')  # commas and quotes survive CSV
    row.update(overrides)
    return row


def test_header_written_once_and_rows_appended(tmp_path):
    path = tmp_path / "results.csv"
    append_result(make_row(users=1), str(path))
    append_result(make_row(users=4), str(path))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert sum(line == lines[0] for line in lines) == 1
    rows = list(csv.DictReader(path.open()))
    assert [r["users"] for r in rows] == ["1", "4"]
    assert rows[0]["command"] == 'vllm bench serve --max-concurrency 1, "quoted"'


def test_missing_column_rejected(tmp_path):
    row = make_row()
    del row["command"]
    with pytest.raises(ValueError, match="command"):
        append_result(row, str(tmp_path / "results.csv"))
    assert not (tmp_path / "results.csv").exists()  # nothing written on a bad row


def test_unknown_column_rejected(tmp_path):
    with pytest.raises(ValueError, match="gpu"):
        append_result(make_row(gpu="A10G"), str(tmp_path / "results.csv"))
