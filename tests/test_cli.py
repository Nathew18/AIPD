"""Testy interfejsu i kontraktu CSV, bez dodatkowych pakietow."""
import csv
import math
import pathlib
import subprocess
import sys
import tempfile

exe = str(pathlib.Path(sys.argv[1]).resolve())


def run(*args, ok=True):
    result = subprocess.run([exe, *args], text=True, capture_output=True, timeout=60)
    assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
    return result


for args in [
    ("--n", "0"), ("--n", "-1"), ("--n", ""), ("--n", "1,,2"),
    ("--n", "2,"), ("--n", "2abc"), ("--n", "2147483648"),
    ("--threads", "0"), ("--steps", "0"), ("--steps", "-2"),
    ("--repeats", "0"), ("--warmup", "-1"), ("--seed", "-1"),
    ("--dt", "nan"), ("--dt", "inf"), ("--dt", "0.1junk"),
    ("--eps", "0"), ("--eps", "1e-200"), ("--mode", "unknown"), ("--mode", "cuda"),
    ("--warmup", "2147483647", "--repeats", "1"), ("--n",),
]:
    run(*args, ok=False)

with tempfile.TemporaryDirectory() as tmp:
    output = pathlib.Path(tmp) / "nested" / "result.csv"
    args = ("--mode", "all", "--n", "1,17", "--threads", "1,2",
            "--steps", "2", "--repeats", "2", "--warmup", "0",
            "--seed", "123", "--check", "--csv", str(output))
    run(*args)
    run(*args)  # dopisywanie bez drugiego naglowka
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    modes = {row["mode"] for row in rows}
    assert {"seq", "omp"} <= modes <= {"seq", "omp", "cuda-naive"}
    omp_threads = {row["threads"] for row in rows if row["mode"] == "omp"}
    expected = 4 * (1 + len(omp_threads) + (1 if "cuda-naive" in modes else 0))
    assert len(rows) == expected, len(rows)
    for row in rows:
        assert None not in row, row
        assert row["seed"] == "123"
        assert float(row["max_diff_vs_seq"]) < 1e-3
        for key, value in row.items():
            if key not in {"mode", "real"} and value:
                assert math.isfinite(float(value)), (key, value)
        assert float(row["median_s"]) > 0
        if row["n"] == "1":
            assert row["energy_rel_drift"] == ""  # energia poczatkowa = 0
    output.write_text("old,format\n", encoding="utf-8")
    run(*args, ok=False)
    assert output.read_text(encoding="utf-8") == "old,format\n"
    if "cuda-naive" not in modes:
        run("--mode", "cuda-naive", "--n", "1", ok=False)
print("CLI i CSV OK")
