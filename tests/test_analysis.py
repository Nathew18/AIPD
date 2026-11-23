"""Kontrola dopasowania baz seq i obliczania udzialu transferow."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("analyze", Path(__file__).parents[1] / "scripts/analyze.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class AnalysisTests(unittest.TestCase):
    def row(self, mode, median, mean, threads=1):
        return dict(mode=mode, real="float", n="256", steps="10", threads=str(threads),
                    seed="42", dt="0.001", eps="0.01", median_s=str(median), mean_s=str(mean),
                    cuda_h2d_s="0.1", cuda_kernel_s="0.5", cuda_d2h_s="0.2")

    def test_baseline_and_mean_timing(self):
        groups = {(256, "seq", 1): [self.row("seq", 4, 5)],
                  (256, "omp", 2): [self.row("omp", 1, 2, 2)],
                  (256, "cuda-naive", 1): [self.row("cuda-naive", 0.5, 1)]}
        result = {r["mode"]: r for r in analyze.summarize(groups)}
        self.assertEqual(result["omp"]["speedup"], 4)
        self.assertEqual(result["omp"]["efficiency"], 2)
        self.assertEqual(result["cuda-naive"]["speedup"], 8)
        self.assertAlmostEqual(result["cuda-naive"]["transfer_fraction"], 0.3)

    def test_missing_reference_stays_empty(self):
        result = analyze.summarize({(256, "omp", 2): [self.row("omp", 1, 1, 2)]})[0]
        self.assertIsNone(result["speedup"])
        self.assertIsNone(result["efficiency"])

    def test_incompatible_experiments_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "input.csv"
            first, second = self.row("seq", 1, 1), self.row("omp", 1, 1)
            second["steps"] = "20"
            with file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=analyze.FIELDS)
                writer.writeheader()
                writer.writerows([first, second])
            with self.assertRaises(ValueError):
                analyze.load_measurements([file])


if __name__ == "__main__":
    unittest.main()
