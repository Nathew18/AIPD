"""Przyspieszenie, efektywnosc OpenMP i wykresy na podstawie CSV z nbody."""
import argparse
import csv
import math
from pathlib import Path
import statistics


CONFIG = ("real", "steps", "seed", "dt", "eps")
FIELDS = ("mode", "real", "n", "steps", "threads", "seed", "dt", "eps",
          "median_s", "mean_s", "cuda_h2d_s", "cuda_kernel_s", "cuda_d2h_s")


def load_measurements(paths):
    groups = {}
    configurations = set()
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(FIELDS) <= set(reader.fieldnames or []):
                raise ValueError(f"{path}: wymagany aktualny format CSV z nbody")
            for line, row in enumerate(reader, 2):
                if None in row or any(row[key] is None for key in FIELDS):
                    raise ValueError(f"{path}:{line}: niepelny wiersz")
                config = (row["real"], int(row["steps"]), int(row["seed"]),
                          float(row["dt"]), float(row["eps"]))
                configurations.add(config)
                n, threads = int(row["n"]), int(row["threads"])
                times = [float(row[key]) for key in ("median_s", "mean_s")]
                if (n <= 0 or threads <= 0 or config[1] <= 0 or
                    any(not math.isfinite(x) or x <= 0 for x in times) or
                    any(not math.isfinite(x) or x <= 0 for x in config[3:])):
                    raise ValueError(f"{path}:{line}: niepoprawne parametry lub czasy")
                if row["mode"] not in {"seq", "omp", "cuda", "cuda-naive"}:
                    raise ValueError(f"{path}:{line}: nieznany wariant")
                key = (n, row["mode"], threads)
                groups.setdefault(key, []).append(row)
    if len(configurations) != 1:
        raise ValueError("Analizuj jedna konfiguracje real/steps/seed/dt/eps naraz; "
                         "rozdziel rozne serie na osobne pliki")
    return groups, next(iter(configurations))


def summarize(groups):
    medians = {key: statistics.median(float(r["median_s"]) for r in rows)
               for key, rows in groups.items()}
    summary = []
    for (n, mode, threads), rows in sorted(groups.items()):
        elapsed = medians[n, mode, threads]
        baseline = medians.get((n, "seq", 1))
        speedup = baseline / elapsed if baseline is not None else None
        result = dict(n=n, mode=mode, threads=threads, median_s=elapsed,
                      speedup=speedup,
                      efficiency=speedup / threads if mode == "omp" and speedup is not None else None,
                      h2d_fraction=None, kernel_fraction=None, d2h_fraction=None,
                      transfer_fraction=None)
        if mode.startswith("cuda"):
            means = [float(r["mean_s"]) for r in rows]
            for source, target in (("h2d", "h2d_fraction"), ("kernel", "kernel_fraction"),
                                   ("d2h", "d2h_fraction")):
                values = [float(r[f"cuda_{source}_s"]) for r in rows]
                if any(not math.isfinite(x) or x < 0 for x in values):
                    raise ValueError("Niepoprawne czasy CUDA")
                # Fazy CUDA sa srednimi, wiec porownujemy je do mean_s.
                result[target] = statistics.mean(v / t for v, t in zip(values, means))
            result["transfer_fraction"] = result["h2d_fraction"] + result["d2h_fraction"]
        summary.append(result)
    return summary


def plot(summary, destination, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    series = sorted({(r["mode"], r["threads"]) for r in summary})
    for mode, threads in series:
        rows = sorted((r for r in summary if (r["mode"], r["threads"]) == (mode, threads)),
                      key=lambda r: r["n"])
        label = f"omp ({threads} w.)" if mode == "omp" else mode
        axes[0, 0].loglog([r["n"] for r in rows], [r["median_s"] for r in rows], "o-", label=label)
        paired = [r for r in rows if r["speedup"] is not None]
        if paired:
            axes[0, 1].semilogx([r["n"] for r in paired], [r["speedup"] for r in paired], "o-", label=label)
    axes[0, 0].set(xlabel="Liczba ciał N", ylabel="Czas całej symulacji [s]", title="Czas (mediana)")
    axes[0, 1].set(xlabel="Liczba ciał N", ylabel="T_seq / T_wariant", title="Przyspieszenie")
    for n in sorted({r["n"] for r in summary}):
        rows = sorted((r for r in summary if r["n"] == n and r["efficiency"] is not None),
                      key=lambda r: r["threads"])
        if rows:
            axes[1, 0].plot([r["threads"] for r in rows], [r["efficiency"] for r in rows], "o-", label=f"N={n}")
    axes[1, 0].axhline(1, color="gray", linestyle="--")
    axes[1, 0].set(xlabel="Wątki OpenMP", ylabel="Przyspieszenie / wątki", title="Efektywność OpenMP")
    for mode in ("cuda", "cuda-naive"):
        rows = sorted((r for r in summary if r["mode"] == mode), key=lambda r: r["n"])
        if rows:
            axes[1, 1].semilogx([r["n"] for r in rows], [100 * r["transfer_fraction"] for r in rows], "o-", label=mode)
    axes[1, 1].set(xlabel="Liczba ciał N", ylabel="Transfery / średni czas całości [%]", title="Koszt transferów CUDA")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"N-body: {config[0]}, kroki={config[1]}, seed={config[2]}, dt={config[3]:.6g}, eps={config[4]:.6g}")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    parser.add_argument("--no-plots", action="store_true", help="tylko summary.csv, bez matplotlib")
    args = parser.parse_args()
    try:
        groups, config = load_measurements(args.csv)
        summary = summarize(groups)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*summary[0], *CONFIG])
            writer.writeheader()
            for row in summary:
                writer.writerow({**row, **dict(zip(CONFIG, config))})
        if not args.no_plots:
            plot(summary, args.output / "comparison.png", config)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Blad analizy: {error}\n")
    print(f"Zapisano analize: {args.output}")


if __name__ == "__main__":
    main()
