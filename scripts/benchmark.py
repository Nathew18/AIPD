"""Powtarzalna seria pomiarow wszystkich implementacji na jednym komputerze."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="sciezka do nbody / nbody.exe")
    parser.add_argument("--n", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"Nie znaleziono programu: {args.executable}")
    if args.csv.exists():
        parser.error(f"Plik juz istnieje: {args.csv}; podaj nowa sciezke dla nowej serii")
    command = [str(args.executable.resolve()), "--mode", "all"]
    for name in ("n", "threads", "steps", "repeats", "warmup", "seed", "dt", "eps", "csv"):
        command += [f"--{name}", str(getattr(args, name))]
    if args.check:
        command.append("--check")
    subprocess.run(command, check=True)
    print(f"Zapisano: {args.csv}")


if __name__ == "__main__":
    main()
