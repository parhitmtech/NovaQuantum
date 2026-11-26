#!/usr/bin/env python3
"""Compiling circuits for a neutral-atom machine: what each step costs or saves, and a check that it is correct.

    qbench.py --build ~/nova-build [--width 3]

For each benchmark circuit (circuits.py), --qnova-stats after three points of the pipeline:
    logical   the circuit as written (h, cx, cp, swap, ...)
    native    after --qnova-decompose: every gate as the machine's r, rz and cz, nothing optimized
    compiled  after --qnova-to-native: decomposed, canonicalized (gates cancelled and merged), routed and scheduled
Then both the logical circuit and the compiled one are compiled to executables (nova-compile) and run on the
runtime's statevector simulator: their probability distributions must be the same to four places, or the
compilation changed what the circuit computes. Exit status 1 if any differs.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circuits  # noqa: E402

BENCHMARKS = [
    ("ghz-8", lambda: circuits.ghz(8)),
    ("ghz-12", lambda: circuits.ghz(12)),
    ("qft-roundtrip-5", lambda: circuits.qft_roundtrip(5)),
    ("qft-roundtrip-8", lambda: circuits.qft_roundtrip(8)),
    ("random-6x60", lambda: circuits.random_circuit(6, 60, 1)),
    ("random-8x120", lambda: circuits.random_circuit(8, 120, 2)),
    ("random-10x200", lambda: circuits.random_circuit(10, 200, 3)),
    ("random-12x300", lambda: circuits.random_circuit(12, 300, 4)),
]

STATS = re.compile(r"(\d+) qubits, (\d+) gates \(([^)]*)\), (\d+) two-qubit, (\d+) moves(?: \(([\d.]+) sites\))?, "
                   r"depth (\d+)")


def stats(tool, path, passes):
    if passes and passes[0].startswith("--pass-pipeline="):  # a textual pipeline takes no other pass flags
        passes = [passes[0][:-1] + ",func.func(qnova-stats))"]
    else:
        passes = passes + ["--qnova-stats"]
    p = subprocess.run([tool, path, *passes, "-o", os.devnull], capture_output=True, text=True)
    m = STATS.search(p.stderr)
    if not m:
        sys.exit(f"no stats for {path} {passes}: {p.stderr}")
    return {"gates": int(m[2]), "two": int(m[4]), "moves": int(m[5]), "distance": float(m[6] or 0),
            "depth": int(m[7])}


def run(compile_tool, path, exe, logical):
    subprocess.run([compile_tool, path, "-o", exe] + (["--logical"] if logical else []), check=True)
    return subprocess.run([exe], capture_output=True, text=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="the MLIR build directory")
    parser.add_argument("--width", type=int, default=0, help="atom grid width for --qnova-schedule (0: square)")
    args = parser.parse_args()
    opt = os.path.join(args.build, "bin", "nova-opt")
    compile_tool = os.path.join(args.build, "bin", "nova-compile")
    # --qnova-to-native, with the grid width if one is given
    schedule = "--qnova-to-native" if args.width == 0 else (
        "--pass-pipeline=builtin.module(canonicalize,qnova-decompose,canonicalize,cse,"
        f"func.func(qnova-schedule{{width={args.width}}}))")

    print(f"{'circuit':<17}{'qubits':>7}{'logical':>9}{'native':>8}{'compiled':>10}{'saved':>7}{'cz':>6}"
          f"{'moves':>7}{'depth':>7}  {'probabilities'}")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, make in BENCHMARKS:
            c = make()
            path = os.path.join(tmp, name + ".mlir")
            with open(path, "w") as f:
                f.write(c.text())
            logical = stats(opt, path, [])
            native = stats(opt, path, ["--qnova-decompose"])
            compiled = stats(opt, path, [schedule])
            same = run(compile_tool, path, os.path.join(tmp, "l"), True) == run(compile_tool, path,
                                                                                   os.path.join(tmp, "n"), False)
            failures += not same
            saved = 100 * (native["gates"] - compiled["gates"]) / native["gates"]
            print(f"{name:<17}{c.n:>7}{logical['gates']:>9}{native['gates']:>8}{compiled['gates']:>10}"
                  f"{saved:>6.0f}%{compiled['two']:>6}{compiled['moves']:>7}{compiled['depth']:>7}  "
                  f"{'same' if same else 'DIFFERENT'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
