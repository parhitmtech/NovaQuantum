#!/usr/bin/env python3
"""Nova programs, interpreted and compiled to native code through MLIR and LLVM: wall-clock time of each.

    benchmark.py --build ~/nova-build --interpreter compiler/build/nova compiler/benchmarks/compare/*.nova

Each program is compiled once with nova-compile (clang -O2), then the interpreter and the executable each run it
--runs times; the table shows the median wall-clock time of each (process start included) and the speedup. The two
outputs must be the same (difftest.py's check), or the program is reported and not timed. A C version next to the
program (prog.c), compiled by the same clang -O2, is timed too: the baseline native code should reach.
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time


def timed(cmd, runs):
    times, out = [], None
    for _ in range(runs):
        start = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True)
        times.append(time.perf_counter() - start)
        out = (p.returncode, p.stdout, p.stderr)
    return statistics.median(times) * 1000, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="the MLIR build directory (bin/nova-compile)")
    parser.add_argument("--interpreter", required=True, help="the interpreter, compiler/build/nova")
    parser.add_argument("--clang", default="/usr/lib/llvm-20/bin/clang", help="for the C baselines")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("programs", nargs="+")
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for src in args.programs:
            exe = os.path.join(tmp, os.path.basename(src)[: -len(".nova")])
            c = subprocess.run([os.path.join(args.build, "bin", "nova-compile"), src, "-o", exe],
                               capture_output=True, text=True)
            if c.returncode != 0:
                print(f"{src}: not compiled: {c.stderr.strip().splitlines()[0]}")
                continue
            interp_ms, interp_out = timed([args.interpreter, src], args.runs)
            native_ms, native_out = timed([exe], args.runs)
            if interp_out != native_out:
                print(f"{src}: outputs differ, not timed")
                continue
            c_ms = None
            c_src = src[: -len(".nova")] + ".c"
            if os.path.exists(c_src) and subprocess.run([args.clang, "-O2", c_src, "-o", exe + "_c"]).returncode == 0:
                c_ms, _ = timed([exe + "_c"], args.runs)
            rows.append((os.path.basename(src), interp_ms, native_ms, c_ms))

    print(f"\n{'program':<20}{'interpreter':>14}{'Nova native':>14}{'speedup':>10}{'C, clang -O2':>15}")
    for name, interp_ms, native_ms, c_ms in rows:
        c = f"{c_ms:>12.1f} ms" if c_ms is not None else f"{'-':>15}"
        print(f"{name:<20}{interp_ms:>11.1f} ms{native_ms:>11.1f} ms{interp_ms / native_ms:>9.1f}x{c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
