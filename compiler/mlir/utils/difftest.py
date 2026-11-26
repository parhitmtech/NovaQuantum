#!/usr/bin/env python3
"""Differential testing: every Nova program, compiled to native code and interpreted, must behave the same.

    difftest.py --build ~/nova-build --interpreter compiler/build/nova compiler/tests compiler/benchmarks/compare

For each .nova file (or each in a directory), nova-compile builds a native executable through MLIR and LLVM, and
both it and the interpreter run with the program's .input file (if any) on stdin. Their stdout, stderr and exit
status must match exactly: the runtime prints and fails as the interpreter does, so any difference is a bug in one
of the two lowerings.

A program nova-compile does not lower yet (strings, classes, containers, ...) is counted, with the reason, but is
not a failure: the report says how much of the language the native path covers. A program the front end rejects is
skipped (both paths share the front end). Exit status 1 if any compiled program differs.
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile


def run(cmd, stdin_path, timeout):
    with open(stdin_path or os.devnull, "rb") as stdin:
        try:
            p = subprocess.run(cmd, stdin=stdin, capture_output=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return "timeout", b"", b""


def programs(paths):
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".nova"):
                    yield os.path.join(path, name)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="the MLIR build directory (bin/nova-compile)")
    parser.add_argument("--interpreter", required=True, help="the interpreter, compiler/build/nova")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--verbose", action="store_true", help="list every program and its outcome")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    compile_tool = os.path.join(args.build, "bin", "nova-compile")
    env_home = os.path.dirname(os.path.abspath(args.interpreter.rstrip("/")))
    os.environ.setdefault("NOVA_HOME", os.path.dirname(env_home))  # compiler/, for the interpreter's stdlib

    match, differ, unlowered, rejected, skipped = [], [], [], [], []
    reasons = collections.Counter()
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "program")
        for src in programs(args.paths):
            base = src[: -len(".nova")]
            if os.path.exists(base + ".args"):  # a test of a compiler flag (--dump-tree, ...), not of the program
                skipped.append(src)
                continue
            stdin_path = base + ".input" if os.path.exists(base + ".input") else None
            c = subprocess.run([compile_tool, src, "-o", exe], capture_output=True, text=True)
            if c.returncode != 0:
                message = c.stderr.strip().splitlines()[0] if c.stderr.strip() else "no message"
                lowering = re.search(r"nova-lower cannot lower (.*) yet", message)
                legalize = re.search(r"failed to legalize operation '([^']*)'", c.stderr)
                if lowering:
                    reason = re.sub(r"'[^']*'", "'...'", lowering.group(1))  # "a call to 'f'" counts once
                    unlowered.append(src)
                    reasons[reason] += 1
                elif legalize:
                    unlowered.append(src)
                    reasons["op " + legalize.group(1)] += 1
                else:  # the front end's own errors: a program the interpreter rejects too
                    rejected.append(src)
                if args.verbose:
                    print(f"  not compiled  {os.path.basename(src)}: {message}")
                continue
            native = run([exe], stdin_path, args.timeout)
            interp = run([args.interpreter, src], stdin_path, args.timeout)
            if native == interp:
                match.append(src)
                if args.verbose:
                    print(f"  same          {os.path.basename(src)}")
            else:
                differ.append(src)
                print(f"DIFFERS: {src}")
                print(f"  exit: native {native[0]}, interpreter {interp[0]}")
                for label, a, b in (("stdout", native[1], interp[1]), ("stderr", native[2], interp[2])):
                    if a != b:
                        print(f"  {label}, native:      {a[:300]!r}")
                        print(f"  {label}, interpreter: {b[:300]!r}")

    total = len(match) + len(differ) + len(unlowered) + len(rejected)
    print(f"\n{total} programs: {len(match) + len(differ)} compiled natively, {len(match)} same as the interpreter, "
          f"{len(differ)} different; {len(unlowered)} not lowered yet; {len(rejected)} rejected by the front end"
          + (f"; {len(skipped)} flag tests skipped" if skipped else ""))
    if reasons:
        print("Not lowered yet, by first reason:")
        for reason, count in reasons.most_common():
            print(f"  {count:4}  {reason}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
