#!/usr/bin/env python3
"""Benchmark circuits as qnova MLIR: GHZ states, QFT round trips and random circuits.

    circuits.py ghz 8 > ghz8.mlir
    circuits.py qft 6 > qft6.mlir          (QFT then its inverse, on a basis state: it comes back to that state)
    circuits.py random 8 120 7 > r.mlir    (8 qubits, 120 gates, seed 7)

Each ends with qnova.probs on its first 12 qubits, so the circuit's output is a probability distribution that the
logical circuit and its compiled form must both give (qbench.py checks that).
"""
import math
import random
import sys


class Circuit:
    def __init__(self, n):
        self.n = n
        self.lines = []
        self.wire = [f"%q{k}_0" for k in range(n)]  # the SSA value each qubit's state is now
        self.version = [0] * n
        self.constants = {}
        for k in range(n):
            self.lines.append(f"  {self.wire[k]} = qnova.alloc")

    def _next(self, k):
        self.version[k] += 1
        self.wire[k] = f"%q{k}_{self.version[k]}"
        return self.wire[k]

    def angle(self, radians):
        name = f"%a{len(self.constants)}"
        self.constants[name] = radians
        return name

    def one(self, gate, k, *angles):
        old = self.wire[k]
        new = self._next(k)
        args = "".join(f", {self.angle(a)}" for a in angles)
        self.lines.append(f"  {new} = qnova.{gate} {old}{args}")

    def two(self, gate, a, b, *angles):
        oa, ob = self.wire[a], self.wire[b]
        na, nb = self._next(a), self._next(b)
        args = "".join(f", {self.angle(x)}" for x in angles)
        self.lines.append(f"  {na}, {nb} = qnova.{gate} {oa}, {ob}{args}")

    def text(self):
        shown = ", ".join(self.wire[: min(self.n, 12)])
        head = [f"  {name} = arith.constant {value!r} : f64" for name, value in self.constants.items()]
        tail = [f"  qnova.probs {shown}", "  return", "}"]
        return "\n".join(["func.func @main() {"] + head + self.lines + tail) + "\n"


def ghz(n):
    c = Circuit(n)
    c.one("h", 0)
    for k in range(1, n):
        c.two("cx", k - 1, k)
    return c


def qft(c, qubits, inverse=False):
    n = len(qubits)
    steps = []
    for i in range(n):
        steps.append(("h", i))
        for j in range(i + 1, n):
            steps.append(("cp", j, i, math.pi / 2 ** (j - i)))
    if inverse:
        steps = [(s[0], *s[1:-1], -s[-1]) if s[0] == "cp" else s for s in reversed(steps)]
    for s in steps:
        if s[0] == "h":
            c.one("h", qubits[s[1]])
        else:
            c.two("cp", qubits[s[1]], qubits[s[2]], s[3])


def qft_roundtrip(n):
    c = Circuit(n)
    for k in range(0, n, 2):  # the basis state |1010...>
        c.one("x", k)
    qft(c, list(range(n)))
    qft(c, list(range(n)), inverse=True)
    return c


def random_circuit(n, gates, seed):
    rng = random.Random(seed)
    c = Circuit(n)
    for _ in range(gates):
        kind = rng.choice(["h", "x", "s", "t", "rx", "ry", "rz", "cx", "cx", "cz", "cp", "swap"])
        if kind in ("cx", "cz", "cp", "swap"):
            a, b = rng.sample(range(n), 2)
            c.two(kind, a, b, *([rng.uniform(-math.pi, math.pi)] if kind == "cp" else []))
        else:
            c.one(kind, rng.randrange(n), *([rng.uniform(-math.pi, math.pi)] if kind in ("rx", "ry", "rz") else []))
    return c


def main():
    kind = sys.argv[1]
    if kind == "ghz":
        c = ghz(int(sys.argv[2]))
    elif kind == "qft":
        c = qft_roundtrip(int(sys.argv[2]))
    elif kind == "random":
        c = random_circuit(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        sys.exit(f"unknown circuit kind {kind!r}: ghz, qft or random")
    sys.stdout.write(c.text())


if __name__ == "__main__":
    main()
