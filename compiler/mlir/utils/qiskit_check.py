#!/usr/bin/env python3
"""NovaQuantum against Qiskit, through OpenQASM 3: an outside check of the whole quantum pipeline.

    qiskit_check.py --build ~/nova-build        (needs qiskit and qiskit-qasm3-import)

Two directions:

  Nova -> Qiskit   Each benchmark circuit (circuits.py) is exported to OpenQASM (nova-translate) twice: as written,
                   and as compiled for the neutral-atom machine (--qnova-to-native: decomposed into r, rz and cz,
                   optimized, routed). Qiskit reads each and computes its statevector; both must give the
                   probabilities Nova's simulator gives for the circuit. So the decompositions, the
                   canonicalizations and the export (r as U) are checked by a simulator that is not Nova's.

  Qiskit -> Nova   Qiskit's random circuits, in its standard gates, are written as OpenQASM (qiskit.qasm3.dumps),
                   read by nova-compile, and run three ways: as written, compiled to native gates, and compiled to
                   hardware instructions (--hardware). Each must give Qiskit's statevector probabilities.

Nova prints probabilities to four places, dropping those below 0.00005; a pair agrees when the two differ by less
than 0.0001. Exit status 1 if any circuit disagrees.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circuits  # noqa: E402
import qbench  # noqa: E402

from qiskit import qasm3, transpile  # noqa: E402
from qiskit.circuit.random import random_circuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

BASIS = ["h", "x", "y", "z", "s", "t", "sdg", "tdg", "rx", "ry", "rz", "p", "u", "cx", "cz", "swap", "cp"]


def nova_probs(text):
    """{'0101': p, ...} from Nova's |0101>: 0.2500 lines"""
    probs = {}
    for line in text.splitlines():
        if line.startswith("|") and ">:" in line:
            key, p = line[1:].split(">:")
            probs[key] = float(p)
    return probs


def qiskit_probs(circuit, qubits):
    """The probabilities of the listed qubits (the first leftmost, as Nova prints them), from Qiskit's statevector"""
    full = Statevector(circuit).probabilities()
    out = {}
    for i, p in enumerate(full):
        key = "".join(str((i >> q) & 1) for q in qubits)
        out[key] = out.get(key, 0.0) + p
    return out


def agree(nova, qiskit):
    """The largest difference, Nova's rounding allowed for"""
    worst = 0.0
    for key in set(nova) | set(qiskit):
        p_nova, p_qiskit = nova.get(key, 0.0), qiskit.get(key, 0.0)
        if key not in nova and p_qiskit < 0.0001:  # below what Nova prints
            continue
        worst = max(worst, abs(p_nova - p_qiskit))
    return worst


def run(cmd, **kwargs):
    p = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if p.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{p.stderr}")
    return p.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="the MLIR build directory")
    parser.add_argument("--random", type=int, default=24, help="how many random Qiskit circuits")
    args = parser.parse_args()
    tool = lambda name: os.path.join(args.build, "bin", name)  # noqa: E731
    failures, rows = 0, []

    with tempfile.TemporaryDirectory() as tmp:
        # Nova -> Qiskit
        for name, make in qbench.BENCHMARKS:
            c = make()
            path = os.path.join(tmp, name + ".mlir")
            with open(path, "w") as f:
                f.write(c.text())
            run([tool("nova-compile"), path, "--logical", "-o", os.path.join(tmp, "run")])
            nova = nova_probs(run([os.path.join(tmp, "run")]))
            shown = list(range(min(c.n, 12)))
            for form, passes in (("as written", ["--canonicalize"]), ("compiled", ["--qnova-to-native"])):
                mlir = run([tool("nova-opt"), path, *passes])
                qasm = run([tool("nova-translate"), "--qnova-to-openqasm"], input=mlir)
                diff = agree(nova, qiskit_probs(qasm3.loads(qasm), shown))
                ok = diff < 0.0001
                failures += not ok
                rows.append(("Nova -> Qiskit", f"{name}, {form}", c.n, diff, ok))

        # Qiskit -> Nova
        rng = random.Random(7)
        for k in range(args.random):
            n, depth = rng.randint(2, 10), rng.randint(2, 12)
            circuit = transpile(random_circuit(n, depth, max_operands=2, seed=1000 + k), basis_gates=BASIS,
                                optimization_level=0)
            path = os.path.join(tmp, f"random{k}.qasm")
            with open(path, "w") as f:
                f.write(qasm3.dumps(circuit))
            expected = qiskit_probs(circuit, list(range(n)))
            for form, flags in (("as written", ["--logical"]), ("native", []), ("hardware", ["--hardware"])):
                run([tool("nova-compile"), path, *flags, "-o", os.path.join(tmp, "run")])
                diff = agree(nova_probs(run([os.path.join(tmp, "run")])), expected)
                ok = diff < 0.0001
                failures += not ok
                rows.append(("Qiskit -> Nova", f"random {k} ({len(circuit.data)} gates), {form}", n, diff, ok))

    print(f"{'direction':<16}{'circuit':<38}{'qubits':>7}{'max |dp|':>11}")
    for direction, name, n, diff, ok in rows:
        print(f"{direction:<16}{name:<38}{n:>7}{diff:>11.1e}{'' if ok else '  DIFFERENT'}")
    print(f"\n{len(rows)} comparisons with Qiskit, {failures} different")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
