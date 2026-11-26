# Nova Quantum

> **Preview.** This early release sets up OpenQASM 3 in VS Code. Running circuits on the simulator and compiling
> them for hardware arrive in the next releases (see the roadmap below).

**Nova Quantum** compiles quantum programs for **neutral-atom quantum computers**. Write circuits in
[Nova](https://marketplace.visualstudio.com/items?itemName=parhitmathur.nova) (a `qubit` type, `qpu` gates, classical
feedback with `if`) or in **OpenQASM 3**, as Qiskit and CUDA-Q write it. The compiler, built on MLIR and LLVM,
decomposes circuits into the machine's native gates, optimizes them, routes qubits by moving atoms, and schedules
them into timed hardware instructions. A statevector simulator checks every compiled circuit against the original.

```python
ignite:                                 # quantum teleportation
    qubit a = qpu.alloc()
    qubit b = qpu.alloc()
    qubit c = qpu.alloc()
    qpu.ry(a, 2.0943951023931953)       # the state to send: |1> with probability 0.75
    qpu.h(b)
    qpu.cx(b, c)
    qpu.cx(a, b)
    qpu.h(a)
    bool m1 = qpu.measure(a)
    bool m2 = qpu.measure(b)
    if (m2):
        qpu.x(c)
    if (m1):
        qpu.z(c)
    qpu.probs(c)                         # |1>: 0.7500, whatever was measured
```

Installing Nova Quantum also installs the **Nova Programming Language** extension, which provides the Nova language
itself.

## In this release

- `.qasm` files (and files whose first line is `OPENQASM`) are recognized as OpenQASM.
- Comments (`//`, `/* */`), bracket matching and auto-closing, and indentation inside `{ }`.

## Roadmap

| | What it adds |
|---|---|
| Next | OpenQASM highlighting and errors as you type; **Run on Simulator** for Nova and OpenQASM |
| Then | **Compile for Hardware** with gate counts and timing above each program; the compiled IR (`qnova`, native gates, `atom` instructions); export to OpenQASM; the GPU simulator |
| 1.0 | Stable release on Linux and Windows |

## Links

- Source and design docs: [github.com/parhitmtech/NovaQuantum](https://github.com/parhitmtech/NovaQuantum)
- Report a bug: [issues](https://github.com/parhitmtech/NovaQuantum/issues)
- Security issues: see [SECURITY.md](https://github.com/parhitmtech/NovaQuantum/blob/main/SECURITY.md)

## Privacy

This extension collects no telemetry.

## License

[Apache 2.0](LICENSE)
