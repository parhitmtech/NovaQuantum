# Nova Quantum

**Nova Quantum** compiles quantum programs for **neutral-atom quantum computers**. Write circuits in
[Nova](https://marketplace.visualstudio.com/items?itemName=parhitmathur.nova) (a `qubit` type, `qpu` gates, classical
feedback with `if`) or in **OpenQASM 3**, as Qiskit and CUDA-Q write it. The compiler, built on MLIR and LLVM,
decomposes circuits into the machine's native gates, optimizes them, routes qubits by moving atoms, and schedules
them into timed hardware instructions; a statevector simulator checks every compiled circuit against the original.

```nova
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
itself (errors as you type, hover, completion).

> **Preview.** The extension is a pre-release, and needs Nova's quantum compiler on your machine for now (see
> [Requirements](#requirements)).

## Features

- **The cost of each circuit, above it:** its native gates, depth, atom moves and time on the machine, as compiled
  for the neutral-atom machine, updated as you type. Click it to see the hardware instructions.
- **Run on Simulator** (Ctrl+F5 in a `.qasm` file, or the lens above the circuit): compiles the program to native
  code and runs it on the statevector simulator, printing the probabilities.
- **Compile for Hardware and Run:** down to the machine's timed instructions, with their layers and time.
- **Show the Compiled Circuit** at each stage: as written (`qnova`), as native gates (decomposed, optimized, routed,
  scheduled), or as the machine's timed instructions (`atom`).
- **Export as OpenQASM:** a Nova circuit, written as OpenQASM 3 for Qiskit or CUDA-Q.
- **OpenQASM 3:** highlighting, and errors as you type (from Nova's compiler, which reads the circuit).
- **Quantum words highlighted in Nova:** `qubit`, `qpu` and its gates.

## Requirements

The extension runs `nova-compile` and `nova-translate`, Nova's quantum compiler and its OpenQASM translator, built
on LLVM and MLIR 20. Build them from [the repository](https://github.com/parhitmtech/NovaQuantum):

```bash
cmake -G Ninja -S compiler/mlir -B build/mlir -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build/mlir nova-compile nova-translate
```

Then put `build/mlir/bin` on your `PATH`, or set the paths below.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `novaQuantum.compiler.path` | the one on `PATH` | `nova-compile` |
| `novaQuantum.translate.path` | the one on `PATH` | `nova-translate`, for Export as OpenQASM |
| `novaQuantum.codeLens.enabled` | `true` | Show each circuit's compiled numbers above it |

## Workspace Trust

In a folder you have not trusted, Nova Quantum still compiles circuits to show their errors and numbers (compiling
runs none of your code), but runs no program, and the folder's settings cannot change which tools are used.

## Links

- Source and design: [github.com/parhitmtech/NovaQuantum](https://github.com/parhitmtech/NovaQuantum), and its
  [quantum design document](https://github.com/parhitmtech/NovaQuantum/blob/main/docs/QUANTUM.md)
- Report a bug: [issues](https://github.com/parhitmtech/NovaQuantum/issues)

## Privacy

The extension collects no telemetry. Your circuits are read only by the Nova tools on your machine.

## License

[Apache 2.0](LICENSE)
