# Nova Quantum Compiler

**Nova Quantum** compiles quantum programs for **neutral-atom quantum computers**. Write circuits in
[Nova](https://marketplace.visualstudio.com/items?itemName=parhitmathur.nova-programming-language) (a `qubit` type, `qpu` gates, classical
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

Installing Nova Quantum also installs the **Nova Compiler** extension, which provides the Nova language
itself (errors as you type, hover, completion).

> **Preview.** The extension is a pre-release, for Linux (x64) and Windows (x64).

**New to it?** [Getting started](https://github.com/parhitmtech/NovaQuantum/blob/main/docs/GETTING_STARTED.md#7-a-first-quantum-program)
walks through a first circuit: its compiled numbers, running it, and the compiled circuit at each stage.

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

None on Linux (x64) and Windows (x64): the extension ships `nova-compile` and `nova-translate`, Nova's quantum
compiler and its OpenQASM translator, with LLVM and MLIR 20 built in, so they run with nothing else installed.

To use tools you built yourself instead (a newer compiler, say), build them from
[the repository](https://github.com/parhitmtech/NovaQuantum) and set their paths below:

```bash
cmake -G Ninja -S compiler/mlir -B build/mlir -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build/mlir nova-compile nova-translate
```

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `novaQuantum.compiler.path` | the one shipped | `nova-compile` |
| `novaQuantum.translate.path` | the one shipped | `nova-translate`, for Export as OpenQASM |
| `novaQuantum.codeLens.enabled` | `true` | Show each circuit's compiled numbers above it |

## Workspace Trust

In a folder you have not trusted, Nova Quantum still compiles circuits to show their errors and numbers (compiling
runs none of your code), but runs no program, and the folder's settings cannot change which tools are used.

## Troubleshooting

| What you see | What to do |
|---|---|
| No numbers above a circuit | Numbers appear above code that uses qubits, in a `.nova` or `.qasm` file; check `novaQuantum.codeLens.enabled`. |
| "Nova's quantum compiler (nova-compile) was not found", or "nova-translate was not found" | `novaQuantum.compiler.path` or `novaQuantum.translate.path` names a file that does not exist: clear it, and the extension uses the one it ships. |
| Nova's own errors, hover or completion missing | Those come from the **Nova Compiler** extension, installed with this one: see its troubleshooting. |

For anything else, please [report it](https://github.com/parhitmtech/NovaQuantum/issues) with the circuit, if you
can share it.

## Links

- Source and design: [github.com/parhitmtech/NovaQuantum](https://github.com/parhitmtech/NovaQuantum), and its
  [quantum design document](https://github.com/parhitmtech/NovaQuantum/blob/main/docs/QUANTUM.md)
- Report a bug: [issues](https://github.com/parhitmtech/NovaQuantum/issues)

## Privacy

The extension collects no telemetry. Your circuits are read only by the Nova tools on your machine.

## License

[Apache 2.0](LICENSE)
