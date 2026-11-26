# Benchmarks

These numbers show how fast Nova runs today next to CPython, Node and C. Running the comparison also
turned up several correctness issues, listed below with minimal reproductions.

## Method

The suite is in [`compiler/benchmarks/compare/`](../compiler/benchmarks/compare/). Each benchmark is the same
algorithm in Nova, Python, JavaScript and C, and each program prints its result so the outputs can be
compared.

```bash
make -C compiler
python3 compiler/benchmarks/compare/run.py --optimize      # --runs N, --timeout S
```

`run.py` compiles the C files, times each program (wall clock, best of N runs, including process startup),
skips languages that aren't installed, and prints a Markdown table. It also flags runs that time out or whose
output differs from the other languages.

| Benchmark | Measures |
|---|---|
| `calls` | Naive recursive `fib(30)`, about 2.7M function calls |
| `int_loop` | A 3M-iteration `while` loop of integer `+`, `-` and `/` |
| `vector_sum` | `push` 1M ints into a `vector<int>`, then read them all back with `%` |

## Results

Measured on an Intel Core Ultra 7 155H (Windows 11) with Python 3.13.6, Node 22.18.0 and gcc 15.2 (MSYS2).
The compiler was built from `main` at `c5f25a8`. Times are best of 7. Starting a process costs about 15 ms
here, so the C column is almost all startup.

| Benchmark | Nova | Nova `--optimize` | CPython | Node | C `-O2` |
|---|---|---|---|---|---|
| `calls` | 1,004 ms | 1,005 ms | 134 ms | 56 ms | 17 ms |
| `int_loop` | 134 ms ⚠️ | 135 ms ⚠️ | 486 ms | 54 ms | 17 ms |
| `vector_sum` | 93 ms | 97 ms | 192 ms | 64 ms | 17 ms |

The first measurement (Apple M4 Pro, `main` at `e2d63e6`) gave the same picture, except that `vector_sum` with
`--optimize` timed out (issue 1, since fixed).

⚠️ Nova prints `-562863104` here; the correct answer is `2250000000000` (issue 2).

What the numbers say:

- **Loops and vectors:** the interpreter runs straight-line integer code 2–3.5× faster than CPython, close to
  Node.
- **Function calls are the bottleneck:** about 7.5× slower than CPython. Each call in
  `runtime/interp_calls.cc`:
  - builds a `CallFrame` that heap-allocates several `std::vector`s (`savedSlots`, `allRetSlots`,
    `memo_key`);
  - copies the callee's whole slot range into the frame and back on return;
  - does hash-set lookups (`pure_functions`, `recursive_functions`, `g_tensorSlots`) on every call and for
    every parameter;
  - converts every argument through `double`.

  Moving the per-call-site decisions into lowering, and using a contiguous frame stack instead of copying
  slots in and out, should remove most of this cost: that is phase 6 of the [roadmap](ROADMAP.md) (frames on
  a value stack), with phase 7's Nova IR.
- **`--optimize` doesn't change these timings.** Its passes simplify the IR, but none of them reduce
  per-instruction interpreter overhead.

### Caveats about the older benchmarks

`compiler/benchmarks/` also has an older Nova vs. Python vs. C suite (`run_benchmarks.sh`). Two of its programs
don't measure the same work in each language:

- **`fibonacci.nova`** is a pure recursive function, so Nova memoizes it automatically. `fib(35)` then computes each
  `fib(n)` once instead of making about 30 million calls, while the C and Python versions do the full recursion. Memoization is
  a real feature, but this comparison measures memoization, not the compiler. `compare/calls.nova` adds a
  never-taken `print` to switch memoization off.
- **`matrix_multiply.nova`** calls the built-in C++ `matmul`, while `matrix_multiply.py` multiplies with
  Python loops. That compares Nova's runtime library with the CPython interpreter. NumPy's `@` would be the
  fair Python baseline.

## Issues found

Found on `main` at `e2d63e6`, and rechecked at `c5f25a8`: issue 1 is fixed, and issues 2–5 still reproduce.

### 1. `--optimize` never finishes on a loop that does arithmetic on a vector element (fixed)

Fixed in #8. Loop-invariant code motion linked the instruction list into a cycle whenever it moved an
instruction out of a loop (`v[i] * 7`'s multiplier was the first invariant it found here), so the passes after
it never finished. The pass was rewritten; `licm_loops` covers it, and the golden tests run with `--optimize` in
CI. The original report:

```python
ignite:
    vector<int> v = vector<int>()
    v.push(9)
    int i = 0
    int s = 0
    while (i < 1):
        s = v[i] * 7          # also with %, and with s = s + v[i] % 7
        i = i + 1
    print(s)
```

Without `--optimize` it prints `63` at once. With `--optimize` it never finishes, and memory keeps
growing: `vector_sum` with only 1,000 elements reached about 780 MB after 90 s. The same expression outside a
loop works.
CI already runs the golden tests with `--optimize`, so adding this program as a test would catch any
regression.

### 2. `int` is 32-bit and overflows silently

`int_loop` sums to 2,250,000,000,000. Nova prints `-562863104` with no warning. Options: make `int` 64-bit
(`long long` slots), trap on overflow the way Rust's debug builds do, or both.

### 3. Memoization skips side effects of the functions it calls

```python
def log(int n) -> int:
    print(n)
    return 0

def f(int n) -> int:
    int z = log(n)
    if (n < 1):
        return 0
    return f(n - 1) + f(n - 1)

ignite:
    print(f(3))
```

This should print 15 log lines, then `0`. It prints `3 2 1 0` and `0`: `f` is memoized, so the repeated
calls never reach `log`. `detect_memoizable_functions` (`runtime/memoize.cc`) only checks a function's own
instructions for `print`, input, tensor ops, array writes and allocation. It doesn't follow `CALL`s. Purity
needs to be computed over the call graph: a function is pure only if every function it calls is pure.

### 4. Training memory grows without bound

The tensor-free pass works: in `examples/xor.nova` only 8 tensors stay live. But `tensor_release`
deliberately keeps each freed `tensor_heap` entry so stale handles fail loudly, and entries are never
reused. The heap therefore grows by about 28 entries per epoch:

| XOR epochs | `tensor_heap` entries (`NOVA_TENSOR_STATS=1`) | Peak memory (RSS) |
|---|---|---|
| 1,000 | 28,012 | — |
| 5,000 | 140,012 | — |
| 10,000 | — | 75 MB |
| 50,000 | — | 312 MB |

Production workers run with `--memory=256m`, so a 50,000-epoch XOR run would be killed as out of memory, and
MNIST-sized training would hit the limit much sooner. **Generational handles** would fix this and keep the
loud failure: store a generation counter in each heap entry and in the handle (for example, the high bits),
reuse freed entries through a free list, and treat a handle whose generation doesn't match as stale.

### 5. Documented syntax that isn't implemented yet

[SYNTAX.md](SYNTAX.md) describes the target design, and its first example uses features the parser doesn't
support yet:

| Documented | Today |
|---|---|
| `10_000` digit separators | Parse error (`expected the end of the statement, got '_000'`) |
| `layer1.W.grad` | `a Tensor has no member 'grad'`; use `tensor_grad(t)` |
| `print(pred)` for a tensor | Fixed: prints the elements, as `tensor_print(t)` does (found by differential testing, below) |
| `"\n"`, `"\t"` escapes | Printed literally (`compiler/tests/grad_stress_loop.expected` currently expects `\n`) |

Either implement these or mark them "planned" in SYNTAX.md, so readers who copy the first example get a
program that runs.

## Native code through MLIR and LLVM

`nova-compile` compiles a Nova program through the MLIR back end and LLVM to a native executable (`clang -O2`,
linked with Nova's C runtime). [`compiler/mlir/utils/benchmark.py`](../compiler/mlir/utils/benchmark.py) times
each program interpreted and native: the median of 7 runs, process start included. It also times the same
algorithm in C, compiled by the same clang, and it refuses to time a program whose two outputs differ.

```bash
python3 compiler/mlir/utils/benchmark.py --build build/mlir --interpreter compiler/build/nova \
        compiler/benchmarks/compare/calls.nova compiler/benchmarks/compare/int_loop.nova compiler/benchmarks/fibonacci.nova
```

Intel Core Ultra 7 155H, WSL2 Ubuntu 24.04, LLVM 20:

| Program | Interpreter | Nova native | Speedup | Same in C, clang -O2 |
|---|---|---|---|---|
| `calls` (naive fib(30), 1.3M calls) | 633 ms | 2.9 ms | 215× | 2.5 ms |
| `int_loop` (3M iterations) | 159 ms | 1.0 ms | 157× | 2.1 ms |
| `fibonacci` (fib(35)) | 53 ms | 26 ms | 2.1× | 34 ms |

- **Native code is on par with C.** The rest of the gap is the C runtime calls and Nova's checked integer division.
  For a constant divisor, canonicalize removes the check: `int_loop` went from 3.7 ms to 1.0 ms once
  `nova-compile` ran canonicalize before lowering.
- **`int_loop` is not the same computation in both languages.** The C version uses 64-bit integers to get the right
  sum; Nova's `int` is 32 bits and wraps (issue 2). Native and interpreted Nova wrap identically.
- **fib(35) is only 2.1× faster** because the interpreter memoizes pure functions automatically (issue 3). It skips
  most of the 30M calls that native code makes, and native code still finishes faster than C.

### Differential testing

[`compiler/mlir/utils/difftest.py`](../compiler/mlir/utils/difftest.py) compiles every program it is given, runs it
natively and in the interpreter, and requires the same stdout, stderr and exit status. CI runs it over the golden
tests, the benchmarks and `compiler/mlir/test/Run`:

| Programs | Count |
|---|---|
| Compiled natively, same as the interpreter | 28 |
| Compiled natively, different | 0 |
| Not lowered yet (first reason: classes and structs 37, autograd 15, strings 7, containers 8, other built-ins 32) | 99 |
| Rejected by the front end (error tests, and 6 programs that use `orbit` imports, which `nova-compile` does not resolve yet) | 95 |

It found two bugs, both fixed:

1. **The interpreter's `print(t)` printed a tensor's handle number.** `print` sent a `Tensor` down the integer path.
   The native runtime printed the elements, so the outputs differed. The interpreter now prints the elements too.
   Built before and after, the interpreter gives identical output on all 216 golden tests.
2. **Integer division by zero was undefined behavior natively.** `arith.divsi` by zero, like LLVM's `sdiv`, is
   undefined, and the binary printed 1 where the interpreter stops with "integer division by zero". Integer `/` and
   `%` now emit `nova.check_divisor` (a runtime check, which canonicalize drops for a nonzero constant divisor) and
   divide by 1 in the `INT_MIN / -1` case, which wraps to `INT_MIN` as the interpreter does.

While making these pass, the harness also caught a third problem, in the new compiler itself. Its tensor ops
declared only `MemAlloc` and `MemRead` effects, so canonicalize deleted an unused `a * b` whose shape mismatch was
the error the test expected. Every op that can raise a runtime error now writes a `nova.runtime_error` resource
([DESIGN.md](DESIGN.md), C9).

## Quantum compilation for a neutral-atom machine

[`compiler/mlir/utils/qbench.py`](../compiler/mlir/utils/qbench.py) compiles each benchmark circuit
([`circuits.py`](../compiler/mlir/utils/circuits.py)) for the neutral-atom machine. It reports `--qnova-stats` at three
points:
- **logical:** the circuit as written;
- **native:** after `--qnova-decompose` alone;
- **compiled:** after the whole `--qnova-to-native` pipeline.

It then writes the compiled circuit as the machine's timed instructions, and runs the circuit three ways on the
runtime's statevector simulator: as written, compiled, and as hardware instructions by site. The last two must give
the first's probabilities, to four places. CI runs it. Design: [QUANTUM.md](QUANTUM.md).

Default grid (about twice as many sites as atoms):

| Circuit | Qubits | Logical gates | Native | Compiled | Saved | cz | Moves | Layers | Gate time | Move time | Native, hardware = logical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GHZ | 8 | 8 | 37 | 29 | 22% | 7 | 4 | 17 | 7.3 µs | 25.0 µs | ✓ ✓ |
| GHZ | 12 | 12 | 57 | 45 | 21% | 11 | 7 | 26 | 10.9 µs | 80.0 µs | ✓ ✓ |
| QFT, then its inverse | 5 | 33 | 283 | 3 | 99% | 0 | 0 | 1 | 1.0 µs | 0 | ✓ ✓ |
| QFT, then its inverse | 8 | 76 | 764 | 4 | 99% | 0 | 0 | 1 | 1.0 µs | 0 | ✓ ✓ |
| random | 6 | 60 | 260 | 203 | 22% | 40 | 8 | 141 | 64.4 µs | 256.8 µs | ✓ ✓ |
| random | 8 | 120 | 483 | 355 | 27% | 68 | 26 | 244 | 109.5 µs | 725.2 µs | ✓ ✓ |
| random | 10 | 200 | 692 | 548 | 21% | 100 | 33 | 314 | 146.5 µs | 714.1 µs | ✓ ✓ |
| random | 12 | 300 | 1026 | 867 | 15% | 166 | 72 | 496 | 219.3 µs | 1727.4 µs | ✓ ✓ |

Gate time (Raman and Rydberg layers) and move time are the hardware program's (`--qnova-to-atom`, default timing
model), without the final readout. Moves take 3–8× as long as all the gates together.

A tighter grid, 4 sites wide (`qbench.py --width 4`), changes only the routing:

| Circuit | Moves (default grid) | Moves (width 4) | Layers (default) | Layers (width 4) |
|---|---|---|---|---|
| random, 10 qubits | 33 | 29 | 314 | 316 |
| random, 12 qubits | 72 | 70 | 496 | 509 |

- **"Saved"** is the share of native gates removed by canonicalization: inverse pairs, merged rotations, diagonal
  gates commuted through `cz`, and unobservable diagonal gates dropped.
- **The QFT round trip** collapses to the `x` gates that prepare its input once canonicalization runs on the logical
  gates before decomposition. Run on the native gates only, it saved 24–26%. That gap is why `--qnova-to-native`
  canonicalizes twice.
- **The first scheduler** placed atoms densely and failed to route the 12-qubit random circuit: every site next to an
  atom was taken. Checkerboard placement on a sparser grid, with two fallbacks (move the other atom; park one atom
  where there is room), routes every circuit here.
- **Running the hardware instructions found a scheduler bug.** On `random-6x60`, a move into a site was scheduled
  before the move out of it, so two atoms would have collided (the simulator stopped with "an atom moved onto
  another"). The gate level could not see it, because qubit values carry no sites. A move now waits for the layer
  after the move that vacated its site; layer counts went up by 1–23.

## The statevector simulator on a GPU

[`compiler/mlir/runtime/gpu/qsim_bench.cu`](../compiler/mlir/runtime/gpu/qsim_bench.cu) runs the same circuits on
three backends:
- the CPU simulator with one thread;
- the CPU simulator with all cores (OpenMP);
- the CUDA simulator.

Up to 24 qubits it compares the GPU's final state with the CPU's, every amplitude. The two circuits:
- **layers:** `depth` = 10 layers, each an `ry` and an `rz` on every qubit, then a `cz` brickwork. This is the shape of
  a variational circuit.
- **qft:** the quantum Fourier transform.

The time is the gates' alone: the state is allocated before the clock starts. Effective bandwidth is the bytes each
gate's kernel must move (a one-qubit gate reads and writes every amplitude, 32 bytes each; a `cz` or `cp` a quarter
of them), over the time.

SOL, one NVIDIA A100-SXM4-80GB (peak memory bandwidth 2,039 GB/s), against 16 cores of an AMD EPYC 7413
(`compiler/mlir/runtime/gpu/run_sol.sh`):

| Circuit | Qubits | Gates | CPU, 1 thread | CPU, 16 threads | GPU | vs 1 thread | vs 16 threads | GPU bandwidth | % of peak | Max \|GPU − CPU\| |
|---|---|---|---|---|---|---|---|---|---|---|
| layers | 16 | 395 | 146.3 ms | 873.4 ms* | 1.32 ms | 111× | – | 539 GB/s | 26% | 2.8e-17 |
| qft | 16 | 136 | 16.9 ms | 2.9 ms | 0.42 ms | 40× | 6.8× | 230 GB/s | 11% | 0 |
| layers | 18 | 445 | 606.6 ms | 55.1 ms | 2.00 ms | 304× | 27.6× | 1,601 GB/s | 79% | 2.1e-17 |
| qft | 18 | 171 | 80.8 ms | 7.5 ms | 0.62 ms | 130× | 12.0× | 759 GB/s | 37% | 0 |
| layers | 20 | 495 | 2,739.6 ms | 193.1 ms | 5.18 ms | 529× | 37.3× | 2,746 GB/s† | 135%† | 1.4e-17 |
| qft | 20 | 210 | 388.3 ms | 31.0 ms | 1.37 ms | 282× | 22.5× | 1,648 GB/s | 81% | 0 |
| layers | 22 | 545 | 12,290.5 ms | 846.1 ms | 39.77 ms | 309× | 21.3× | 1,574 GB/s | 77% | 1.3e-17 |
| qft | 22 | 253 | 2,005.6 ms | 146.1 ms | 6.80 ms | 295× | 21.5× | 1,574 GB/s | 77% | 0 |
| layers | 24 | 595 | 53,556.0 ms | 5,811.4 ms | 166.94 ms | 321× | 34.8× | 1,636 GB/s | 80% | 9.8e-18 |
| qft | 24 | 300 | 9,290.7 ms | 1,305.8 ms | 36.36 ms | 256× | 35.9× | 1,373 GB/s | 67% | 0 |
| layers | 26 | 645 | – | – | 714.14 ms | – | – | 1,658 GB/s | 81% | – |
| qft | 26 | 351 | – | – | 164.50 ms | – | – | 1,400 GB/s | 69% | – |
| layers | 28 | 695 | – | – | 3,077.35 ms | – | – | 1,657 GB/s | 81% | – |
| qft | 28 | 406 | – | – | 760.78 ms | – | – | 1,383 GB/s | 68% | – |

\* The first timed run started OpenMP's thread pool; the benchmark now warms both backends up before timing.
† The state, 16 MB, fits in the A100's 40 MB L2 cache, so the kernels run faster than DRAM bandwidth allows.

**Correctness.** At every size up to 24 qubits, the GPU's final state matches the CPU's to 2.8 × 10⁻¹⁷, which is
floating-point rounding: the two compute each amplitude in the same order. `make dropin-check` runs a compiled
circuit's runtime calls through the CPU runtime and through the GPU runtime: teleportation, a rotated six-qubit GHZ
state, and a Bell pair from hardware instructions by site. The two print byte-identical output.

**Three regimes**, as a memory-bound code should show:
- **Up to 18 qubits (≤ 4 MB), launch-bound.** Each gate is one kernel over a small array, so a few microseconds of
  launch and scheduling per gate dominate: 11–37% of peak at 16 qubits. Fusing consecutive one-qubit gates on a
  qubit into one matrix (the compiler already sees them as one SSA chain), or capturing the circuit in a CUDA graph,
  would cut the launches.
- **20 qubits (16 MB), L2-resident.** The state lives in the 40 MB L2 cache, and the layered circuit runs at 135% of
  the DRAM peak.
- **22–28 qubits (64 MB–4 GB), DRAM-bound.** The steady state: the layered circuit sustains 1.57–1.66 TB/s, 77–81%
  of the A100's peak. The QFT reports 67–77% on the same model, but its `cp` kernels touch a quarter of the amplitudes
  in a strided pattern, so whole 32-byte sectors are fetched for a quarter of their data. The bytes really moved are
  more than the model counts.

**Speedups.** At 20–24 qubits, the GPU is 256–529× faster than one CPU thread and 21–37× faster than all 16
cores. A 28-qubit, 695-gate circuit (a 4 GiB state) runs in 3.1 s. On the CPU with 16 threads, the 24-qubit one
alone takes 5.8 s.

## Against Qiskit, through OpenQASM

[`compiler/mlir/utils/qiskit_check.py`](../compiler/mlir/utils/qiskit_check.py) checks the quantum pipeline with a
simulator that isn't Nova's: Qiskit 2.6's statevector. Circuits cross in OpenQASM 3 (`nova-translate`).

| Direction | Circuits | Forms compared | Comparisons | Different | Largest difference |
|---|---|---|---|---|---|
| Nova → Qiskit | the 8 benchmark circuits (GHZ, QFT round trips, random, up to 12 qubits) | as written; compiled for the machine | 16 | 0 | 5.0e-05 |
| Qiskit → Nova | 24 random circuits (2–10 qubits, 11–271 gates) | as written; native gates; hardware instructions | 72 | 0 | 5.0e-05 |

The largest differences are Nova's four-place printing; where a circuit's probabilities are exact (GHZ, the QFT
round trip), the two agree to 10⁻¹⁶.
