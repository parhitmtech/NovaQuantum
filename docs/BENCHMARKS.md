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

It then runs the logical and the compiled circuit on the runtime's statevector simulator and requires the same
probabilities to four places. CI runs it. Design: [QUANTUM.md](QUANTUM.md).

Default grid (about twice as many sites as atoms):

| Circuit | Qubits | Logical gates | Native | Compiled | Saved | cz | Moves | Depth | Compiled = logical |
|---|---|---|---|---|---|---|---|---|---|
| GHZ | 8 | 8 | 37 | 29 | 22% | 7 | 4 | 17 | ✓ |
| GHZ | 12 | 12 | 57 | 45 | 21% | 11 | 7 | 25 | ✓ |
| QFT, then its inverse | 5 | 33 | 283 | 3 | 99% | 0 | 0 | 1 | ✓ |
| QFT, then its inverse | 8 | 76 | 764 | 4 | 99% | 0 | 0 | 1 | ✓ |
| random | 6 | 60 | 260 | 203 | 22% | 40 | 8 | 140 | ✓ |
| random | 8 | 120 | 483 | 355 | 27% | 68 | 26 | 242 | ✓ |
| random | 10 | 200 | 692 | 548 | 21% | 100 | 33 | 312 | ✓ |
| random | 12 | 300 | 1026 | 867 | 15% | 166 | 72 | 473 | ✓ |

A tighter grid, 4 sites wide (`qbench.py --width 4`), changes only the routing:

| Circuit | Moves (default grid) | Moves (width 4) | Depth (default) | Depth (width 4) |
|---|---|---|---|---|
| GHZ, 12 qubits | 7 | 7 | 25 | 27 |
| random, 10 qubits | 33 | 29 | 312 | 311 |
| random, 12 qubits | 72 | 70 | 473 | 482 |

- **"Saved"** is the share of native gates removed by canonicalization: inverse pairs, merged rotations, diagonal
  gates commuted through `cz`, and unobservable diagonal gates dropped.
- **The QFT round trip** collapses to the `x` gates that prepare its input once canonicalization runs on the logical
  gates before decomposition. Run on the native gates only, it saved 24–26%. That gap is why `--qnova-to-native`
  canonicalizes twice.
- **The first scheduler** placed atoms densely and failed to route the 12-qubit random circuit: every site next to an
  atom was taken. Checkerboard placement on a sparser grid, with two fallbacks (move the other atom; park one atom
  where there is room), routes every circuit here.
