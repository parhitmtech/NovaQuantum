# Compiling quantum circuits for a neutral-atom machine

NovaQuantum adds a second MLIR dialect to Nova's compiler, `qnova`. Its job is to take a quantum circuit, written in
Nova (`qubit`, `qpu.h(q)`, ...) or directly in MLIR, and compile it for a neutral-atom quantum computer: decompose it
into the machine's native gates, optimize it, route it by moving atoms, and schedule it into hardware layers. A
statevector simulator in the C runtime then runs both the circuit as written and the compiled one, and checks that
they agree.

All of it reuses the infrastructure the rest of the compiler is built on: ODS (TableGen) definitions, MLIR's pattern
rewriting and pass manager, the lowering to LLVM through a C runtime, and lit/FileCheck tests.

```
circuit.mlir (logical: h, cx, cp, swap, rx, ...)
  │  canonicalize          cancel and merge logical gates (h h, cx cx, cp cp, ...)
  │  --qnova-decompose     every gate as native r(θ, φ), rz(θ), cz
  │  canonicalize          cancel and merge native gates; drop unobservable ones
  │  --qnova-schedule      place atoms, insert moves, assign hardware layers
  │  --qnova-to-atom       timed hardware instructions on sites (the atom dialect), with nova-compile --hardware
  ▼
native circuit or hardware program  ──--convert-nova-to-runtime──▶  simulator calls  ──LLVM──▶  executable
```

`nova-compile circuit.mlir -o circuit` runs all of it; `--logical` skips the compilation, so the circuit runs as
written; `--qnova-stats` prints what the compiled circuit costs.

## Writing circuits in Nova

Circuits are written in Nova itself, in the style Python quantum libraries use: a `qubit` variable, and gates from
the built-in `qpu` module that change the qubits they are given.

```python
# Quantum teleportation (examples/quantum/teleport.nova): the state of a moves to c
ignite:
    qubit a = qpu.alloc()
    qubit b = qpu.alloc()
    qubit c = qpu.alloc()
    qpu.ry(a, 2.0943951023931953)   # the state to send: |1> with probability 0.75
    qpu.h(b)
    qpu.cx(b, c)                     # b and c entangled
    qpu.cx(a, b)
    qpu.h(a)
    bool m1 = qpu.measure(a)
    bool m2 = qpu.measure(b)
    if (m2):                         # classical feedback: gates that depend on a measurement
        qpu.x(c)
    if (m1):
        qpu.z(c)
    qpu.probs(c)                     # |0>: 0.2500  |1>: 0.7500, whatever was measured
```

`nova-compile examples/quantum/teleport.nova -o teleport` compiles it for the machine and runs it on the simulator.
The gates are `h x y z s t`, `rx ry rz` (with an angle), `cx cz swap`, and `cp` (with an angle). There are also
`qpu.alloc()`, `qpu.measure(q)`, which gives a `bool`, and `qpu.probs(a, b, ...)`.

**How the language maps to SSA.** In Nova a gate changes a variable in place; in `qnova` a gate consumes a value and
produces a new one. The bridge is the SSA construction the compiler already uses for ordinary variables
([DESIGN.md](DESIGN.md), C7). Lowering keeps, for each variable, the SSA value it holds at this point:
- an assignment `x = ...` makes a new value for `x`;
- a gate `qpu.h(q)` makes the new value `%q1 = qnova.h %q0` for `q`.

Because of that, control flow needs nothing new:
- **Loops.** A loop that applies a gate to `q` carries `q` as an `scf.while` block argument, exactly as it carries a
  counter.
- **Ifs.** An `if` that applies a gate carries `q` out as an `scf.if` result, a phi of the wire. Teleportation's
  corrections lower to two `scf.if` ops with a `!qnova.qubit` result each.

**No-cloning at two levels.**
- **In the language,** the check pass rejects any Nova code that would copy a quantum state:
  - a `qubit` that doesn't start from `qpu.alloc()`;
  - assigning a qubit (`a = b`);
  - passing a qubit to a function, or returning one (a function gets a copy of a value);
  - giving one gate the same qubit twice (`qpu.cx(a, a)`).

  Each has its own error message, for example "a quantum state cannot be copied (no-cloning)".
- **In the IR,** the `LinearQubits` trait verifies that each qubit value is consumed at most once *on every path*.
  The two branches of an `if` are two paths, so each may consume the qubit. A loop may not consume a qubit from
  outside it, because it would do so once per iteration. This makes linearity correct in the presence of classical
  control flow, which the use count that MLIR's `hasOneUse` gives cannot do.

**Where it runs.** A quantum program runs only as native code, on the runtime's simulator. The interpreter stops
with "quantum operations run only in native code: compile this program with nova-compile". The differential test
counts such programs as native-only rather than as failures. A circuit with classical control flow is decomposed and
optimized; the scheduler (below) places only straight-line circuits for now.

## Qubits as SSA values

A qubit is a value of type `!qnova.qubit`, and each gate consumes the qubits it acts on and gives new values for them:

```mlir
%a = qnova.alloc
%b = qnova.alloc
%a1 = qnova.h %a
%a2, %b1 = qnova.cx %a1, %b
qnova.probs %a2, %b1
```

This is the design of QSSA (Peduri, Bhat, Grosser, CC 2022) and of Catalyst's quantum dialect. The alternative is
the reference style of QIR, where `h(q)` modifies a qubit in place. The value style is what makes the rest simple:

- **Def-use chains are the circuit's wires.** The gate before a gate on the same qubit is the op that defines its
  operand, with no analysis needed. Every peephole rewrite below is a few lines because of this.
- **The dependence graph is SSA.** Two gates can run in parallel exactly when neither is reachable from the other
  through def-use, which is what the scheduler reads.
- **Linearity enforces no-cloning.** A quantum state cannot be copied, so each qubit value may be used at most once.
  A C++ op trait, `LinearQubits`, makes that part of the verifier: `%b = qnova.x %a` then `%c = qnova.z %a` is
  rejected with "a quantum state cannot be copied (no-cloning)".

Gates are `Pure`, which is sound for two reasons:

- **CSE never merges two gates,** because merging would need one value used twice.
- **A gate on a qubit nothing uses again is dead.** By the no-signaling theorem, an operation on a qubit no one
  measures cannot change any other qubit's statistics.

`alloc` allocates (`MemAlloc`), so two allocations are never merged. `measure` and `probs` have effects (they draw
randomness and print), so they are never removed or reordered.

## The native gate set

A neutral-atom machine (atoms held in optical tweezers) has three native operations:

| Native op | Physical operation | Matrix |
|---|---|---|
| `qnova.r %q, θ, φ` | Raman laser pulse | rotation by θ about cos φ·X + sin φ·Y |
| `qnova.rz %q, θ` | phase shift | diag(e^(−iθ/2), e^(iθ/2)) |
| `qnova.cz %a, %b` | Rydberg blockade between two atoms within the interaction radius | diag(1, 1, 1, −1) |

`--qnova-decompose` rewrites each logical gate into native ones. Each identity holds up to a global phase, which no
measurement can observe:

| Logical | Native |
|---|---|
| `x`, `y` | `r(π, 0)`, `r(π, π/2)` |
| `rx(t)`, `ry(t)` | `r(t, 0)`, `r(t, π/2)` |
| `z`, `s`, `t` | `rz(π)`, `rz(π/2)`, `rz(π/4)` |
| `h` | `rz(π)`, then `r(π/2, π/2)` |
| `cx(a, b)` | `h(b)`, `cz(a, b)`, `h(b)` |
| `swap(a, b)` | `cx(a, b)`, `cx(b, a)`, `cx(a, b)` |
| `cp(a, b, λ)` | `rz(b, λ/2)`, `cx(a, b)`, `rz(b, −λ/2)`, `cx(a, b)`, `rz(a, λ/2)` |

Decomposition is a greedy pattern rewrite, so the `cx` that `swap` produces are rewritten in turn. It deliberately does
not optimize: canonicalize does that, before and after, so each step's effect shows up separately in the stats.

## Canonicalization, and the physics behind each rewrite

| Rewrite | Why it is correct |
|---|---|
| `h h`, `x x`, `y y`, `z z`, `cx cx` (same wires), `cz cz` and `swap swap` (either order) → nothing | each is its own inverse |
| `rz(a) rz(b)` → `rz(a + b)`; `r(a, φ) r(b, φ)` → `r(a + b, φ)`; `cp(a) cp(b)` → `cp(a + b)` | rotations about one axis compose by adding angles |
| `rz(2πk)`, `r(2πk, φ)`, `cp(2πk)` → nothing | a full turn is the identity up to global phase |
| `rz` after a `cz` → `rz` before it | diagonal matrices commute; moving rz earlier lets it merge with an rz there |
| `rz`, `cz` or `cp` on a freshly allocated qubit → nothing | a diagonal gate only multiplies \|0⟩ by a phase |
| `rz`, `cz` or `cp` just before `probs` → nothing | a diagonal gate changes no basis-state magnitude |

Canonicalization runs on the logical circuit first, where cancellations are easiest to see. A QFT followed by its
inverse cancels down to the gates that prepare its input. It runs again after decomposition, on the native gates
that decomposition leaves.

A rewrite that moves an op (rz through cz) checks that its angle operand is defined before the new position, so the
result is still valid SSA.

## The machine model: placement, routing by moving atoms, scheduling

`--qnova-schedule` works on the native circuit:

1. **Placement.** Atoms go to the black squares of a checkerboard on a grid with about twice as many sites as qubits.
   Real arrays have spare sites, and the checkerboard gives each atom four diagonal neighbours it can interact with
   immediately, plus four free sites beside it.
2. **Routing.** A `cz` needs its two atoms within the interaction radius (neighbouring sites). A superconducting chip
   would route with chains of SWAP gates, each costing three two-qubit gates. A neutral-atom machine can instead
   *carry* an atom to another site, its state intact. So routing inserts a `qnova.move {to = [x, y]}` before each `cz`
   whose atoms are not neighbours, taking the free site next to the partner that is nearest. If that atom is boxed
   in, the partner moves instead; if both are, the first parks at the nearest site with room and the second joins it.
3. **Scheduling.** The machine runs layers, each one kind of operation for many atoms at once: a Raman layer, a
   Rydberg layer, a move layer, a measurement layer. Each op goes to the earliest layer of its kind after everything
   it depends on (SSA def-use). A Rydberg pulse drives every pair within the blockade radius, so two `cz` in one layer
   must be at least two sites apart, or the second waits for a later layer.

Each op gets `{layer = n}`, and the function gets `qnova.depth`, `qnova.moves` and `qnova.move_distance`. The model is
deliberately simple: one move per `cz` with no moving back, and no limit on parallel moves. This pass is where a
richer machine description would plug in.

## The hardware layer: timed instructions, the `atom` dialect

Below the native gates is a third dialect, `atom`: the machine's instructions, with their times. `--qnova-to-atom`
writes a scheduled circuit as instructions:

```mlir
atom.load [0, 0]                                                  // an atom at a site, in |0>
atom.raman [0, 0] theta 1.5707963267948966 phi 1.5707963267948966 at 0 for 500   // a local Raman pulse
atom.shift [1, 1] theta 3.1415926535897931 at 500 for 500         // a phase shift (rz)
atom.move [0, 2] to [0, 1] at 1000 for 25000                      // a tweezer move
atom.rydberg [0, 0, 1, 0, 1, 1, 2, 1] at 26000 for 400            // ONE global pulse: a cz on every listed pair
atom.readout [0, 0, 1, 0, 1, 1, 0, 1] at 26400 for 1000000
```

How it differs from the layer above:
- **No qubit values.** An instruction names a site, as the hardware does. The pass follows each SSA qubit value to its
  atom's site: its alloc's site, then each move's.
- **Times instead of order.** Every instruction carries `start` and `duration` in nanoseconds. Layers run one after
  another, each as long as its longest instruction.
- **The Rydberg pulse is global.** A layer's `cz` gates become *one* `atom.rydberg` naming all their pairs, which is
  how the hardware drives them. The verifier checks that each pair is neighbouring atoms.

The timing model's parameters are options of the pass. The defaults are illustrative orders of magnitude, not one
machine's numbers:

| Parameter | Default |
|---|---|
| `raman-pi-ns` | 1,000 ns per π of rotation |
| `shift-pi-ns` | 500 ns per π |
| `rydberg-ns` | 400 ns |
| `move-site-ns`, `move-overhead-ns` | 10,000 ns per site of travel, plus 15,000 ns to pick up and drop |
| `measure-ns` | 1,000,000 ns |

`--atom-timing` prints a program's run time by kind of instruction, and `nova-compile --hardware` compiles a circuit
all the way down to these instructions.

**The simulator runs the hardware program too, by site.** It keeps which qubit's atom sits at each site: a load puts
one there, a move carries it, and each pulse acts on the atom at its site. So the lowest layer is checked the same
way as the others: the hardware program must give the logical circuit's probabilities.

**That check found a scheduler bug.** Routing chooses each move's destination in program order, when the site is
free because another atom has already left it. The scheduler then packs ops into the earliest layer their SSA
dependences allow. Two moves of different atoms have no SSA dependence on each other, so the move *into* a site
could land in an earlier layer than the move *out* of it, and on the hardware the two atoms would collide. The gate
level couldn't see this, because qubit values carry no sites. Running the timed instructions by site stopped with
"an atom moved onto another". The fix is a resource dependence on sites: a move to a site now waits for the layer
after the move that vacated it.

## Running circuits, and checking the compiler

`--convert-nova-to-runtime`, the same pass that lowers Nova's tensors, lowers `qnova` too:

- a qubit becomes an `i32`, its index in the simulator;
- each gate becomes a call that updates the state;
- a move becomes nothing, since moving an atom doesn't change its state.

`runtime/nova_quantum.c` is a statevector simulator for up to 24 qubits. It runs the logical gates and the native ones
alike, as well as the hardware instructions by site, which makes **differential testing** possible.
`utils/qbench.py` runs each benchmark circuit three ways: as written, compiled to native gates, and compiled to
hardware instructions. It requires the second and third to give the first's probability distribution, to four
places. CI runs it on every push.

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

"Saved" is the share of native gates canonicalization removes. The QFT round trip collapses to the `x` gates that
prepare its input, because the circuit is a unitary followed by its inverse. Gate time and move time are the hardware
program's, without the final readout. **Moves take 3–8× as long as all the gates together.** On this machine model,
routing rather than gate count dominates a circuit's run time, which is what a move-aware router would optimize next.
Reproduce the table with `python3 compiler/mlir/utils/qbench.py --build <build dir>`, or pass `--width 4` to see the
effect of a tighter grid.

## On the GPU

The simulator also runs on a GPU ([`compiler/mlir/runtime/gpu`](../compiler/mlir/runtime/gpu/README.md)): the same
operations in CUDA, one coalesced kernel per gate, because a statevector simulator is memory-bound (a one-qubit
gate reads and writes every amplitude for a few flops each). Its C functions are the runtime's, so a compiled
program runs its circuits on the GPU by linking with it (`nova-compile --gpu`). `qsim_bench` runs the same circuits
on the CPU (one thread and every core) and the GPU, compares every amplitude, and reports the speedup and the GPU's
effective memory bandwidth against its peak; `run_sol.sh` runs it on an A100.

## Tests

- `test/QNova/ops.mlir`: every op round-trips through the printer and parser.
- `test/QNova/invalid.mlir`: the no-cloning check and the `probs` limit.
- `test/QNova/canonicalize.mlir`: each rewrite above.
- `test/QNova/decompose.mlir`: the decompositions.
- `test/QNova/schedule.mlir`: placement, a move, layers and the stats line, all hand-checked.
- `test/Run/quantum.mlir`: a circuit compiled both ways, run, with the probabilities checked.
- `test/Run/measure.mlir`: a Bell pair's measured bits agree.
- `test/Lower/quantum.nova`: Nova's `qubit` and `qpu` gates lowered to qnova, with a qubit carried through a loop and
  an if.
- `test/Lower/quantum_errors.nova`: each no-cloning error of the check pass.
- `test/Run/teleport.nova`: teleportation written in Nova, run with three measurement seeds and as written; the
  teleported state is the same every time.
- `test/QNova/ops.mlir` and `invalid.mlir`: an if's two branches may each consume a qubit, but a loop may not
  consume one from outside it.
- `test/Atom/qnova-to-atom.mlir`: a scheduled circuit as timed instructions, every start time checked by hand, and
  a timing option changing the result.
- `test/Atom/ops.mlir`, `invalid.mlir`, `unscheduled.mlir`: the instructions round-trip; a Rydberg pair must be
  neighbours; a circuit must be scheduled first.
- `test/Run/hardware.mlir`: a circuit compiled to hardware instructions, run by site, gives the logical probabilities.

## What would come next

- **Qubits passed to and returned from functions,** as in-out parameters lowered to extra results, and qubit arrays
  (`qubit[8] r`) for circuits whose size is a parameter.
- **A richer machine model:** move parallelism and collision constraints for the tweezer arrays, moving atoms back
  after use, and reusing atoms across a long circuit.
- **Commutation-aware optimization beyond diagonal gates,** such as Pauli frame tracking and two-qubit block
  resynthesis.
- **A move-aware router.** Moves dominate the run time (above), so the next optimization is routing that moves atoms
  less: placement from the circuit's interaction graph, parallel moves in one tweezer sweep, and moving atoms back.
- **Pulse shapes under the instructions,** and a machine description file for the timing model instead of options.
