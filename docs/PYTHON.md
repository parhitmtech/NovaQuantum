# Nova and Python: the same syntax, different semantics

Nova is written like Python: indentation, `def`, `class`, `if` / `elif` / `else`, `while`, `for`. It does not
*behave* like Python. It is statically typed and compiled, and where Python's semantics and C's differ, Nova mostly
follows C. Some of these differences were chosen, and some are limitations. This page lists them. Every example was
run through Nova's interpreter, and the Python column is CPython 3.12.

## Values and arithmetic

| | Nova | Python | Why Nova differs |
|---|---|---|---|
| `-7 / 2` on ints | `-3`: truncates toward zero | `-7 // 2` is `-4`: floors | Nova's `/` on ints is C's division, and `int` is a machine type. It compiles to one instruction (`arith.divsi`, LLVM's `sdiv`). |
| `-7 % 2` | `-1`: takes the dividend's sign | `1`: takes the divisor's sign | It follows from truncating division: `a % b` is `a - (a / b) * b`. |
| `2147483647 + 1` | `-2147483648`: `int` is 32 bits and wraps | `2147483648`: ints are unbounded | A fixed-width `int` fits a register and compiles to `i32`. Unbounded ints need heap allocation and checks on every operation ([BENCHMARKS.md](BENCHMARKS.md#issues-found), issue 2). |
| `7 / 0` | Runtime error at the line: "integer division by zero" | `ZeroDivisionError` | The same idea. Natively, `nova.check_divisor` keeps the error, where LLVM's `sdiv` would be undefined ([DESIGN.md](DESIGN.md), C10). |
| `INT_MIN / -1` | `INT_MIN`: wraps | an int one larger | Matches what the hardware gives. The native path divides by 1 in that case so as never to execute LLVM's undefined `sdiv`. |
| `1 + 2.5` | Compile error: no implicit mixing of `int` and `float` | `3.5` | Conversions are explicit, so the type of every expression is known at compile time. |
| `true + true` | Compile error | `2` (`bool` is a subclass of `int`) | `bool` is its own type, `i1`. |

## Names, types and scope

| | Nova | Python |
|---|---|---|
| When a type error is found | Before the program runs: `f("a")` for `def f(int n)` is "argument 1 of 'f' expects int but got string" | When the line runs: `TypeError` |
| Scope of a variable declared in a block | The block: an `int y` declared inside an `if` is undeclared after it | The whole function |
| Declaring a variable | With its type, `int x = 1`, once per scope | By assigning to it |
| A function with untyped parameters | Generic: compiled once per argument-type combination, each copy type-checked (`def g(x)` with `g(2)` and `g(1.5)` makes `g<int>` and `g<float>`) | Duck-typed: one function object, its operations dispatched at run time |
| Returning several values | Native: `int q, int r = divmod(17, 5)` | A tuple, unpacked |

## Objects and tensors

| | Nova | Python |
|---|---|---|
| `m = l` for an object | Both names refer to one object, so a function that changes its parameter's fields changes the caller's object | The same: names are references |
| `u = t` for a tensor, then `u[0, 0] = 5.0` | `t[0, 0]` is `5.0` as well: tensors are references | The same for a NumPy array or a torch tensor |
| When an object is freed | Reference counting, when nothing refers to it (cycles leak; `weak` fields are planned) | Reference counting, plus a cycle collector |
| When a tensor is freed | At its last use, found by a liveness pass at compile time ([DESIGN.md](DESIGN.md), 8) | Reference counting |
| Method calls | Virtual dispatch on the object's own class, like a vtable; which methods exist, and an override's signature, are checked at compile time | Attribute lookup at run time (the MRO, descriptors) |

## How each one runs a program

**CPython** compiles source to bytecode for a stack-based virtual machine. Its evaluation loop dispatches one
opcode at a time over boxed objects, and every `a + b` looks up the operand types' `__add__` at run time.

**Nova's interpreter** works differently:
- **The check pass resolves every name and type before the program runs.** Lowering then writes a typed, linear
  IR: instructions over numbered slots in separate memory banks for ints, floats, doubles and strings, so
  `a + b` on ints is one instruction on two int slots, with no type dispatch at run time.
- **Optional optimization passes** run on that IR: constant folding, copy propagation, dead-code elimination, LICM
  and inlining.
- **Pure recursive functions are memoized automatically.** That is why naive `fib(35)` runs fast in the
  interpreter ([BENCHMARKS.md](BENCHMARKS.md), issue 3).
- **An instruction budget** stops a runaway loop with an error instead of hanging.

**Nova's native path** compiles the same checked program ahead of time through MLIR and LLVM to machine code:
2–215× faster than the interpreter, and on par with C ([BENCHMARKS.md](BENCHMARKS.md#native-code-through-mlir-and-llvm)).
The interpreter stays the reference: in CI, every natively compiled program must produce the same stdout, stderr and
exit status as the interpreter.
