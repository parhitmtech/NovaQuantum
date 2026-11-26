# The GPU statevector simulator

The quantum runtime's simulator in CUDA. It runs the same operations as the CPU simulator
([`../nova_quantum.c`](../nova_quantum.c)), so the two can be compared amplitude by amplitude, and a compiled Nova
program or `qnova` circuit can run on either.

| File | What it is |
|---|---|
| `statevector.cu`, `.cuh` | The simulator: the state's 2^n complex doubles in device memory, one kernel per gate |
| `nova_quantum_gpu.cu` | The runtime's C functions (`nova_q_*`) on the GPU: the drop-in for compiled programs |
| `qsim_bench.cu` | The benchmark: the same circuits on the CPU (1 thread, all cores) and the GPU, checked and timed |
| `dropin_check.c` | The calls a compiled circuit makes, by hand, run with each runtime: their outputs must match |
| `run_sol.sh` | All of it as a SOL batch job |

## Why the kernels look the way they do

A statevector simulator is **memory-bound**. A one-qubit gate on n qubits reads and writes all 2^n amplitudes (16
bytes each) and does about 14 flops per pair, an arithmetic intensity well under 1 flop per byte. So the speed of a
gate is the GPU's memory bandwidth, and every kernel is a single coalesced pass over the state:

| Kernel | Threads | Work |
|---|---|---|
| `k_apply` | one per pair of amplitudes that differ in bit q (index k with a 0 inserted at bit q) | the 2x2 matrix |
| `k_phase_both` | one per amplitude | multiply those with both bits set by a phase (`cz`: −1, `cp`: e^(iλ)) |
| `k_exchange` | one per amplitude; those with the "set" bit and not the "clear" bit act | exchange with the partner (`cx`, `swap`) |
| `k_p1`, `k_collapse` | one per amplitude | P(1) by a block reduction in shared memory and one `atomicAdd` per block; then the collapse |
| `k_histogram` | one per amplitude | add \|amplitude\|² to the bin of the measured qubits' bits, with `atomicAdd` |

The benchmark reports each run's effective bandwidth (the bytes each gate's kernel must move, over the time) next
to the GPU's peak, which is the right yardstick for a memory-bound code.

## Running it on SOL

```sh
cd compiler/mlir/runtime/gpu
sbatch run_sol.sh                          # sizes 16 to 28; CPU and GPU compared up to 24 qubits
sbatch run_sol.sh --max 30 --cpu-max 26    # up to 30 qubits (a 16 GiB state)
cat qsim-<job id>.out
```

The job builds everything (`make`, with `ARCH=sm_80` for an A100), runs `make dropin-check` (the same circuit
through the CPU runtime and the GPU runtime, outputs compared), then `qsim_bench`. It needs only a CUDA toolkit, not
LLVM or MLIR. Without a GPU, `make` still compiles everything, and the CPU side of `make dropin-check` runs.

To run a compiled Nova program on the GPU instead, on a machine with both the MLIR build and CUDA:
`make` here, then `nova-compile prog.nova --gpu -o prog`.
