// A statevector simulator on the GPU: the state of n qubits, 2^n complex doubles, lives in device memory, and each
// gate is a kernel over it. The same operations as the CPU simulator (runtime/nova_quantum.c), so the two can be
// compared amplitude by amplitude (qsim_bench.cu) and either can run a compiled circuit (nova_quantum_gpu.cu).
#ifndef NOVA_GPU_STATEVECTOR_CUH
#define NOVA_GPU_STATEVECTOR_CUH

#include <cstddef>
#include <cstdint>

#include <cuComplex.h>

namespace nova_gpu
{
// A 2x2 complex matrix, row by row: [[a, b], [c, d]]
struct Mat2
{
    cuDoubleComplex a, b, c, d;
};

int32_t alloc();                                    // a new qubit, |0>: the state doubles
void reset();                                       // no qubits
int32_t qubits();
void apply(int32_t q, Mat2 m);                      // a one-qubit gate
void phase_both(int32_t a, int32_t b, cuDoubleComplex phase);  // states with both bits set times phase (cz, cp)
void cx(int32_t control, int32_t target);
void swap(int32_t a, int32_t b);
int32_t measure(int32_t q, double random);          // outcome 1 if random < P(1); the state collapses to it
void probabilities(int32_t n, const int32_t* q, double* probs);  // 2^n probabilities of n qubits, first bit leftmost
void state(cuDoubleComplex* host);                  // the whole state, copied to the host (for checking)
void synchronize();                                 // wait for the kernels launched so far (for timing)

// The gates as matrices (the CPU simulator's, in runtime/nova_quantum.c)
Mat2 h();
Mat2 r(double theta, double phi);
Mat2 rz(double theta);
Mat2 rx(double theta);
Mat2 ry(double theta);
}  // namespace nova_gpu

#endif  // NOVA_GPU_STATEVECTOR_CUH
