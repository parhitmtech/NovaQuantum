// The quantum runtime's statevector functions on the GPU: the same C functions as runtime/nova_quantum.c (nova_q_*),
// so a compiled circuit runs on either by linking one or the other. libnova_runtime_gpu.a is this file, the GPU
// simulator (statevector.cu), and nova_quantum.c built with NOVA_QUANTUM_GPU, which keeps the printing, the random
// numbers and the hardware instructions by site (they call these functions).
#include <cmath>
#include <cstdint>

#include "statevector.cuh"

extern "C"
{
void nova_q_print_probs(int32_t n, const double* probs);  // runtime/nova_quantum.c
double nova_q_random(void);

int32_t nova_q_alloc(void) { return nova_gpu::alloc(); }
void nova_q_reset(void) { nova_gpu::reset(); }

void nova_q_h(int32_t q) { nova_gpu::apply(q, nova_gpu::h()); }
// The CPU simulator's exact matrices, so the two agree amplitude by amplitude, not only up to a global phase
static cuDoubleComplex c(double re, double im = 0) { return make_cuDoubleComplex(re, im); }
void nova_q_x(int32_t q) { nova_gpu::apply(q, {c(0), c(1), c(1), c(0)}); }
void nova_q_y(int32_t q) { nova_gpu::apply(q, {c(0), c(0, -1), c(0, 1), c(0)}); }
void nova_q_z(int32_t q) { nova_gpu::apply(q, {c(1), c(0), c(0), c(-1)}); }
void nova_q_s(int32_t q) { nova_gpu::apply(q, {c(1), c(0), c(0), c(0, 1)}); }
void nova_q_t(int32_t q) { nova_gpu::apply(q, {c(1), c(0), c(0), c(M_SQRT1_2, M_SQRT1_2)}); }
void nova_q_rx(int32_t q, double t) { nova_gpu::apply(q, nova_gpu::rx(t)); }
void nova_q_ry(int32_t q, double t) { nova_gpu::apply(q, nova_gpu::ry(t)); }
void nova_q_rz(int32_t q, double t) { nova_gpu::apply(q, nova_gpu::rz(t)); }
void nova_q_r(int32_t q, double t, double phi) { nova_gpu::apply(q, nova_gpu::r(t, phi)); }

void nova_q_cz(int32_t a, int32_t b) { nova_gpu::phase_both(a, b, make_cuDoubleComplex(-1, 0)); }
void nova_q_cp(int32_t a, int32_t b, double l)
{
    nova_gpu::phase_both(a, b, make_cuDoubleComplex(std::cos(l), std::sin(l)));
}
void nova_q_cx(int32_t c, int32_t t) { nova_gpu::cx(c, t); }
void nova_q_swap(int32_t a, int32_t b) { nova_gpu::swap(a, b); }

int32_t nova_q_measure(int32_t q) { return nova_gpu::measure(q, nova_q_random()); }

void nova_q_probabilities(int32_t n, const int32_t* q, double* probs) { nova_gpu::probabilities(n, q, probs); }

void nova_q_probs(int32_t n, int32_t q0, int32_t q1, int32_t q2, int32_t q3, int32_t q4, int32_t q5, int32_t q6,
                  int32_t q7, int32_t q8, int32_t q9, int32_t q10, int32_t q11)
{
    int32_t q[12] = {q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11};
    static double probs[4096];
    nova_gpu::probabilities(n, q, probs);
    nova_q_print_probs(n, probs);
}
}
