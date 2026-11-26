// The GPU statevector simulator (statevector.cuh). A statevector simulator is memory-bound: a one-qubit gate on n
// qubits reads and writes all 2^n amplitudes (16 bytes each) and does a handful of flops per amplitude, so its speed
// is the GPU's memory bandwidth. Each kernel is therefore one pass over the state, coalesced: consecutive threads
// touch consecutive amplitudes (for all but the lowest qubits, whose pairs are next to each other anyway).
//
//   apply        one thread per pair of amplitudes that differ only in bit q: index k with a 0 inserted at bit q
//   phase_both   one thread per amplitude; those with both bits set are multiplied by the phase (cz: -1)
//   cx, swap     one thread per pair to exchange
//   measure      a reduction (each block sums its part in shared memory, one atomicAdd per block), then a collapse
//   probabilities  a histogram of |amplitude|^2 into the 2^n bins of the qubits read, with atomicAdd
#include "statevector.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace
{
constexpr int Threads = 256;
constexpr int32_t MaxQubits = 32;

cuDoubleComplex* d_state = nullptr;
int32_t n_qubits = 0;
double* d_scratch = nullptr;  // the reductions' results: 4096 doubles

void check(cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return;
    fprintf(stderr, "Runtime error: CUDA %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
}

unsigned blocks(size_t threads) { return unsigned((threads + Threads - 1) / Threads); }

__device__ size_t insert_zero(size_t k, size_t bit) { return ((k & ~(bit - 1)) << 1) | (k & (bit - 1)); }

__global__ void k_apply(cuDoubleComplex* s, size_t half, size_t bit, nova_gpu::Mat2 m)
{
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= half) return;
    size_t i = insert_zero(k, bit), j = i | bit;
    cuDoubleComplex x0 = s[i], x1 = s[j];
    s[i] = cuCadd(cuCmul(m.a, x0), cuCmul(m.b, x1));
    s[j] = cuCadd(cuCmul(m.c, x0), cuCmul(m.d, x1));
}

__global__ void k_phase_both(cuDoubleComplex* s, size_t size, size_t both, cuDoubleComplex phase)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size && (i & both) == both) s[i] = cuCmul(s[i], phase);
}

// The pairs (i with bit `from` set and bit `to` clear, j = i with them exchanged) swap their amplitudes. cx(c, t):
// from = c, the pair i, i | t (both with c set); swap(a, b): i with a set and b clear, j with a clear and b set.
__global__ void k_exchange(cuDoubleComplex* s, size_t size, size_t set, size_t clear, size_t flip)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size || !(i & set) || (i & clear)) return;
    size_t j = i ^ flip;
    cuDoubleComplex x = s[i];
    s[i] = s[j];
    s[j] = x;
}

__global__ void k_p1(const cuDoubleComplex* s, size_t size, size_t bit, double* out)
{
    __shared__ double part[Threads];
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double v = 0;
    if (i < size && (i & bit)) v = cuCreal(s[i]) * cuCreal(s[i]) + cuCimag(s[i]) * cuCimag(s[i]);
    part[threadIdx.x] = v;
    __syncthreads();
    for (int stride = Threads / 2; stride > 0; stride /= 2)
    {
        if (int(threadIdx.x) < stride) part[threadIdx.x] += part[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, part[0]);
}

__global__ void k_collapse(cuDoubleComplex* s, size_t size, size_t bit, int outcome, double norm)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    s[i] = (((i & bit) != 0) == (outcome != 0)) ? make_cuDoubleComplex(cuCreal(s[i]) / norm, cuCimag(s[i]) / norm)
                                                : make_cuDoubleComplex(0, 0);
}

struct Qubits
{
    int32_t n;
    int32_t q[12];
};

__global__ void k_histogram(const cuDoubleComplex* s, size_t size, Qubits qs, double* bins)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    size_t k = 0;
    for (int32_t j = 0; j < qs.n; j++) k = (k << 1) | ((i >> qs.q[j]) & 1);
    double p = cuCreal(s[i]) * cuCreal(s[i]) + cuCimag(s[i]) * cuCimag(s[i]);
    if (p != 0) atomicAdd(&bins[k], p);
}

__global__ void k_set_one(cuDoubleComplex* s) { s[0] = make_cuDoubleComplex(1, 0); }
}  // namespace

namespace nova_gpu
{
int32_t alloc()
{
    if (n_qubits == MaxQubits) fprintf(stderr, "Runtime error: more qubits than the GPU simulator holds\n"), exit(1);
    size_t size = size_t(1) << n_qubits;
    cuDoubleComplex* grown = nullptr;
    check(cudaMalloc(&grown, 2 * size * sizeof(cuDoubleComplex)), "allocating the state");
    if (n_qubits == 0)
    {
        check(cudaMemset(grown, 0, 2 * sizeof(cuDoubleComplex)), "clearing the state");
        k_set_one<<<1, 1>>>(grown);
        check(cudaMalloc(&d_scratch, 4096 * sizeof(double)), "allocating scratch memory");
    }
    else
    {
        check(cudaMemcpy(grown, d_state, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice), "growing");
        check(cudaMemset(grown + size, 0, size * sizeof(cuDoubleComplex)), "clearing the new half");
        check(cudaFree(d_state), "freeing the old state");
    }
    d_state = grown;
    return n_qubits++;
}

void reset()
{
    if (d_state) cudaFree(d_state);
    if (d_scratch) cudaFree(d_scratch);
    d_state = nullptr;
    d_scratch = nullptr;
    n_qubits = 0;
}

int32_t qubits() { return n_qubits; }

void apply(int32_t q, Mat2 m)
{
    size_t half = size_t(1) << (n_qubits - 1);
    k_apply<<<blocks(half), Threads>>>(d_state, half, size_t(1) << q, m);
}

void phase_both(int32_t a, int32_t b, cuDoubleComplex phase)
{
    size_t size = size_t(1) << n_qubits;
    k_phase_both<<<blocks(size), Threads>>>(d_state, size, (size_t(1) << a) | (size_t(1) << b), phase);
}

void cx(int32_t control, int32_t target)
{
    size_t size = size_t(1) << n_qubits, t = size_t(1) << target;
    k_exchange<<<blocks(size), Threads>>>(d_state, size, size_t(1) << control, t, t);
}

void swap(int32_t a, int32_t b)
{
    size_t size = size_t(1) << n_qubits, ba = size_t(1) << a, bb = size_t(1) << b;
    k_exchange<<<blocks(size), Threads>>>(d_state, size, ba, bb, ba | bb);
}

int32_t measure(int32_t q, double random)
{
    size_t size = size_t(1) << n_qubits, bit = size_t(1) << q;
    double p1 = 0;
    check(cudaMemset(d_scratch, 0, sizeof(double)), "clearing a sum");
    k_p1<<<blocks(size), Threads>>>(d_state, size, bit, d_scratch);
    check(cudaMemcpy(&p1, d_scratch, sizeof(double), cudaMemcpyDeviceToHost), "reading a sum");
    int outcome = random < p1;
    k_collapse<<<blocks(size), Threads>>>(d_state, size, bit, outcome, std::sqrt(outcome ? p1 : 1 - p1));
    return outcome;
}

void probabilities(int32_t n, const int32_t* q, double* probs)
{
    size_t size = size_t(1) << n_qubits, bins = size_t(1) << n;
    Qubits qs{n, {}};
    for (int32_t j = 0; j < n; j++) qs.q[j] = q[j];
    check(cudaMemset(d_scratch, 0, bins * sizeof(double)), "clearing the bins");
    k_histogram<<<blocks(size), Threads>>>(d_state, size, qs, d_scratch);
    check(cudaMemcpy(probs, d_scratch, bins * sizeof(double), cudaMemcpyDeviceToHost), "reading the bins");
}

void state(cuDoubleComplex* host)
{
    check(cudaMemcpy(host, d_state, (size_t(1) << n_qubits) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost),
          "reading the state");
}

void synchronize() { check(cudaDeviceSynchronize(), "running the kernels"); }

Mat2 h()
{
    cuDoubleComplex s = make_cuDoubleComplex(M_SQRT1_2, 0), m = make_cuDoubleComplex(-M_SQRT1_2, 0);
    return {s, s, s, m};
}

Mat2 r(double theta, double phi)  // cos(t/2) on the diagonal; -i sin(t/2) e^(-+ i phi) off it
{
    double c = std::cos(theta / 2), s = std::sin(theta / 2);
    cuDoubleComplex off_minus = make_cuDoubleComplex(-s * std::sin(phi), -s * std::cos(phi));  // -i s e^(-i phi)
    cuDoubleComplex off_plus = make_cuDoubleComplex(s * std::sin(phi), -s * std::cos(phi));    // -i s e^(+i phi)
    return {make_cuDoubleComplex(c, 0), off_minus, off_plus, make_cuDoubleComplex(c, 0)};
}

Mat2 rz(double theta)
{
    return {make_cuDoubleComplex(std::cos(theta / 2), -std::sin(theta / 2)), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(std::cos(theta / 2), std::sin(theta / 2))};
}

Mat2 rx(double theta) { return r(theta, 0); }
Mat2 ry(double theta) { return r(theta, M_PI / 2); }
}  // namespace nova_gpu
