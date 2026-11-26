// The statevector simulator on the GPU against the CPU: the same circuits on both, timed, and checked amplitude by
// amplitude.
//
//   ./qsim_bench [--min 16] [--max 28] [--cpu-max 24] [--depth 10]
//
// Two circuits per size n:
//   layers  `depth` layers, each an ry and an rz on every qubit, then a cz brickwork (q, q+1), offset by layer: the
//           shape of a variational circuit
//   qft     the quantum Fourier transform: h and a cascade of controlled phases, n (n + 1) / 2 gates
// Each runs on the CPU simulator (runtime/nova_quantum.c) with one thread and with every core (OpenMP), and on the
// GPU simulator (statevector.cu), with the state allocated first, so the time is the gates' alone. Up to --cpu-max
// qubits the GPU's final state is compared with the CPU's, every amplitude; above it only the GPU runs.
//
// A statevector simulator is memory-bound, so the table also gives the GPU's effective bandwidth: the bytes each
// gate's kernel must move (a one-qubit gate reads and writes every amplitude, 32 bytes each; a cz or cp a quarter of
// them; a cx or swap half), over the time, next to the GPU's peak.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "statevector.cuh"

extern "C"
{
int32_t nova_q_alloc(void);
void nova_q_reset(void);
void nova_q_ry(int32_t q, double t);
void nova_q_rz(int32_t q, double t);
void nova_q_h(int32_t q);
void nova_q_cz(int32_t a, int32_t b);
void nova_q_cp(int32_t a, int32_t b, double l);
const void* nova_q_amplitudes(void);  // double complex*, as C lays it out: two doubles, as std::complex<double>
}

namespace
{
enum Kind { Ry, Rz, H, Cz, Cp };

struct Gate
{
    Kind kind;
    int32_t a, b;
    double angle;
};

struct Circuit
{
    std::string name;
    int32_t n;
    std::vector<Gate> gates;
};

Circuit layers(int32_t n, int32_t depth)
{
    Circuit c{"layers", n, {}};
    uint64_t seed = 12345;
    auto angle = [&] {  // reproducible angles in [0, 2 pi)
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return double(seed >> 11) / 9007199254740992.0 * 2 * M_PI;
    };
    for (int32_t l = 0; l < depth; l++)
    {
        for (int32_t q = 0; q < n; q++)
        {
            c.gates.push_back({Ry, q, 0, angle()});
            c.gates.push_back({Rz, q, 0, angle()});
        }
        for (int32_t q = l % 2; q + 1 < n; q += 2) c.gates.push_back({Cz, q, q + 1, 0});
    }
    return c;
}

Circuit qft(int32_t n)
{
    Circuit c{"qft", n, {}};
    for (int32_t i = 0; i < n; i++)
    {
        c.gates.push_back({H, i, 0, 0});
        for (int32_t j = i + 1; j < n; j++) c.gates.push_back({Cp, j, i, M_PI / double(1ull << (j - i))});
    }
    return c;
}

double seconds_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// The CPU simulator, with this many threads: the gates' time, the state left allocated for the comparison
double run_cpu(const Circuit& c, int threads)
{
    omp_set_num_threads(threads);
    nova_q_reset();
    for (int32_t q = 0; q < c.n; q++) nova_q_alloc();
    auto start = std::chrono::steady_clock::now();
    for (const Gate& g : c.gates)
    {
        switch (g.kind)
        {
            case Ry: nova_q_ry(g.a, g.angle); break;
            case Rz: nova_q_rz(g.a, g.angle); break;
            case H:  nova_q_h(g.a); break;
            case Cz: nova_q_cz(g.a, g.b); break;
            case Cp: nova_q_cp(g.a, g.b, g.angle); break;
        }
    }
    return seconds_since(start);
}

double run_gpu(const Circuit& c)
{
    nova_gpu::reset();
    for (int32_t q = 0; q < c.n; q++) nova_gpu::alloc();
    nova_gpu::synchronize();
    auto start = std::chrono::steady_clock::now();
    for (const Gate& g : c.gates)
    {
        switch (g.kind)
        {
            case Ry: nova_gpu::apply(g.a, nova_gpu::ry(g.angle)); break;
            case Rz: nova_gpu::apply(g.a, nova_gpu::rz(g.angle)); break;
            case H:  nova_gpu::apply(g.a, nova_gpu::h()); break;
            case Cz: nova_gpu::phase_both(g.a, g.b, make_cuDoubleComplex(-1, 0)); break;
            case Cp: nova_gpu::phase_both(g.a, g.b, make_cuDoubleComplex(std::cos(g.angle), std::sin(g.angle))); break;
        }
    }
    nova_gpu::synchronize();
    return seconds_since(start);
}

// The bytes the gates' kernels move (see the top of the file)
double traffic(const Circuit& c)
{
    double size = std::ldexp(1.0, c.n), bytes = 0;
    for (const Gate& g : c.gates) bytes += (g.kind == Cz || g.kind == Cp) ? 8 * size : 32 * size;
    return bytes;
}

// The largest difference between the GPU's amplitudes and the CPU's
double max_difference(int32_t n)
{
    size_t size = size_t(1) << n;
    std::vector<cuDoubleComplex> gpu(size);
    nova_gpu::state(gpu.data());
    const std::complex<double>* cpu = static_cast<const std::complex<double>*>(nova_q_amplitudes());
    double worst = 0;
    for (size_t i = 0; i < size; i++)
        worst = std::max(worst, std::abs(cpu[i] - std::complex<double>(cuCreal(gpu[i]), cuCimag(gpu[i]))));
    return worst;
}

int option(int argc, char** argv, const char* name, int fallback)
{
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], name)) return atoi(argv[i + 1]);
    return fallback;
}
}  // namespace

int main(int argc, char** argv)
{
    int lo = option(argc, argv, "--min", 16), hi = option(argc, argv, "--max", 28);
    int cpu_max = option(argc, argv, "--cpu-max", 24), depth = option(argc, argv, "--depth", 10);
    int cores = omp_get_max_threads();

    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "no CUDA GPU to run on: %s\n", cudaGetErrorString(e));
        return 1;
    }
    int clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // GB/s: double data rate
    printf("GPU: %s, %.0f GiB, peak memory bandwidth %.0f GB/s; CPU: %d threads\n\n", prop.name,
           prop.totalGlobalMem / 1073741824.0, peak, cores);
    printf("%-8s %6s %6s %12s %12s %10s %8s %8s %10s %7s %10s\n", "circuit", "qubits", "gates", "CPU 1 thr",
           "CPU all", "GPU", "vs 1 thr", "vs all", "GPU GB/s", "% peak", "max |diff|");

    int failures = 0;
    for (int n = lo; n <= hi; n += 2)
    {
        for (const Circuit& c : {layers(n, depth), qft(n)})
        {
            double gpu = run_gpu(c);
            double bandwidth = traffic(c) / gpu / 1e9;
            if (n <= cpu_max)
            {
                double one = run_cpu(c, 1);
                double all = run_cpu(c, cores);
                double diff = max_difference(n);
                bool ok = diff < 1e-9;
                failures += !ok;
                printf("%-8s %6d %6zu %10.1f ms %10.1f ms %7.2f ms %7.0fx %7.1fx %10.0f %6.0f%% %10.1e%s\n",
                       c.name.c_str(), n, c.gates.size(), one * 1e3, all * 1e3, gpu * 1e3, one / gpu, all / gpu,
                       bandwidth, 100 * bandwidth / peak, diff, ok ? "" : "  DIFFERENT");
            }
            else
            {
                printf("%-8s %6d %6zu %13s %12s %7.2f ms %8s %8s %10.0f %6.0f%% %10s\n", c.name.c_str(), n,
                       c.gates.size(), "-", "-", gpu * 1e3, "-", "-", bandwidth, 100 * bandwidth / peak, "-");
            }
            fflush(stdout);
        }
    }
    nova_gpu::reset();
    nova_q_reset();
    if (failures) printf("\n%d circuit(s) differ between the CPU and the GPU\n", failures);
    return failures ? 1 : 0;
}
