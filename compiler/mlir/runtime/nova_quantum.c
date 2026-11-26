// Nova's quantum runtime: a statevector simulator that compiled qnova circuits call (--convert-nova-to-runtime turns
// each qnova op into a call here). A qubit is an index; the state of n qubits is 2^n complex amplitudes, qubit k
// being bit k of a basis state's index. Each gate is a 2x2 (or diagonal 4x4) update of pairs of amplitudes.
//
// It runs the logical gates (h, cx, ...) and the native ones (r, rz, cz) alike, so a circuit can be run before and
// after --qnova-to-native, and the probabilities compared: utils/qbench.py and test/Run/quantum*.mlir do that.
//
// The same functions exist for the GPU (runtime/gpu): built with NOVA_QUANTUM_GPU, this file leaves the statevector
// functions (nova_q_*) to runtime/gpu/nova_quantum_gpu.cu and keeps what does not touch the state, the printing and
// the hardware instructions by site. Built with OpenMP (-fopenmp), each gate's loop runs on every core.
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#define PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#define PARALLEL_SUM _Pragma("omp parallel for schedule(static) reduction(+ : p1)")
#else
#define PARALLEL_FOR
#define PARALLEL_SUM
#endif

#define MAX_QUBITS 30  // 2^30 amplitudes of 16 bytes: 16 GiB

static void fail(const char* message)
{
    fflush(stdout);
    fprintf(stderr, "nova: runtime error: %s\n", message);
    exit(1);
}

// probs[k] for the 2^n basis states of n qubits, printed when at least 0.00005, as |01>: 0.5000 (the first qubit
// leftmost). Printed to four places, so a circuit and its compiled form, or the CPU and the GPU, print the same.
void nova_q_print_probs(int32_t n, const double* probs)
{
    for (size_t k = 0; k < ((size_t)1 << n); k++)
    {
        if (probs[k] < 0.00005) continue;
        printf("|");
        for (int32_t j = n - 1; j >= 0; j--) printf("%d", (int)((k >> j) & 1));
        printf(">: %.4f\n", probs[k]);
    }
}

// The random numbers measurement outcomes are drawn from: xorshift64*, seeded from NOVA_SEED if set, so runs
// repeat. The GPU runtime draws from this same generator, so a seed gives the same outcomes there.
double nova_q_random(void)
{
    static uint64_t rng = 0x9E3779B97F4A7C15ull;
    static int seeded;
    if (!seeded)
    {
        const char* seed = getenv("NOVA_SEED");
        if (seed) rng = strtoull(seed, NULL, 10) * 0x9E3779B97F4A7C15ull + 1;
        seeded = 1;
    }
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return (double)((rng * 0x2545F4914F6CDD1Dull) >> 11) / 9007199254740992.0;  // [0, 1)
}

#ifndef NOVA_QUANTUM_GPU

static double complex* state;
static int32_t qubits;

// A new qubit, |0>: the state tensored with |0>, so every amplitude with the new bit set is 0
int32_t nova_q_alloc(void)
{
    if (qubits == MAX_QUBITS) fail("more qubits than the simulator holds (30)");
    size_t size = (size_t)1 << qubits;
    double complex* grown = (double complex*)realloc(state, 2 * size * sizeof(double complex));
    if (!grown) fail("out of memory for the state");
    state = grown;
    if (qubits == 0) state[0] = 1;
    for (size_t i = size; i < 2 * size; i++) state[i] = 0;
    return qubits++;
}

// No qubits again: the benchmark (runtime/gpu/qsim_bench.cu) runs one circuit after another
void nova_q_reset(void)
{
    free(state);
    state = NULL;
    qubits = 0;
}

// The amplitudes, for checking the GPU simulator against this one (runtime/gpu/qsim_bench.cu)
const double complex* nova_q_amplitudes(void) { return state; }

// The 2x2 matrix [[a, b], [c, d]] on qubit q: each pair of basis states that differ only in bit q. Iterating over
// the pairs (k with a 0 inserted at bit q) gives each its own iteration, so the loop parallelizes.
static void apply(int32_t q, double complex a, double complex b, double complex c, double complex d)
{
    size_t half = (size_t)1 << (qubits - 1), bit = (size_t)1 << q;
    PARALLEL_FOR
    for (size_t k = 0; k < half; k++)
    {
        size_t i = ((k & ~(bit - 1)) << 1) | (k & (bit - 1));  // bit q of i is 0
        double complex x0 = state[i], x1 = state[i | bit];
        state[i] = a * x0 + b * x1;
        state[i | bit] = c * x0 + d * x1;
    }
}

void nova_q_h(int32_t q) { apply(q, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2); }
void nova_q_x(int32_t q) { apply(q, 0, 1, 1, 0); }
void nova_q_y(int32_t q) { apply(q, 0, -I, I, 0); }
void nova_q_z(int32_t q) { apply(q, 1, 0, 0, -1); }
void nova_q_s(int32_t q) { apply(q, 1, 0, 0, I); }
void nova_q_t(int32_t q) { apply(q, 1, 0, 0, cexp(I * M_PI / 4)); }

void nova_q_rx(int32_t q, double t) { apply(q, cos(t / 2), -I * sin(t / 2), -I * sin(t / 2), cos(t / 2)); }
void nova_q_ry(int32_t q, double t) { apply(q, cos(t / 2), -sin(t / 2), sin(t / 2), cos(t / 2)); }
void nova_q_rz(int32_t q, double t) { apply(q, cexp(-I * t / 2), 0, 0, cexp(I * t / 2)); }

// r(theta, phi): exp(-i theta/2 (cos(phi) X + sin(phi) Y)), the native Raman rotation
void nova_q_r(int32_t q, double t, double phi)
{
    double complex off = -I * sin(t / 2);
    apply(q, cos(t / 2), off * cexp(-I * phi), off * cexp(I * phi), cos(t / 2));
}

// Two-qubit gates, on basis states by their bits
void nova_q_cz(int32_t a, int32_t b)
{
    size_t size = (size_t)1 << qubits, both = ((size_t)1 << a) | ((size_t)1 << b);
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++)
        if ((i & both) == both) state[i] = -state[i];
}

void nova_q_cp(int32_t a, int32_t b, double lambda)
{
    size_t size = (size_t)1 << qubits, both = ((size_t)1 << a) | ((size_t)1 << b);
    double complex phase = cexp(I * lambda);
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++)
        if ((i & both) == both) state[i] *= phase;
}

void nova_q_cx(int32_t control, int32_t target)
{
    size_t size = (size_t)1 << qubits, c = (size_t)1 << control, t = (size_t)1 << target;
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++)
    {
        if (!(i & c) || (i & t)) continue;  // each pair once: control set, target clear
        double complex x = state[i];
        state[i] = state[i | t];
        state[i | t] = x;
    }
}

void nova_q_swap(int32_t a, int32_t b)
{
    size_t size = (size_t)1 << qubits, ba = (size_t)1 << a, bb = (size_t)1 << b;
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++)
    {
        if (!(i & ba) || (i & bb)) continue;  // a set, b clear: exchange with a clear, b set
        size_t j = (i & ~ba) | bb;
        double complex x = state[i];
        state[i] = state[j];
        state[j] = x;
    }
}

// Measure qubit q: 1 with the probability of its bit being set, then the state collapses to the outcome
int32_t nova_q_measure(int32_t q)
{
    size_t size = (size_t)1 << qubits, bit = (size_t)1 << q;
    double p1 = 0;
    PARALLEL_SUM
    for (size_t i = 0; i < size; i++)
        if (i & bit) p1 += creal(state[i] * conj(state[i]));
    int32_t outcome = nova_q_random() < p1;
    double norm = sqrt(outcome ? p1 : 1 - p1);
    PARALLEL_FOR
    for (size_t i = 0; i < size; i++)
        state[i] = ((i & bit) != 0) == outcome ? state[i] / norm : 0;
    return outcome;
}

// probs[k] for the 2^n basis states of the qubits q[0..n), the first the most significant bit of k
void nova_q_probabilities(int32_t n, const int32_t* q, double* probs)
{
    for (size_t k = 0; k < ((size_t)1 << n); k++) probs[k] = 0;
    size_t size = (size_t)1 << qubits;
    for (size_t i = 0; i < size; i++)
    {
        size_t k = 0;
        for (int32_t j = 0; j < n; j++) k = (k << 1) | ((i >> q[j]) & 1);
        probs[k] += creal(state[i] * conj(state[i]));
    }
}

void nova_q_probs(int32_t n, int32_t q0, int32_t q1, int32_t q2, int32_t q3, int32_t q4, int32_t q5, int32_t q6,
                  int32_t q7, int32_t q8, int32_t q9, int32_t q10, int32_t q11)
{
    int32_t q[12] = {q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11};
    static double probs[4096];
    nova_q_probabilities(n, q, probs);
    nova_q_print_probs(n, probs);
}

#else  // the GPU runtime provides these (runtime/gpu/nova_quantum_gpu.cu)

int32_t nova_q_alloc(void);
void nova_q_r(int32_t q, double t, double phi);
void nova_q_rz(int32_t q, double t);
void nova_q_cz(int32_t a, int32_t b);
int32_t nova_q_measure(int32_t q);
void nova_q_probs(int32_t n, int32_t q0, int32_t q1, int32_t q2, int32_t q3, int32_t q4, int32_t q5, int32_t q6,
                  int32_t q7, int32_t q8, int32_t q9, int32_t q10, int32_t q11);

#endif

// ─── Hardware instructions (the atom dialect), on the same simulator ─────────
//
// A compiled program's instructions name sites of the atom array, not qubits. The simulator keeps which qubit's atom
// is at each site: a load puts a new one there, a move carries it to another site, and each instruction acts on the
// qubit at its site. So the hardware program runs on the same statevector as the circuit, and its probabilities can
// be checked against the circuit's.

#define GRID 64

static int32_t at_site[GRID][GRID];  // the qubit at each site, plus one (0: no atom)

static int32_t atom_at(int32_t x, int32_t y)
{
    if (x < 0 || y < 0 || x >= GRID || y >= GRID || !at_site[x][y])
    {
        char message[64];
        snprintf(message, sizeof message, "no atom at site (%d, %d)", x, y);
        fail(message);
    }
    return at_site[x][y] - 1;
}

void nova_hw_load(int32_t x, int32_t y)
{
    if (x < 0 || y < 0 || x >= GRID || y >= GRID || at_site[x][y]) fail("an atom loaded where there is one");
    at_site[x][y] = nova_q_alloc() + 1;
}

void nova_hw_raman(int32_t x, int32_t y, double theta, double phi) { nova_q_r(atom_at(x, y), theta, phi); }

void nova_hw_shift(int32_t x, int32_t y, double theta) { nova_q_rz(atom_at(x, y), theta); }

// One pair of a global Rydberg pulse: the two atoms' cz
void nova_hw_rydberg(int32_t x0, int32_t y0, int32_t x1, int32_t y1) { nova_q_cz(atom_at(x0, y0), atom_at(x1, y1)); }

void nova_hw_move(int32_t x0, int32_t y0, int32_t x1, int32_t y1)
{
    int32_t q = atom_at(x0, y0);
    if (x1 < 0 || y1 < 0 || x1 >= GRID || y1 >= GRID || at_site[x1][y1]) fail("an atom moved onto another");
    at_site[x0][y0] = 0;
    at_site[x1][y1] = q + 1;
}

int32_t nova_hw_measure(int32_t x, int32_t y) { return nova_q_measure(atom_at(x, y)); }

// The probabilities of the atoms at n sites (x, y pairs), as nova_q_probs prints them
void nova_hw_readout(int32_t n, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t x3,
                     int32_t y3, int32_t x4, int32_t y4, int32_t x5, int32_t y5, int32_t x6, int32_t y6, int32_t x7,
                     int32_t y7, int32_t x8, int32_t y8, int32_t x9, int32_t y9, int32_t x10, int32_t y10, int32_t x11,
                     int32_t y11)
{
    int32_t xy[24] = {x0, y0, x1, y1, x2, y2, x3, y3, x4,  y4,  x5,  y5,
                      x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11};
    int32_t q[12] = {0};
    for (int32_t k = 0; k < n; k++) q[k] = atom_at(xy[2 * k], xy[2 * k + 1]);
    nova_q_probs(n, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11]);
}
