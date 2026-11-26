// Nova's quantum runtime: a statevector simulator that compiled qnova circuits call (--convert-nova-to-runtime turns
// each qnova op into a call here). A qubit is an index; the state of n qubits is 2^n complex amplitudes, qubit k
// being bit k of a basis state's index. Each gate is a 2x2 (or diagonal 4x4) update of pairs of amplitudes.
//
// It runs the logical gates (h, cx, ...) and the native ones (r, rz, cz) alike, so a circuit can be run before and
// after --qnova-to-native, and the probabilities compared: utils/qbench.py and test/Run/quantum*.mlir do that.
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_QUBITS 24

static double complex* state;
static int32_t qubits;
static uint64_t rng = 0x9E3779B97F4A7C15ull;  // xorshift64*, seeded from NOVA_SEED if set, so runs repeat

static void fail(const char* message)
{
    fflush(stdout);
    fprintf(stderr, "Runtime error: %s\n", message);
    exit(1);
}

static double random_unit(void)
{
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

// A new qubit, |0>: the state tensored with |0>, so every amplitude with the new bit set is 0
int32_t nova_q_alloc(void)
{
    if (qubits == MAX_QUBITS) fail("more qubits than the simulator holds (24)");
    size_t size = (size_t)1 << qubits;
    state = (double complex*)realloc(state, 2 * size * sizeof(double complex));
    if (qubits == 0) state[0] = 1;
    for (size_t i = size; i < 2 * size; i++) state[i] = 0;
    return qubits++;
}

// The 2x2 matrix [[a, b], [c, d]] on qubit q: each pair of basis states that differ only in bit q
static void apply(int32_t q, double complex a, double complex b, double complex c, double complex d)
{
    size_t size = (size_t)1 << qubits, bit = (size_t)1 << q;
    for (size_t i = 0; i < size; i++)
    {
        if (i & bit) continue;
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
    for (size_t i = 0; i < size; i++)
        if ((i & both) == both) state[i] = -state[i];
}

void nova_q_cp(int32_t a, int32_t b, double lambda)
{
    size_t size = (size_t)1 << qubits, both = ((size_t)1 << a) | ((size_t)1 << b);
    double complex phase = cexp(I * lambda);
    for (size_t i = 0; i < size; i++)
        if ((i & both) == both) state[i] *= phase;
}

void nova_q_cx(int32_t control, int32_t target)
{
    size_t size = (size_t)1 << qubits, c = (size_t)1 << control, t = (size_t)1 << target;
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
    for (size_t i = 0; i < size; i++)
        if (i & bit) p1 += creal(state[i] * conj(state[i]));
    int32_t outcome = random_unit() < p1;
    double norm = sqrt(outcome ? p1 : 1 - p1);
    for (size_t i = 0; i < size; i++)
        state[i] = ((i & bit) != 0) == outcome ? state[i] / norm : 0;
    return outcome;
}

// The probability of each basis state of the n qubits q[0..n) (n up to 12, the first leftmost), printed when at
// least 0.00005, as |01>: 0.5000. Printed to four places, so a circuit and its compiled form print the same.
void nova_q_probs(int32_t n, int32_t q0, int32_t q1, int32_t q2, int32_t q3, int32_t q4, int32_t q5, int32_t q6,
                  int32_t q7, int32_t q8, int32_t q9, int32_t q10, int32_t q11)
{
    int32_t q[12] = {q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11};
    static double probs[4096];
    for (size_t k = 0; k < ((size_t)1 << n); k++) probs[k] = 0;
    size_t size = (size_t)1 << qubits;
    for (size_t i = 0; i < size; i++)
    {
        size_t k = 0;
        for (int32_t j = 0; j < n; j++) k = (k << 1) | ((i >> q[j]) & 1);
        probs[k] += creal(state[i] * conj(state[i]));
    }
    for (size_t k = 0; k < ((size_t)1 << n); k++)
    {
        if (probs[k] < 0.00005) continue;
        printf("|");
        for (int32_t j = n - 1; j >= 0; j--) printf("%d", (int)((k >> j) & 1));
        printf(">: %.4f\n", probs[k]);
    }
}
