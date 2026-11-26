// The drop-in check: the calls a compiled circuit makes into the quantum runtime (what --convert-nova-to-runtime
// writes), made by hand, so the runtime can be checked on a GPU machine without the MLIR toolchain. `make
// dropin-check` builds this twice, with the CPU runtime (nova_quantum.c) and with the GPU runtime
// (libnova_runtime_gpu.a), and the two must print the same.
//
// The circuit: teleportation (examples/quantum/teleport.nova), with seeded measurements, then a GHZ state on six
// qubits, rotated, and its probabilities; then the hardware instructions of a Bell pair, by site.
#include <stdint.h>
#include <stdio.h>

int32_t nova_q_alloc(void);
void nova_q_h(int32_t q);
void nova_q_x(int32_t q);
void nova_q_z(int32_t q);
void nova_q_ry(int32_t q, double t);
void nova_q_rz(int32_t q, double t);
void nova_q_cx(int32_t c, int32_t t);
void nova_q_cp(int32_t a, int32_t b, double l);
void nova_q_swap(int32_t a, int32_t b);
int32_t nova_q_measure(int32_t q);
void nova_q_reset(void);
void nova_q_probs(int32_t n, int32_t q0, int32_t q1, int32_t q2, int32_t q3, int32_t q4, int32_t q5, int32_t q6,
                  int32_t q7, int32_t q8, int32_t q9, int32_t q10, int32_t q11);
void nova_hw_load(int32_t x, int32_t y);
void nova_hw_raman(int32_t x, int32_t y, double theta, double phi);
void nova_hw_shift(int32_t x, int32_t y, double theta);
void nova_hw_rydberg(int32_t x0, int32_t y0, int32_t x1, int32_t y1);
void nova_hw_readout(int32_t n, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t x3,
                     int32_t y3, int32_t x4, int32_t y4, int32_t x5, int32_t y5, int32_t x6, int32_t y6, int32_t x7,
                     int32_t y7, int32_t x8, int32_t y8, int32_t x9, int32_t y9, int32_t x10, int32_t y10, int32_t x11,
                     int32_t y11);

int main(void)
{
    // Teleportation: c ends in a's state, |1> with probability 0.75, whatever the measurements
    int32_t a = nova_q_alloc(), b = nova_q_alloc(), c = nova_q_alloc();
    nova_q_ry(a, 2.0943951023931953);
    nova_q_h(b);
    nova_q_cx(b, c);
    nova_q_cx(a, b);
    nova_q_h(a);
    int32_t m1 = nova_q_measure(a), m2 = nova_q_measure(b);
    if (m2) nova_q_x(c);
    if (m1) nova_q_z(c);
    printf("teleport: measured %d %d\n", m1, m2);
    nova_q_probs(1, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    nova_q_reset();

    // A GHZ state on six qubits, then phases and a swap
    int32_t q[6];
    for (int k = 0; k < 6; k++) q[k] = nova_q_alloc();
    nova_q_h(q[0]);
    for (int k = 1; k < 6; k++) nova_q_cx(q[k - 1], q[k]);
    nova_q_rz(q[2], 0.7);
    nova_q_cp(q[1], q[4], 1.1);
    nova_q_h(q[3]);
    nova_q_swap(q[0], q[5]);
    nova_q_ry(q[5], 0.3);
    printf("ghz, rotated:\n");
    nova_q_probs(6, q[0], q[1], q[2], q[3], q[4], q[5], 0, 0, 0, 0, 0, 0);
    nova_q_reset();

    // The hardware instructions of a Bell pair: h as rz(pi) then r(pi/2, pi/2), cz, h on the second atom
    nova_hw_load(0, 0);
    nova_hw_load(1, 1);
    nova_hw_shift(0, 0, 3.141592653589793);
    nova_hw_raman(0, 0, 1.5707963267948966, 1.5707963267948966);
    nova_hw_shift(1, 1, 3.141592653589793);
    nova_hw_raman(1, 1, 1.5707963267948966, 1.5707963267948966);
    nova_hw_rydberg(0, 0, 1, 1);
    nova_hw_shift(1, 1, 3.141592653589793);
    nova_hw_raman(1, 1, 1.5707963267948966, 1.5707963267948966);
    printf("bell, as hardware instructions:\n");
    nova_hw_readout(2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    return 0;
}
