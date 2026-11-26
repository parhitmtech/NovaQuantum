// RUN: nova-translate --openqasm-to-qnova %s | FileCheck %s
//
// OpenQASM as another toolchain writes it, read into qnova: registers become allocs, each gate replaces its qubits'
// values with its results (the SSA chains of the circuit's wires), angles are constant expressions, and gates outside
// qnova's own set are written in it (U as rz, ry, rz; sdg as rz(-pi/2); p as rz).

OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
qubit r;
bit[2] c;

// CHECK-LABEL: func.func @main()
// CHECK: %[[Q0:.*]] = qnova.alloc
// CHECK: %[[Q1:.*]] = qnova.alloc
// CHECK: %[[R:.*]] = qnova.alloc
// CHECK: %[[H:.*]] = qnova.h %[[Q0]]
h q[0];
// CHECK: %[[A:.*]], %[[B:.*]] = qnova.cx %[[H]], %[[Q1]]
cx q[0], q[1];
// CHECK: %[[PI4:.*]] = arith.constant 0.78539816339744828 : f64
// CHECK: qnova.rx %[[B]], %[[PI4]]
rx(pi / 4) q[1];
// CHECK: arith.constant -2.3561944901923448 : f64
ry(-3 * pi / 4) r;
// CHECK: arith.constant -1.5707963267948966 : f64
// CHECK: qnova.rz
sdg q[0];
// U(theta, phi, lambda) as rz(lambda), ry(theta), rz(phi)
// CHECK: qnova.rz
// CHECK: qnova.ry
// CHECK: qnova.rz
U(0.1, 0.2, 0.3) r;
/* a block comment,
   over two lines */
cp(1.5e-1) q[1], r;
// CHECK: qnova.cp
c[0] = measure q[0];
// CHECK: qnova.measure
measure q[1] -> c[1];
// CHECK: qnova.measure
barrier q[0], q[1];
// Without a nova.probs comment, the probabilities of every qubit at the end
// CHECK: qnova.probs %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-NEXT: return
