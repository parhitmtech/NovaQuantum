// RUN: nova-compile %s --hardware -o %t.exe
// RUN: %t.exe | FileCheck %s
// RUN: nova-compile %s --logical -o %t.logical
// RUN: %t.logical | FileCheck %s
//
// An OpenQASM 2 circuit from another toolchain, compiled by nova-compile all the way to the neutral-atom machine's
// timed instructions and run on the simulator by site; and run as written. A GHZ state on three qubits, and a fourth
// rotated to |1> with probability sin^2(pi / 3) = 0.75.

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
ry(2*pi/3) q[3];
// nova.probs q[0], q[1], q[2], q[3]

// CHECK:      |0000>: 0.1250
// CHECK-NEXT: |0001>: 0.3750
// CHECK-NEXT: |1110>: 0.1250
// CHECK-NEXT: |1111>: 0.3750
