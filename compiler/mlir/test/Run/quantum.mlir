// RUN: nova-compile %s --logical -o %t.logical
// RUN: nova-compile %s --qnova-stats -o %t.native 2>&1 | FileCheck %s --check-prefix=STATS
// RUN: %t.logical | FileCheck %s
// RUN: %t.native | FileCheck %s
//
// A circuit run on the runtime's statevector simulator, as written (--logical) and compiled for the neutral-atom
// machine (decomposed into r, rz and cz, optimized, routed and scheduled): both give the same probabilities, so
// compiling kept what the circuit computes.

// A GHZ state on three qubits, and on a fourth a rotation whose probabilities are not a half: ry(2 pi / 3) leaves
// |1> with probability sin^2(pi / 3) = 0.75
// CHECK:      |0000>: 0.1250
// CHECK-NEXT: |0001>: 0.3750
// CHECK-NEXT: |1110>: 0.1250
// CHECK-NEXT: |1111>: 0.3750
// STATS: qnova-stats @main: 4 qubits, {{[0-9]+}} gates ({{.*}}), 2 two-qubit
func.func @main() {
  %t = arith.constant 2.0943951023931953 : f64
  %a = qnova.alloc
  %b = qnova.alloc
  %c = qnova.alloc
  %d = qnova.alloc
  %a1 = qnova.h %a
  %a2, %b1 = qnova.cx %a1, %b
  %b2, %c1 = qnova.cx %b1, %c
  %d1 = qnova.ry %d, %t
  qnova.probs %a2, %b2, %c1, %d1
  return
}
