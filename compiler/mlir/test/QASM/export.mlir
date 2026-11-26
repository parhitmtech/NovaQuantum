// RUN: nova-translate --qnova-to-openqasm %s | FileCheck %s

// A qnova circuit as OpenQASM 3. The SSA chains become named wires: each alloc is q[k], and a gate's results are its
// operands' qubits. The native r(theta, phi) is U(theta, phi - pi/2, pi/2 - phi); what OpenQASM has no statement for,
// an atom move and Nova's probabilities, goes in comments that other tools skip and the importer reads back.

// CHECK:      OPENQASM 3.0;
// CHECK-NEXT: include "stdgates.inc";
// CHECK-NEXT: qubit[3] q;
// CHECK-NEXT: bit[1] c;
// CHECK-NEXT: h q[0];
// CHECK-NEXT: cx q[0], q[1];
// CHECK-NEXT: rz(0.25) q[1];
// CHECK-NEXT: cp(0.5) q[1], q[2];
// CHECK-NEXT: U(1, 0, 0) q[2];
// CHECK-NEXT: swap q[0], q[2];
// CHECK-NEXT: // nova.move q[0] to [1, 0]
// CHECK-NEXT: c[0] = measure q[1];
// CHECK-NEXT: // nova.probs q[0], q[2]
func.func @main() {
  %quarter = arith.constant 0.25 : f64
  %half = arith.constant 0.5 : f64
  %one = arith.constant 1.0 : f64
  %halfpi = arith.constant 1.5707963267948966 : f64
  %a = qnova.alloc
  %b = qnova.alloc
  %c = qnova.alloc
  %a1 = qnova.h %a
  %a2, %b1 = qnova.cx %a1, %b
  %b2 = qnova.rz %b1, %quarter
  %b3, %c1 = qnova.cp %b2, %c, %half
  %c2 = qnova.r %c1, %one, %halfpi
  %a3, %c3 = qnova.swap %a2, %c2
  %a4 = qnova.move %a3 to [1, 0]
  %bit, %b4 = qnova.measure %b3
  qnova.probs %a4, %c3
  return
}
