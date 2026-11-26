// RUN: nova-opt %s --canonicalize | FileCheck %s

// Peephole rewrites need no analysis on SSA qubits: the gate before a gate on the same qubit is its operand's
// definition. Self-inverse gates cancel; rotations about one axis merge; diagonal gates (rz, cz, cp) commute with
// each other, change nothing on a fresh |0> qubit, and change no probability just before qnova.probs.

// CHECK-LABEL: func.func @cancel
func.func @cancel(%a: !qnova.qubit, %b: !qnova.qubit) -> (!qnova.qubit, !qnova.qubit) {
  // h h, x x, cx cx (same wires) and cz cz (either order) are nothing
  // CHECK-NOT: = qnova.
  // CHECK: return %arg0, %arg1
  %a1 = qnova.h %a
  %a2 = qnova.h %a1
  %a3 = qnova.x %a2
  %a4 = qnova.x %a3
  %a5, %b1 = qnova.cx %a4, %b
  %a6, %b2 = qnova.cx %a5, %b1
  %a7, %b3 = qnova.cz %a6, %b2
  %b4, %a8 = qnova.cz %b3, %a7
  return %a8, %b4 : !qnova.qubit, !qnova.qubit
}

// CHECK-LABEL: func.func @merge
func.func @merge(%q: !qnova.qubit, %x: f64) -> !qnova.qubit {
  %c1 = arith.constant 1.0 : f64
  %c2 = arith.constant 2.0 : f64
  %phi = arith.constant 0.5 : f64
  // Two rz in a row are one, the angles added (folded: 1 + 2 = 3); an rz by 2 pi is nothing
  // CHECK-DAG: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
  // CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f64
  // CHECK: %[[Z:.*]] = qnova.rz %arg0, %[[THREE]]
  %q1 = qnova.rz %q, %c1
  %q2 = qnova.rz %q1, %c2
  %turn = arith.constant 6.2831853071795862 : f64
  %q3 = qnova.rz %q2, %turn
  // Two r about the same axis are one; the sum of a constant and a value stays an addf
  // CHECK: %[[SUM:.*]] = arith.addf %arg1, %{{.*}} : f64
  // CHECK: %[[R:.*]] = qnova.r %[[Z]], %[[SUM]], %[[HALF]]
  // CHECK: return %[[R]]
  %q4 = qnova.r %q3, %x, %phi
  %q5 = qnova.r %q4, %c1, %phi
  return %q5 : !qnova.qubit
}

// An rz after a cz moves before it (both are diagonal, so they commute) and merges with the rz there
// CHECK-LABEL: func.func @commute
func.func @commute(%a: !qnova.qubit, %b: !qnova.qubit, %t: f64) -> (!qnova.qubit, !qnova.qubit) {
  // CHECK: %[[SUM:.*]] = arith.addf %arg2, %arg2
  // CHECK: %[[Z:.*]] = qnova.rz %arg0, %[[SUM]]
  // CHECK: %{{.*}}, %{{.*}} = qnova.cz %[[Z]], %arg1
  // CHECK-NOT: qnova.rz
  %a1 = qnova.rz %a, %t
  %a2, %b1 = qnova.cz %a1, %b
  %a3 = qnova.rz %a2, %t
  return %a3, %b1 : !qnova.qubit, !qnova.qubit
}

// On fresh qubits (|0>) and before probs, diagonal gates do nothing that can be observed; a cp pair merges
// CHECK-LABEL: func.func @diagonal
func.func @diagonal(%t: f64) {
  // CHECK: %[[A:.*]] = qnova.alloc
  // CHECK: %[[B:.*]] = qnova.alloc
  // CHECK: %[[H:.*]] = qnova.h %[[A]]
  // CHECK: %[[X:.*]] = qnova.x %[[B]]
  // CHECK: %[[L:.*]] = arith.addf %arg0, %arg0
  // The two cp (their wires in either order) are one, by the sum of the angles
  // CHECK: %[[PX:.*]], %[[PH:.*]] = qnova.cp %[[X]], %[[H]], %[[L]]
  // CHECK: %[[F:.*]] = qnova.h %[[PH]]
  // CHECK: qnova.probs %[[F]], %[[PX]]
  %a = qnova.alloc
  %b = qnova.alloc
  %a1 = qnova.rz %a, %t
  %a2, %b1 = qnova.cz %a1, %b
  %a3 = qnova.h %a2
  %b2 = qnova.x %b1
  %a4, %b3 = qnova.cp %a3, %b2, %t
  %b4, %a5 = qnova.cp %b3, %a4, %t
  %a6 = qnova.h %a5
  %a7 = qnova.rz %a6, %t
  qnova.probs %a7, %b4
  return
}
