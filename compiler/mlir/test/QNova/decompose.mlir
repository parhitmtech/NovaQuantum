// RUN: nova-opt %s --qnova-decompose | FileCheck %s

// Logical gates as the neutral-atom machine's native ones (r, rz, cz), up to a global phase. Nothing is optimized
// here: canonicalize does that after.

// CHECK-LABEL: func.func @one
func.func @one(%q: !qnova.qubit, %t: f64) -> !qnova.qubit {
  // CHECK-DAG: %[[PI:.*]] = arith.constant 3.1415926535897931 : f64
  // CHECK-DAG: %[[HALF_PI:.*]] = arith.constant 1.5707963267948966 : f64
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
  // h: rz(pi), then r(pi/2, pi/2)
  // CHECK: %[[Z:.*]] = qnova.rz %arg0, %[[PI]]
  // CHECK: %[[H:.*]] = qnova.r %[[Z]], %[[HALF_PI]], %[[HALF_PI]]
  %q1 = qnova.h %q
  // x: r(pi, 0); rx(t): r(t, 0)
  // CHECK: %[[X:.*]] = qnova.r %[[H]], %[[PI]], %[[ZERO]]
  // CHECK: %[[RX:.*]] = qnova.r %[[X]], %arg1, %[[ZERO]]
  %q2 = qnova.x %q1
  %q3 = qnova.rx %q2, %t
  // CHECK: return %[[RX]]
  return %q3 : !qnova.qubit
}

// cx: h on the target, cz, h on the target
// CHECK-LABEL: func.func @cx
func.func @cx(%a: !qnova.qubit, %b: !qnova.qubit) -> (!qnova.qubit, !qnova.qubit) {
  // CHECK: qnova.rz %arg1
  // CHECK: %[[T:.*]] = qnova.r
  // CHECK: %[[A:.*]], %[[B:.*]] = qnova.cz %arg0, %[[T]]
  // CHECK: qnova.rz %[[B]]
  // CHECK: %[[T2:.*]] = qnova.r
  // CHECK: return %[[A]], %[[T2]]
  %a1, %b1 = qnova.cx %a, %b
  return %a1, %b1 : !qnova.qubit, !qnova.qubit
}

// swap is three cx, each three native steps: three cz, and no logical gate left
// CHECK-LABEL: func.func @swap
// CHECK-COUNT-3: qnova.cz
// CHECK-NOT: qnova.cx
// CHECK-NOT: qnova.swap
func.func @swap(%a: !qnova.qubit, %b: !qnova.qubit) -> (!qnova.qubit, !qnova.qubit) {
  %a1, %b1 = qnova.swap %a, %b
  return %a1, %b1 : !qnova.qubit, !qnova.qubit
}
