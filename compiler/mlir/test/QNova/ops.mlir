// RUN: nova-opt %s | nova-opt | FileCheck %s

// Every qnova op prints as it is read. Qubits are values: each gate takes the qubits it acts on and gives new ones.

// CHECK-LABEL: func.func @gates
func.func @gates(%theta: f64, %phi: f64) {
  // CHECK: %[[A:.*]] = qnova.alloc
  // CHECK: %[[B:.*]] = qnova.alloc
  %a = qnova.alloc
  %b = qnova.alloc
  // CHECK: qnova.h %[[A]]
  %a1 = qnova.h %a
  %a2 = qnova.x %a1
  %a3 = qnova.y %a2
  %a4 = qnova.z %a3
  %a5 = qnova.s %a4
  %a6 = qnova.t %a5
  // CHECK: qnova.rx %{{.*}}, %arg0
  %a7 = qnova.rx %a6, %theta
  %a8 = qnova.ry %a7, %theta
  %a9 = qnova.rz %a8, %theta
  // CHECK: qnova.r %{{.*}}, %arg0, %arg1
  %a10 = qnova.r %a9, %theta, %phi
  // CHECK: %{{.*}}, %{{.*}} = qnova.cx %{{.*}}, %[[B]]
  %a11, %b1 = qnova.cx %a10, %b
  %a12, %b2 = qnova.cz %a11, %b1
  %a13, %b3 = qnova.cp %a12, %b2, %theta
  %a14, %b4 = qnova.swap %a13, %b3
  // CHECK: qnova.move %{{.*}} to [2, 3]
  %b5 = qnova.move %b4 to [2, 3]
  // CHECK: %{{.*}}, %{{.*}} = qnova.measure
  %bit, %a15 = qnova.measure %a14
  // CHECK: qnova.probs %{{.*}}, %{{.*}}
  qnova.probs %a15, %b5
  return
}
