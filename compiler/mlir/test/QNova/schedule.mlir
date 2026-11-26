// RUN: nova-opt %s --qnova-schedule | FileCheck %s
// RUN: nova-opt %s --qnova-schedule --qnova-stats -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS

// Four atoms on a checkerboard of a 3-wide grid: (0,0), (2,0), (1,1), (0,2). (0,0) and (1,1) are neighbours, so their
// cz runs where they are; (0,0) and (0,2) are not, so one moves next to the other first. Each op gets the layer it
// runs in: a layer is one kind of operation (Raman, Rydberg, move), and each op comes after those it depends on.

// CHECK-LABEL: func.func @main()
// CHECK-SAME: qnova.depth = 5 : i64, qnova.move_distance = 1.000000e+00 : f64, qnova.moves = 1 : i64
func.func @main() {
  // CHECK: qnova.alloc {site = array<i64: 0, 0>}
  // CHECK: qnova.alloc {site = array<i64: 2, 0>}
  // CHECK: qnova.alloc {site = array<i64: 1, 1>}
  // CHECK: qnova.alloc {site = array<i64: 0, 2>}
  %a = qnova.alloc
  %b = qnova.alloc
  %c = qnova.alloc
  %d = qnova.alloc
  %pi = arith.constant 3.14159 : f64
  // Raman gates on different atoms share layer 0
  // CHECK: qnova.r %{{.*}} {layer = 0 : i64}
  // CHECK: qnova.r %{{.*}} {layer = 0 : i64}
  %a1 = qnova.r %a, %pi, %pi
  %c1 = qnova.r %c, %pi, %pi
  // Neighbours: a Rydberg layer, no move
  // CHECK: qnova.cz %{{.*}}, %{{.*}} {layer = 1 : i64}
  %a2, %c2 = qnova.cz %a1, %c1
  // Not neighbours: d moves to the free site next to a nearest it, (0,1), then the cz
  // CHECK: %[[M:.*]] = qnova.move %{{.*}} to [0, 1] {layer = 2 : i64}
  // CHECK: qnova.cz %{{.*}}, %[[M]] {layer = 3 : i64}
  %a3, %d1 = qnova.cz %a2, %d
  // b's Raman gate depends on nothing before it: layer 0
  // CHECK: qnova.r %{{.*}} {layer = 0 : i64}
  %b1 = qnova.r %b, %pi, %pi
  // after the second cz
  // CHECK: qnova.r %{{.*}} {layer = 4 : i64}
  %d2 = qnova.r %d1, %pi, %pi
  qnova.probs %a3, %b1, %c2, %d2
  return
}

// STATS: qnova-stats @main: 4 qubits, 6 gates (cz 2, r 4), 2 two-qubit, 1 moves (1.000 sites), depth 5
