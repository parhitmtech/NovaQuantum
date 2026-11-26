// RUN: nova-opt %s --cse --canonicalize | FileCheck %s
// RUN: nova-opt %s --loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

// MLIR's general passes work on nova ops through the effects each op declares, with no code of Nova's. A Nova
// tensor can change in place, so an op that makes one allocates: --cse never merges two of them, and LICM never
// hoists one out of a loop. An op that can stop the program with a runtime error writes nova.runtime_error, so it
// stays even when nothing uses its result. print writes, so it stays.

// CHECK-LABEL: func.func @twice
func.func @twice(%n: index, %k: f64) {
  // A zeros nothing uses still stays: a negative size would stop the program
  // CHECK-COUNT-2: nova.zeros
  %a = nova.zeros %n, %n : !nova.tensor<f64>
  %unused = nova.zeros %n, %n : !nova.tensor<f32>
  // A scale cannot fail, so one nothing uses goes
  // CHECK-NOT: nova.scale
  %scaled = nova.scale %a, %k : !nova.tensor<f64>
  // Two products are two tensors: a write to one must not show in the other
  // CHECK: %[[C:.*]] = nova.matmul
  // CHECK: %[[D:.*]] = nova.matmul
  %c = nova.matmul %a, %a : !nova.tensor<f64>
  %d = nova.matmul %a, %a : !nova.tensor<f64>
  // CHECK: nova.print %[[C]]
  // CHECK: nova.print %[[D]]
  nova.print %c : !nova.tensor<f64>
  nova.print %d : !nova.tensor<f64>
  return
}

// A read can fail (an index out of range), so each read stays where it is, even one whose value is unused
// CHECK-LABEL: func.func @reads
func.func @reads(%t: !nova.tensor<f64>, %i: index, %x: f64) -> (f64, f64) {
  // CHECK: %[[R:.*]] = nova.get %{{.*}}[%{{.*}}] : !nova.tensor<f64>
  // CHECK-NEXT: nova.get
  // CHECK-NEXT: nova.set
  // CHECK-NEXT: %[[S:.*]] = nova.get %{{.*}}[%{{.*}}] : !nova.tensor<f64>
  // CHECK-NEXT: return %[[R]], %[[S]]
  %r1 = nova.get %t[%i] : !nova.tensor<f64>
  %unused = nova.get %t[%i] : !nova.tensor<f64>
  nova.set %x, %t[%i] : !nova.tensor<f64>
  %r3 = nova.get %t[%i] : !nova.tensor<f64>
  return %r1, %r3 : f64, f64
}

// A zeros in a loop is a new tensor each iteration, so LICM leaves it there; the constant moves out
// LICM-LABEL: func.func @fresh
// LICM: arith.constant 1.000000e+00 : f64
// LICM: scf.for
// LICM-NEXT: nova.zeros
func.func @fresh(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %one = arith.constant 1.0 : f64
    %t = nova.zeros %n, %n : !nova.tensor<f64>
    nova.set %one, %t[%i, %i] : !nova.tensor<f64>
    nova.print %t : !nova.tensor<f64>
  }
  return
}

// A divisor check stays unless the divisor is a constant other than 0: x / 2 needs none, x / 0 must still fail
// CHECK-LABEL: func.func @divisors
func.func @divisors(%x: i32, %y: i32) -> (i32, i32, i32) {
  %c2 = arith.constant 2 : i32
  %c0 = arith.constant 0 : i32
  // CHECK-NOT: nova.check_divisor %c2
  // CHECK: nova.check_divisor %c0
  // CHECK: nova.check_divisor %arg1
  nova.check_divisor %c2 : i32
  %a = arith.divsi %x, %c2 : i32
  nova.check_divisor %c0 : i32
  %b = arith.divsi %x, %c0 : i32
  nova.check_divisor %y : i32
  %c = arith.divsi %x, %y : i32
  return %a, %b, %c : i32, i32, i32
}
