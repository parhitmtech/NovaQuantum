// RUN: nova-opt %s --convert-nova-to-runtime | FileCheck %s

// A tensor is a pointer to the runtime's NovaTensor (!llvm.ptr), everywhere one can be: a function's signature, a
// call, a return, scf's results and block arguments. Each nova op is a call; one that can fail takes the Nova line
// first, and sizes and subscripts come as a count and six i64 values, the unused ones 0.

// CHECK-LABEL: func.func @product(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
// CHECK: %[[C:.*]] = call @nova_matmul(%{{.*}}, %arg0, %arg1) : (i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK: return %[[C]] : !llvm.ptr
func.func @product(%a: !nova.tensor<f64>, %b: !nova.tensor<f64>) -> !nova.tensor<f64> {
  %c = nova.matmul %a, %b : !nova.tensor<f64>
  return %c : !nova.tensor<f64>
}

// CHECK-LABEL: func.func @elements(%arg0: index, %arg1: f64, %arg2: i1) -> f64 {
func.func @elements(%n: index, %x: f64, %flag: i1) -> f64 {
  // CHECK: arith.index_cast %arg0 : index to i64
  // CHECK: %[[T:.*]] = call @nova_zeros(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[Z:.*]], %[[Z]], %[[Z]], %[[Z]])
  // CHECK-SAME: : (i32, i32, i64, i64, i64, i64, i64, i64) -> !llvm.ptr
  %t = nova.zeros %n, %n : !nova.tensor<f64>
  // CHECK: call @nova_set(%{{.*}}, %[[T]], %arg1, %{{.*}}
  nova.set %x, %t[%n, %n] : !nova.tensor<f64>
  // CHECK: %[[S:.*]] = call @nova_scale(%[[T]], %arg1) : (!llvm.ptr, f64) -> !llvm.ptr
  %s = nova.scale %t, %x : !nova.tensor<f64>
  // The if's result is a pointer too
  // CHECK: %[[R:.*]] = scf.if %arg2 -> (!llvm.ptr) {
  // CHECK:   call @nova_add(%{{.*}}, %[[S]], %[[T]])
  %r = scf.if %flag -> (!nova.tensor<f64>) {
    %sum = nova.add %s, %t : !nova.tensor<f64>
    scf.yield %sum : !nova.tensor<f64>
  } else {
    scf.yield %t : !nova.tensor<f64>
  }
  // CHECK: call @nova_print_tensor(%[[R]]) : (!llvm.ptr) -> ()
  nova.print %r : !nova.tensor<f64>
  // CHECK: %[[V:.*]] = call @nova_get(%{{.*}}, %[[R]], %{{.*}}) : (i32, !llvm.ptr, i32, i64, i64, i64, i64, i64, i64) -> f64
  %v = nova.get %r[%n] : !nova.tensor<f64>
  // CHECK: return %[[V]] : f64
  return %v : f64
}

// A tensor not assigned yet is poison of the pointer type
// CHECK-LABEL: func.func @unassigned
// CHECK: ub.poison : !llvm.ptr
func.func @unassigned() -> !nova.tensor<f64> {
  %p = ub.poison : !nova.tensor<f64>
  return %p : !nova.tensor<f64>
}

// An integer division's check is a call that stops the program when the divisor is 0
// CHECK-LABEL: func.func @divide
// CHECK: call @nova_check_divisor(%{{.*}}, %arg1) : (i32, i32) -> ()
func.func @divide(%a: i32, %b: i32) -> i32 {
  nova.check_divisor %b : i32
  %q = arith.divsi %a, %b : i32
  return %q : i32
}
