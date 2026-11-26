// RUN: not nova-opt %s --convert-nova-to-runtime 2>&1 | FileCheck %s

// Every nova op must be lowered: one no pattern applies to stops the pass, and the error names it, instead of a
// binary being made without it. The runtime works in doubles, so a tensor of floats is not lowered yet.

func.func @main(%n: index) {
  // CHECK: error: failed to legalize operation 'nova.zeros'
  %t = nova.zeros %n, %n : !nova.tensor<f32>
  nova.print %t : !nova.tensor<f32>
  return
}
