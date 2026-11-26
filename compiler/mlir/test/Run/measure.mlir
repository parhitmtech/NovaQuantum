// RUN: nova-compile %s -o %t.exe
// RUN: %t.exe | FileCheck %s
//
// Measurement collapses: a Bell pair's two qubits always agree, so measuring the first fixes the second. The
// simulator draws the outcome from NOVA_SEED (or a fixed seed), and both qubits print the same bit.

// CHECK: [[BIT:[01]]]
// CHECK-NEXT: [[BIT]]
func.func @main() {
  %a = qnova.alloc
  %b = qnova.alloc
  %a1 = qnova.h %a
  %a2, %b1 = qnova.cx %a1, %b
  %m, %a3 = qnova.measure %a2
  %n, %b2 = qnova.measure %b1
  nova.print %m : i1
  nova.print %n : i1
  return
}
