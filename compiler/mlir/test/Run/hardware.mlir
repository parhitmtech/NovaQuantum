// RUN: nova-compile %s --hardware --qnova-stats -o %t.exe 2>&1 | FileCheck %s --check-prefix=TIME
// RUN: %t.exe | FileCheck %s
// RUN: nova-compile %s --logical -o %t.logical
// RUN: %t.logical | FileCheck %s
//
// A circuit compiled all the way down, to the neutral-atom machine's timed instructions (--hardware), and run on the
// simulator by site: a load puts an atom at a site, a move carries it, each pulse acts on the atom at its site. It
// gives the circuit's probabilities, as written (--logical), so no step from logical gates to hardware instructions
// changed what the circuit computes. (The GHZ state of a, b, c, then cx(c, a) and cx(a, c), which leave a in |0>; d
// rotated to |1> with probability 0.75.)

// TIME: atom-timing @main: {{[0-9]+}} layers, {{[0-9]+}} instructions
// CHECK:      |0000>: 0.1250
// CHECK-NEXT: |0001>: 0.3750
// CHECK-NEXT: |0110>: 0.1250
// CHECK-NEXT: |0111>: 0.3750
func.func @main() {
  %t = arith.constant 2.0943951023931953 : f64
  %a = qnova.alloc
  %b = qnova.alloc
  %c = qnova.alloc
  %d = qnova.alloc
  %a1 = qnova.h %a
  %a2, %b1 = qnova.cx %a1, %b
  %b2, %c1 = qnova.cx %b1, %c
  %c2, %a3 = qnova.cx %c1, %a2
  %a4, %c3 = qnova.cx %a3, %c2
  %d1 = qnova.ry %d, %t
  qnova.probs %a4, %b2, %c3, %d1
  return
}
