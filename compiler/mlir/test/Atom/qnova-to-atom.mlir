// RUN: nova-opt %s --qnova-schedule --qnova-to-atom | FileCheck %s
// RUN: nova-opt %s --qnova-schedule --qnova-to-atom --atom-timing -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIME
// RUN: nova-opt %s --qnova-schedule --qnova-to-atom="raman-pi-ns=2000" --atom-timing -o /dev/null 2>&1 | FileCheck %s --check-prefix=SLOW

// The circuit of QNova/schedule.mlir, scheduled, as the machine's timed instructions. The qubit values are gone:
// each instruction names its atom's site, as the hardware does, and the layers run one after another, each as long as
// its longest instruction. By hand, with the default timing: Raman pulses by pi take 1000 ns; a Rydberg pulse 400 ns;
// d's move of one site 15000 ns (pick up and drop) + 10000 ns (travel); then the readout.

// CHECK-LABEL: func.func @main()
// CHECK-SAME: atom.duration_ns = 1027800 : i64
// CHECK-NOT: qnova.alloc
// CHECK: atom.load [0, 0]
// CHECK: atom.load [2, 0]
// CHECK: atom.load [1, 1]
// CHECK: atom.load [0, 2]
// CHECK: atom.raman [0, 0] theta {{.*}} at 0 for 1000
// CHECK: atom.raman [1, 1] theta {{.*}} at 0 for 1000
// CHECK: atom.raman [2, 0] theta {{.*}} at 0 for 1000
// One global Rydberg pulse for the layer's cz
// CHECK: atom.rydberg [0, 0, 1, 1] at 1000 for 400
// CHECK: atom.move [0, 2] to [0, 1] at 1400 for 25000
// CHECK: atom.rydberg [0, 0, 0, 1] at 26400 for 400
// CHECK: atom.raman [0, 1] theta {{.*}} at 26800 for 1000
// CHECK: atom.readout [0, 0, 2, 0, 1, 1, 0, 1] at 27800 for 1000000

// TIME: atom-timing @main: 5 layers, 12 instructions, 1027.8 us (raman 2.0 us, rydberg 0.8 us, move 25.0 us, readout 1000.0 us)
// The timing model is the pass's options: slower Raman pulses make both Raman layers twice as long
// SLOW: raman 4.0 us
func.func @main() {
  %a = qnova.alloc
  %b = qnova.alloc
  %c = qnova.alloc
  %d = qnova.alloc
  %pi = arith.constant 3.14159 : f64
  %a1 = qnova.r %a, %pi, %pi
  %c1 = qnova.r %c, %pi, %pi
  %a2, %c2 = qnova.cz %a1, %c1
  %a3, %d1 = qnova.cz %a2, %d
  %b1 = qnova.r %b, %pi, %pi
  %d2 = qnova.r %d1, %pi, %pi
  qnova.probs %a3, %b1, %c2, %d2
  return
}
