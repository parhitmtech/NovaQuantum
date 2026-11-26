// RUN: not nova-opt %s --qnova-to-atom 2>&1 | FileCheck %s

// Hardware instructions need a schedule: sites, moves and layers
// CHECK: error: the circuit is not scheduled: run --qnova-schedule first
func.func @main() {
  %a = qnova.alloc
  qnova.probs %a
  return
}
