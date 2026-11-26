// RUN: nova-opt %s | nova-opt | FileCheck %s

// The atom dialect's instructions print as they are read: each names sites, and the time it starts and takes (ns)

// CHECK-LABEL: func.func @program
func.func @program() {
  // CHECK: atom.load [0, 0]
  atom.load [0, 0]
  atom.load [1, 1]
  // CHECK: atom.raman [0, 0] theta 1.5{{.*}} phi 0.000000e+00 at 0 for 500
  atom.raman [0, 0] theta 1.5707963267948966 phi 0.0 at 0 for 500
  // CHECK: atom.shift [1, 1] theta 3.14{{.*}} at 0 for 500
  atom.shift [1, 1] theta 3.1415926535897931 at 0 for 500
  // CHECK: atom.rydberg [0, 0, 1, 1] at 500 for 400
  atom.rydberg [0, 0, 1, 1] at 500 for 400
  // CHECK: atom.move [1, 1] to [1, 0] at 900 for 25000
  atom.move [1, 1] to [1, 0] at 900 for 25000
  // CHECK: %{{.*}} = atom.measure [0, 0] at 25900 for 1000000
  %bit = atom.measure [0, 0] at 25900 for 1000000
  // CHECK: atom.readout [1, 0] at 1025900 for 1000000
  atom.readout [1, 0] at 1025900 for 1000000
  return
}
