// RUN: nova-opt %s --split-input-file --verify-diagnostics

// A Rydberg pulse drives pairs of neighbouring atoms: a pair two sites apart is not in its interaction radius
func.func @far() {
  // expected-error @+1 {{'atom.rydberg' op pairs atoms that are not neighbours: (0, 0) and (2, 0)}}
  atom.rydberg [0, 0, 2, 0] at 0 for 400
  return
}

// -----

func.func @odd() {
  // expected-error @+1 {{'atom.rydberg' op needs pairs of sites, four numbers each}}
  atom.rydberg [0, 0, 1] at 0 for 400
  return
}
