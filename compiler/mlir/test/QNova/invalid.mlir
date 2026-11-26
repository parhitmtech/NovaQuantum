// RUN: nova-opt %s --split-input-file --verify-diagnostics

// A qubit value is used at most once: a quantum state cannot be copied (no-cloning)
func.func @cloned() {
  %q = qnova.alloc
  // expected-error @+1 {{'qnova.h' op gives a qubit that is used more than once}}
  %a = qnova.h %q
  %b = qnova.x %a
  %c = qnova.z %a
  return
}

// -----

func.func @twice_alloc() {
  // expected-error @+1 {{'qnova.alloc' op gives a qubit that is used more than once}}
  %q = qnova.alloc
  %a, %b = qnova.cz %q, %q
  return
}

// -----

func.func @too_many(%q: !qnova.qubit) {
  // expected-error @+1 {{'qnova.probs' op prints up to 12 qubits, not 13}}
  qnova.probs %q, %q, %q, %q, %q, %q, %q, %q, %q, %q, %q, %q, %q
  return
}

// -----

// A loop may not consume a qubit from outside it: it would consume it once per iteration. (Lowering carries a
// qubit through a loop as a block argument instead.)
func.func @loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{'qnova.alloc' op gives a qubit that is used more than once}}
  %q = qnova.alloc
  scf.for %i = %c0 to %n step %c1 {
    %q1 = qnova.h %q
  }
  return
}
