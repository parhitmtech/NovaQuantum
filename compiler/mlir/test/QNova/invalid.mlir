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
