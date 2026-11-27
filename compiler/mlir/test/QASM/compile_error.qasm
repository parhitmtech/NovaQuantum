// RUN: not nova-compile %s --emit=mlir 2>&1 | FileCheck %s
//
// A circuit that does not import stops nova-compile with the error in the form editors read (file:line:col: error:),
// with the source line, and a non-zero exit status.
OPENQASM 3.0;
qubit[2] q;
h q[0];
// CHECK: compile_error.qasm:[[@LINE+2]]:1: error: the gate 'foo' is not read yet
// CHECK-NEXT: foo q[1];
foo q[1];
