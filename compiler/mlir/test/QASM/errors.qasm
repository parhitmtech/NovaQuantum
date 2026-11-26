// RUN: not nova-translate --openqasm-to-qnova %s 2>&1 | FileCheck %s
//
// What the importer does not read yet is an error at its line

OPENQASM 3.0;
qubit[2] q;
h q[0];
// CHECK: [[@LINE+1]]:1: error: 'gate' is not read yet: Nova imports straight-line circuits of standard gates
gate mygate a { h a; }
