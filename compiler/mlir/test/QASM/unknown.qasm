// RUN: not nova-translate --openqasm-to-qnova %s 2>&1 | FileCheck %s

OPENQASM 2.0;
qreg q[3];
// CHECK: [[@LINE+1]]:1: error: the gate 'ccx' is not read yet
ccx q[0], q[1], q[2];
