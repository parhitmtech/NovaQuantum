// A GHZ state in OpenQASM 3, as Qiskit or CUDA-Q write it.
// nova-compile ghz.qasm --run --hardware --qnova-stats: compiled to timed hardware instructions, then run.
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
