// SYNTAX TEST "source.openqasm" "the scopes of OpenQASM's tokens"
OPENQASM 3.0;
// <-------- keyword.other.version.openqasm
//       ^^^ constant.numeric.version.openqasm
//          ^ punctuation.terminator.statement.openqasm
include "stdgates.inc";
// <------- keyword.control.import.openqasm
//      ^^^^^^^^^^^^^^ string.quoted.double.openqasm
qubit[3] q;
// <----- storage.type.openqasm
//    ^ constant.numeric.integer.openqasm
bit[3] c;
// <--- storage.type.openqasm
const float theta = pi / 4;
// <----- storage.modifier.openqasm
//    ^^^^^ storage.type.openqasm
//                  ^^ constant.language.openqasm
h q[0];
// <- support.function.gate.openqasm
cx q[0], q[1];
// <-- support.function.gate.openqasm
//     ^ punctuation.separator.comma.openqasm
rz(theta) q[2];
// <-- support.function.gate.openqasm
//^ punctuation.section.parens.openqasm
ry(0.5e-1) q[1];
// ^^^^^^ constant.numeric.float.openqasm
gate bell a, b { h a; cx a, b; }
// <---- storage.type.function.openqasm
//   ^^^^ entity.name.function.openqasm
//             ^ punctuation.section.braces.openqasm
c[0] = measure q[0];
//     ^^^^^^^ keyword.other.quantum.openqasm
measure q[1] -> c[1];
//           ^^ keyword.operator.arrow.openqasm
if (c[0] == 1) x q[2];
// <-- keyword.control.flow.openqasm
//       ^^ keyword.operator.comparison.openqasm
delay[100ns] q[0];
//    ^^^^^ constant.numeric.duration.openqasm
/* a block
   comment */
// <------------ comment.block.openqasm
   // nova.probs q[0], q[1]
//    ^^^^^^^^^^ keyword.other.directive.nova.openqasm
