// The qnova dialect's type and ops: allocation and measurement, the logical gates a circuit is written in, and the
// native gates of a neutral-atom machine (Raman rotations r and rz, the Rydberg cz), plus the atom moves routing adds
#ifndef QNOVA_OPS
#define QNOVA_OPS

include "QNova/QNovaDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// !qnova.qubit: the state of one qubit at one point of the circuit
def QNova_QubitType : TypeDef<QNova_Dialect, "Qubit"> {
    let mnemonic = "qubit";
    let summary = "one qubit, as a value each gate consumes and produces";
}

// Each qubit an op gives is used at most once: a quantum state cannot be copied (the no-cloning theorem), so two
// uses of one value would be two gates on one state at once. A C++ trait (QNovaOps.h) checks it for every op.
def LinearQubits : NativeOpTrait<"LinearQubits">;

// Gates are Pure: a function of their input states. CSE never merges two (it would need one value used twice), and
// a gate whose qubits nothing uses again is dead: by the no-signaling theorem, a gate on a qubit no one measures
// changes no other qubit's statistics.
class QNova_Gate<string mnemonic, list<Trait> traits = []>
    : QNova_Op<mnemonic, !listconcat([Pure, LinearQubits], traits)>;

// ─── Allocation, measurement, output ────────────────────────────────────────

// %q = qnova.alloc : a fresh qubit, |0>. It allocates (MemAlloc), so two are never merged.
def QNova_AllocOp : QNova_Op<"alloc", [LinearQubits]> {
    let summary = "a new qubit in state |0>";
    let results = (outs Res<QNova_QubitType, "the qubit", [MemAlloc]>:$qubit);
    let builders = [OpBuilder<(ins), [{ build($_builder, $_state, ::qnova::qubit($_builder)); }]>];
    let assemblyFormat = "attr-dict";
}

// %bit, %q1 = qnova.measure %q0 : the qubit's value as an i1, and the qubit after (collapsed to it). A measurement
// draws a random outcome, an effect: it is never removed or reordered with another.
def QNova_MeasureOp : QNova_Op<"measure", [LinearQubits, MemoryEffects<[MemWrite]>]> {
    let summary = "measure a qubit in the computational basis";
    let arguments = (ins QNova_QubitType:$in);
    let results = (outs I1:$bit, QNova_QubitType:$out);
    let builders = [OpBuilder<(ins "::mlir::Value":$in), [{
        build($_builder, $_state, $_builder.getI1Type(), ::qnova::qubit($_builder), in);
    }]>];
    let assemblyFormat = "$in attr-dict";
}

// qnova.probs %a, %b : the probability of each basis state of these qubits, printed (|01>: 0.5000), the qubits'
// last use. A program's output, as print is.
def QNova_ProbsOp : QNova_Op<"probs", [MemoryEffects<[MemWrite]>]> {
    let summary = "print the probabilities of the qubits' basis states";
    let arguments = (ins Variadic<QNova_QubitType>:$qubits);
    let assemblyFormat = "$qubits attr-dict";
    let hasVerifier = 1;  // up to 12 qubits
}

// ─── Logical gates ──────────────────────────────────────────────────────────

class QNova_OneQubitGate<string mnemonic, string what, list<Trait> traits = []> : QNova_Gate<mnemonic, traits> {
    let summary = what;
    let arguments = (ins QNova_QubitType:$in);
    let results = (outs QNova_QubitType:$out);
    let builders = [OpBuilder<(ins "::mlir::Value":$in), [{
        build($_builder, $_state, ::qnova::qubit($_builder), in);
    }]>];
    let assemblyFormat = "$in attr-dict";
}

// h, x, y and z are their own inverses: canonicalize cancels two in a row (h(h(q)) is q)
let hasCanonicalizeMethod = 1 in {
def QNova_HOp : QNova_OneQubitGate<"h", "Hadamard">;
def QNova_XOp : QNova_OneQubitGate<"x", "Pauli X (NOT)">;
def QNova_YOp : QNova_OneQubitGate<"y", "Pauli Y">;
def QNova_ZOp : QNova_OneQubitGate<"z", "Pauli Z">;
}
def QNova_SOp : QNova_OneQubitGate<"s", "phase, Rz(pi/2) up to a global phase">;
def QNova_TOp : QNova_OneQubitGate<"t", "pi/8 gate, Rz(pi/4) up to a global phase">;

// A rotation by an angle in radians, an f64 value: a constant, or computed (a variational circuit's parameter)
class QNova_RotationGate<string mnemonic, string what> : QNova_Gate<mnemonic> {
    let summary = what;
    let arguments = (ins QNova_QubitType:$in, F64:$theta);
    let results = (outs QNova_QubitType:$out);
    let builders = [OpBuilder<(ins "::mlir::Value":$in, "::mlir::Value":$theta), [{
        build($_builder, $_state, ::qnova::qubit($_builder), in, theta);
    }]>];
    let assemblyFormat = "$in `,` $theta attr-dict";
}

def QNova_RxOp : QNova_RotationGate<"rx", "rotation about X">;
def QNova_RyOp : QNova_RotationGate<"ry", "rotation about Y">;

class QNova_TwoQubitGate<string mnemonic, string what> : QNova_Gate<mnemonic> {
    let summary = what;
    let arguments = (ins QNova_QubitType:$a, QNova_QubitType:$b);
    let results = (outs QNova_QubitType:$a_out, QNova_QubitType:$b_out);
    let builders = [OpBuilder<(ins "::mlir::Value":$a, "::mlir::Value":$b), [{
        build($_builder, $_state, ::qnova::qubit($_builder), ::qnova::qubit($_builder), a, b);
    }]>];
    let assemblyFormat = "$a `,` $b attr-dict";
    let hasCanonicalizeMethod = 1;  // two in a row on the same pair cancel
}

def QNova_CxOp : QNova_TwoQubitGate<"cx", "controlled NOT: a controls, b is flipped">;
def QNova_SwapOp : QNova_TwoQubitGate<"swap", "exchange two qubits' states">;

// cp: controlled phase, diag(1, 1, 1, e^(i lambda)); QFT is made of them
def QNova_CpOp : QNova_Gate<"cp"> {
    let summary = "controlled phase by lambda";
    let arguments = (ins QNova_QubitType:$a, QNova_QubitType:$b, F64:$lambda);
    let results = (outs QNova_QubitType:$a_out, QNova_QubitType:$b_out);
    let builders = [OpBuilder<(ins "::mlir::Value":$a, "::mlir::Value":$b, "::mlir::Value":$lambda), [{
        build($_builder, $_state, ::qnova::qubit($_builder), ::qnova::qubit($_builder), a, b, lambda);
    }]>];
    let assemblyFormat = "$a `,` $b `,` $lambda attr-dict";
    let hasCanonicalizeMethod = 1;  // two in a row on one pair merge; a full turn, or one on a fresh qubit, goes
}

// ─── Native gates of a neutral-atom machine ─────────────────────────────────

// %q1 = qnova.r %q0, %theta, %phi : rotation by theta about the axis cos(phi) X + sin(phi) Y, a Raman pulse. rx is
// r(theta, 0) and ry r(theta, pi/2). Canonicalize merges two in a row with the same phi and drops r(0, phi).
def QNova_ROp : QNova_Gate<"r"> {
    let summary = "rotation about an axis in the XY plane (native)";
    let arguments = (ins QNova_QubitType:$in, F64:$theta, F64:$phi);
    let results = (outs QNova_QubitType:$out);
    let builders = [OpBuilder<(ins "::mlir::Value":$in, "::mlir::Value":$theta, "::mlir::Value":$phi), [{
        build($_builder, $_state, ::qnova::qubit($_builder), in, theta, phi);
    }]>];
    let assemblyFormat = "$in `,` $theta `,` $phi attr-dict";
    let hasCanonicalizeMethod = 1;
}

// %q1 = qnova.rz %q0, %theta : rotation about Z (native, and logical). Canonicalize merges two in a row, drops
// rz(0), and moves an rz on a cz's output to before the cz (both are diagonal, so they commute), where it can merge
// with an rz before.
def QNova_RzOp : QNova_Gate<"rz"> {
    let summary = "rotation about Z";
    let arguments = (ins QNova_QubitType:$in, F64:$theta);
    let results = (outs QNova_QubitType:$out);
    let builders = [OpBuilder<(ins "::mlir::Value":$in, "::mlir::Value":$theta), [{
        build($_builder, $_state, ::qnova::qubit($_builder), in, theta);
    }]>];
    let assemblyFormat = "$in `,` $theta attr-dict";
    let hasCanonicalizeMethod = 1;
}

// %a1, %b1 = qnova.cz %a0, %b0 : controlled Z (native: two atoms within the Rydberg blockade radius). Symmetric, and
// its own inverse.
def QNova_CzOp : QNova_TwoQubitGate<"cz", "controlled Z">;

// %q1 = qnova.move %q0 {to = [x, y]} : the atom carried to site (x, y) of the array by an optical tweezer, so it
// can interact with another; its state is unchanged. --qnova-schedule adds them.
def QNova_MoveOp : QNova_Op<"move", [LinearQubits]> {
    let summary = "move an atom to another site (its state unchanged)";
    let arguments = (ins QNova_QubitType:$in, DenseI64ArrayAttr:$to);
    let results = (outs QNova_QubitType:$out);
    let builders = [OpBuilder<(ins "::mlir::Value":$in, "::mlir::DenseI64ArrayAttr":$to), [{
        build($_builder, $_state, ::qnova::qubit($_builder), in, to);
    }]>];
    let assemblyFormat = "$in `to` $to attr-dict";
}

#endif  // QNOVA_OPS
