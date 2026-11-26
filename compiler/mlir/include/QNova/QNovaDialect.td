// The qnova dialect: quantum circuits, for Nova's compiler toward neutral-atom hardware (docs/QUANTUM.md)
#ifndef QNOVA_DIALECT
#define QNOVA_DIALECT

include "mlir/IR/OpBase.td"

def QNova_Dialect : Dialect {
    let name = "qnova";
    let summary = "Quantum circuits as SSA: qubits are values that each gate consumes and produces";
    let description = [{
        A qubit is an SSA value of type `!qnova.qubit`. A gate takes the qubits it acts on and gives new values for
        them: `%q1 = qnova.h %q0`. Each value is used at most once (no-cloning), so the def-use chains are the
        circuit's wires, and data dependence between gates is plain SSA dependence. Two levels of gates: the
        logical ones a program is written in (h, x, cx, ...) and the native ones of a neutral-atom machine (r, rz,
        cz), which --qnova-decompose rewrites the first into.
    }];
    let cppNamespace = "::qnova";
    let useDefaultTypePrinterParser = 1;
}

class QNova_Op<string mnemonic, list<Trait> traits = []> : Op<QNova_Dialect, mnemonic, traits>;

#endif  // QNOVA_DIALECT
