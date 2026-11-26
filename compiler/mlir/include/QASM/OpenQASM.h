// OpenQASM 3, the text format quantum toolchains exchange circuits in (Qiskit, CUDA-Q, hardware providers), to and
// from the qnova dialect (lib/QASM, docs/QUANTUM.md)
#ifndef QASM_OPENQASM_H
#define QASM_OPENQASM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace qasm
{
// The circuit in m (its function with qubits) as OpenQASM 3. Fails, with an error on the op, for what this export
// does not write yet: control flow, a measurement whose bit the program uses, an angle that is not a constant.
mlir::LogicalResult exportOpenQASM(mlir::ModuleOp m, llvm::raw_ostream& os);

// A circuit in OpenQASM 3 (or 2) as a qnova module: func.func @main with the circuit, then qnova.probs on its qubits.
// Null, with an error at the line, for what it does not read: gate definitions, control flow, modifiers.
mlir::OwningOpRef<mlir::ModuleOp> importOpenQASM(llvm::StringRef source, llvm::StringRef file,
                                                 mlir::MLIRContext* context);

// --qnova-to-openqasm and --openqasm-to-qnova, for nova-translate
void registerTranslations();
}  // namespace qasm

#endif  // QASM_OPENQASM_H
