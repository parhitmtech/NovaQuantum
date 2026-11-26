// The atom dialect's instructions (atom::RamanOp, ...), as TableGen writes them from AtomOps.td
#ifndef ATOM_ATOMOPS_H
#define ATOM_ATOMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Atom/AtomDialect.h"

#define GET_OP_CLASSES
#include "Atom/AtomOps.h.inc"

#endif  // ATOM_ATOMOPS_H
