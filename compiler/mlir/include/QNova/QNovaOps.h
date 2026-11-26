// The qnova dialect's type (qnova::QubitType) and ops (qnova::HOp, ...), as TableGen writes them from QNovaOps.td
#ifndef QNOVA_QNOVAOPS_H
#define QNOVA_QNOVAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "QNova/QNovaDialect.h"

#define GET_TYPEDEF_CLASSES
#include "QNova/QNovaOpsTypes.h.inc"

namespace qnova
{
// The qubit type, for the ops' builders (QNovaOps.td), which give their results this type
inline mlir::Type qubit(mlir::Builder& b) { return b.getType<QubitType>(); }
}  // namespace qnova

namespace mlir::OpTrait
{
// LinearQubits (QNovaOps.td): each qubit an op gives is used at most once. A quantum state cannot be copied, so a
// second use of a value would be a second gate on a state the first already consumed.
template <typename ConcreteType>
class LinearQubits : public TraitBase<ConcreteType, LinearQubits>
{
    public:
        static LogicalResult verifyTrait(Operation* op)
        {
            for (Value result : op->getResults())
            {
                if (!llvm::isa<qnova::QubitType>(result.getType())) continue;
                if (!result.use_empty() && !result.hasOneUse())
                    return op->emitOpError("gives a qubit that is used more than once: a quantum state cannot be "
                                           "copied (no-cloning)");
            }
            return success();
        }
};
}  // namespace mlir::OpTrait

#define GET_OP_CLASSES
#include "QNova/QNovaOps.h.inc"

#endif  // QNOVA_QNOVAOPS_H
