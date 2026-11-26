// The qnova dialect's type (qnova::QubitType) and ops (qnova::HOp, ...), as TableGen writes them from QNovaOps.td
#ifndef QNOVA_QNOVAOPS_H
#define QNOVA_QNOVAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "QNova/QNovaDialect.h"

#define GET_TYPEDEF_CLASSES
#include "QNova/QNovaOpsTypes.h.inc"

namespace qnova
{
// The qubit type, for the ops' builders (QNovaOps.td), which give their results this type
inline mlir::Type qubit(mlir::Builder& b) { return b.getType<QubitType>(); }
}  // namespace qnova

namespace qnova
{
// Whether the qubit v is consumed at most once on every path through block: by at most one op of the block, and if
// that op holds regions (an scf.if's two branches), at most once in each of them, recursively. A loop may not
// consume a qubit from outside it at all: it would consume it once per iteration (lowering carries a qubit through
// a loop as a block argument instead).
inline bool linear_in(mlir::Value v, mlir::Block* block)
{
    mlir::Operation* consumer = nullptr;
    bool direct = false, nested = false;
    for (mlir::OpOperand& use : v.getUses())
    {
        mlir::Operation* user = block->findAncestorOpInBlock(*use.getOwner());
        if (!user) continue;  // not under this block
        if (consumer && consumer != user) return false;  // two ops of the block consume it
        consumer = user;
        if (use.getOwner() == user)
        {
            if (direct) return false;  // the same op takes it twice
            direct = true;
        }
        else nested = true;
    }
    if (!consumer || !nested) return true;
    if (direct || llvm::isa<mlir::LoopLikeOpInterface>(consumer)) return false;
    for (mlir::Region& region : consumer->getRegions())
        for (mlir::Block& b : region)
            if (!linear_in(v, &b)) return false;
    return true;
}
}  // namespace qnova

namespace mlir::OpTrait
{
// LinearQubits (QNovaOps.td): each qubit an op gives is consumed at most once on any path (qnova::linear_in). A
// quantum state cannot be copied, so a second use of a value would be a second gate on a state the first already
// consumed. The two branches of an if are two paths, so each may consume it.
template <typename ConcreteType>
class LinearQubits : public TraitBase<ConcreteType, LinearQubits>
{
    public:
        static LogicalResult verifyTrait(Operation* op)
        {
            for (Value result : op->getResults())
            {
                if (!llvm::isa<qnova::QubitType>(result.getType())) continue;
                if (!qnova::linear_in(result, result.getParentBlock()))
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
