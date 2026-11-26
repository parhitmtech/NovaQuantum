// The qnova dialect's ops: TableGen's generated definitions, a verifier, and the canonicalizations.
//
// Because each qubit is an SSA value used once, a peephole rewrite needs no analysis: the gate before this one on
// the same qubit is simply the op that defines this gate's operand. h(h(q)) is q; rz(b, rz(a, q)) is rz(a + b, q).
// canonicalize applies these until none applies, and removes what they leave unused.
#include "QNova/QNovaOps.h"

#include <cmath>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "QNova/QNovaOps.cpp.inc"

using namespace mlir;
using namespace qnova;

LogicalResult ProbsOp::verify()
{
    if (getQubits().size() > 12) return emitOpError("prints up to 12 qubits, not ") << getQubits().size();
    return success();
}

namespace
{
// A gate that is its own inverse, applied twice in a row, is nothing: the second gives back the first's input
template <typename Op>
LogicalResult cancel_self_inverse(Op op, PatternRewriter& rewriter)
{
    auto before = op.getIn().template getDefiningOp<Op>();
    if (!before) return failure();
    rewriter.replaceOp(op, before.getIn());
    return success();  // before's result is now unused, and canonicalize removes it (it is Pure)
}

// The same for a two-qubit gate on the same pair, wires in the same order (or either order, if it is symmetric)
template <typename Op>
LogicalResult cancel_pair(Op op, PatternRewriter& rewriter, bool symmetric)
{
    auto before = op.getA().template getDefiningOp<Op>();
    if (!before || op.getB().getDefiningOp() != before.getOperation()) return failure();
    if (op.getA() == before.getAOut() && op.getB() == before.getBOut())
        rewriter.replaceOp(op, {before.getA(), before.getB()});
    else if (symmetric && op.getA() == before.getBOut() && op.getB() == before.getAOut())
        rewriter.replaceOp(op, {before.getB(), before.getA()});
    else
        return failure();
    return success();
}

// An angle that is a constant: its value
bool constant_angle(Value v, double& angle)
{
    FloatAttr attr;
    if (!matchPattern(v, m_Constant(&attr))) return false;
    angle = attr.getValueAsDouble();
    return true;
}

// A rotation by a multiple of 2 pi is the identity, up to a global phase no measurement can see
bool full_turns(Value v)
{
    double angle;
    if (!constant_angle(v, angle)) return false;
    double turns = angle / (2 * M_PI);
    return std::abs(turns - std::round(turns)) < 1e-12;
}

// Whether v is defined before op in op's block (or is a block argument), so a new op placed before op may use it
bool available_before(Value v, Operation* op)
{
    Operation* def = v.getDefiningOp();
    if (!def) return v.getParentBlock() == op->getBlock() || v.getParentRegion()->isAncestor(op->getParentRegion());
    return def->getBlock() == op->getBlock() && def->isBeforeInBlock(op);
}
}  // namespace

LogicalResult HOp::canonicalize(HOp op, PatternRewriter& rewriter) { return cancel_self_inverse(op, rewriter); }
LogicalResult XOp::canonicalize(XOp op, PatternRewriter& rewriter) { return cancel_self_inverse(op, rewriter); }
LogicalResult YOp::canonicalize(YOp op, PatternRewriter& rewriter) { return cancel_self_inverse(op, rewriter); }
LogicalResult ZOp::canonicalize(ZOp op, PatternRewriter& rewriter) { return cancel_self_inverse(op, rewriter); }

LogicalResult CxOp::canonicalize(CxOp op, PatternRewriter& rewriter) { return cancel_pair(op, rewriter, false); }
LogicalResult SwapOp::canonicalize(SwapOp op, PatternRewriter& rewriter) { return cancel_pair(op, rewriter, true); }
// Two facts about diagonal gates (rz, cz), whose matrices only multiply each basis state by a phase:
//   - on a qubit just allocated, in |0>, one changes nothing but a global phase: rz|0> is e^(-i theta/2)|0>, and a
//     cz with one atom in |0> is the identity;
//   - just before qnova.probs, one changes no probability: each basis state keeps its magnitude.
// So both are dropped there.
static bool fresh(Value q) { return q.getDefiningOp<AllocOp>() != nullptr; }

static bool only_probed(Operation* op)
{
    for (Value r : op->getResults())
        for (Operation* user : r.getUsers())
            if (!isa<ProbsOp>(user)) return false;
    return true;
}

LogicalResult CzOp::canonicalize(CzOp op, PatternRewriter& rewriter)
{
    if (fresh(op.getA()) || fresh(op.getB()) || only_probed(op))
    {
        rewriter.replaceOp(op, {op.getA(), op.getB()});
        return success();
    }
    return cancel_pair(op, rewriter, true);
}

// rz: drop a full turn, or one on a fresh qubit or before probs; merge with an rz just before (the angles add); move
// to before a cz, through which it commutes (both are diagonal), so that it can meet an rz there
LogicalResult RzOp::canonicalize(RzOp op, PatternRewriter& rewriter)
{
    if (full_turns(op.getTheta()) || fresh(op.getIn()) || only_probed(op))
    {
        rewriter.replaceOp(op, op.getIn());
        return success();
    }
    if (auto before = op.getIn().getDefiningOp<RzOp>())
    {
        Value sum = rewriter.create<arith::AddFOp>(op.getLoc(), before.getTheta(), op.getTheta());
        rewriter.replaceOpWithNewOp<RzOp>(op, before.getIn(), sum);
        return success();
    }
    if (auto cz = op.getIn().getDefiningOp<CzOp>())
    {
        if (!available_before(op.getTheta(), cz)) return failure();
        bool first = op.getIn() == cz.getAOut();  // which of the cz's qubits this rz is on
        rewriter.setInsertionPoint(cz);
        Value moved = rewriter.create<RzOp>(op.getLoc(), first ? cz.getA() : cz.getB(), op.getTheta());
        auto swapped = rewriter.create<CzOp>(cz.getLoc(), first ? moved : cz.getA(), first ? cz.getB() : moved);
        rewriter.replaceOp(op, first ? swapped.getAOut() : swapped.getBOut());
        rewriter.replaceOp(cz, swapped.getResults());
        return success();
    }
    return failure();
}

// r: drop a full turn; merge with an r just before about the same axis (the same phi value)
LogicalResult ROp::canonicalize(ROp op, PatternRewriter& rewriter)
{
    if (full_turns(op.getTheta()))
    {
        rewriter.replaceOp(op, op.getIn());
        return success();
    }
    auto before = op.getIn().getDefiningOp<ROp>();
    if (!before) return failure();
    double phiBefore, phi;
    bool sameAxis = before.getPhi() == op.getPhi() ||
                    (constant_angle(before.getPhi(), phiBefore) && constant_angle(op.getPhi(), phi) &&
                     std::abs(phiBefore - phi) < 1e-12);
    if (!sameAxis) return failure();
    Value sum = rewriter.create<arith::AddFOp>(op.getLoc(), before.getTheta(), op.getTheta());
    rewriter.replaceOpWithNewOp<ROp>(op, before.getIn(), sum, before.getPhi());
    return success();
}

// cp is diagonal too: drop a full turn, one with a qubit in |0>, or one before probs. Two on the same pair, in
// either order (cp is symmetric), merge: their phases add.
LogicalResult CpOp::canonicalize(CpOp op, PatternRewriter& rewriter)
{
    if (full_turns(op.getLambda()) || fresh(op.getA()) || fresh(op.getB()) || only_probed(op))
    {
        rewriter.replaceOp(op, {op.getA(), op.getB()});
        return success();
    }
    auto before = op.getA().getDefiningOp<CpOp>();
    if (!before || op.getB().getDefiningOp() != before.getOperation()) return failure();
    bool same = op.getA() == before.getAOut(), swapped = op.getA() == before.getBOut();
    if (!same && !swapped) return failure();
    Value sum = rewriter.create<arith::AddFOp>(op.getLoc(), before.getLambda(), op.getLambda());
    rewriter.replaceOpWithNewOp<CpOp>(op, same ? before.getA() : before.getB(), same ? before.getB() : before.getA(),
                                      sum);
    return success();
}
