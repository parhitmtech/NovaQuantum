// --qnova-decompose: logical gates as the native gates of a neutral-atom machine.
//
// The machine's native operations: r(theta, phi), a rotation about an axis in the XY plane (a Raman laser pulse);
// rz(theta), a rotation about Z (a phase shift); and cz, a controlled Z between two atoms close enough for the
// Rydberg blockade. Every logical gate is some sequence of them. Each identity below holds up to a global phase,
// which no measurement can observe:
//
//   x = r(pi, 0)          y = r(pi, pi/2)        rx(t) = r(t, 0)        ry(t) = r(t, pi/2)
//   z = rz(pi)            s = rz(pi/2)           t = rz(pi/4)
//   h = r(pi/2, pi/2) after rz(pi)                               (H is Ry(pi/2) Rz(pi))
//   cx(a, b) = h(b), then cz(a, b), then h(b)                    (CX is (I x H) CZ (I x H))
//   swap(a, b) = cx(a, b), cx(b, a), cx(a, b)
//   cp(a, b, l) = rz(b, l/2), cx(a, b), rz(b, -l/2), cx(a, b), rz(a, l/2)
//
// The rewrite is greedy: a cx a swap makes is rewritten in turn. What it leaves is often wasteful (h then h is two
// rz(pi) and two r(pi/2) that cancel), which canonicalize then cleans up; --qnova-to-native runs both.
// Utils/qbench.py checks every identity numerically: the logical and the native circuit give the same
// probabilities in the simulator.
#include "QNova/QNovaPasses.h"

#include <cmath>

#include "QNova/QNovaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace qnova;

namespace
{
Value angle(PatternRewriter& rewriter, Location loc, double radians)
{
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(radians));
}

// h as native gates: rz(pi), then r(pi/2, pi/2)
Value hadamard(PatternRewriter& rewriter, Location loc, Value q)
{
    Value z = rewriter.create<RzOp>(loc, q, angle(rewriter, loc, M_PI));
    return rewriter.create<ROp>(loc, z, angle(rewriter, loc, M_PI / 2), angle(rewriter, loc, M_PI / 2));
}

// A one-qubit gate that is one native r(theta, phi) or rz(theta) with constant angles
template <typename Op>
struct AsR : OpRewritePattern<Op>
{
    AsR(MLIRContext* context, double theta, double phi) : OpRewritePattern<Op>(context), theta(theta), phi(phi) {}
    double theta, phi;

    LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        rewriter.replaceOpWithNewOp<ROp>(op, op.getIn(), angle(rewriter, loc, theta), angle(rewriter, loc, phi));
        return success();
    }
};

template <typename Op>
struct AsRz : OpRewritePattern<Op>
{
    AsRz(MLIRContext* context, double theta) : OpRewritePattern<Op>(context), theta(theta) {}
    double theta;

    LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<RzOp>(op, op.getIn(), angle(rewriter, op.getLoc(), theta));
        return success();
    }
};

// rx(t) and ry(t): r(t, 0) and r(t, pi/2), t kept as the value it is
template <typename Op>
struct RotationAsR : OpRewritePattern<Op>
{
    RotationAsR(MLIRContext* context, double phi) : OpRewritePattern<Op>(context), phi(phi) {}
    double phi;

    LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<ROp>(op, op.getIn(), op.getTheta(), angle(rewriter, op.getLoc(), phi));
        return success();
    }
};

struct HLowering : OpRewritePattern<HOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(HOp op, PatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, hadamard(rewriter, op.getLoc(), op.getIn()));
        return success();
    }
};

struct CxLowering : OpRewritePattern<CxOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(CxOp op, PatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Value b = hadamard(rewriter, loc, op.getB());
        auto cz = rewriter.create<CzOp>(loc, op.getA(), b);
        rewriter.replaceOp(op, {cz.getAOut(), hadamard(rewriter, loc, cz.getBOut())});
        return success();
    }
};

struct SwapLowering : OpRewritePattern<SwapOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(SwapOp op, PatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        auto first = rewriter.create<CxOp>(loc, op.getA(), op.getB());
        auto second = rewriter.create<CxOp>(loc, first.getBOut(), first.getAOut());   // b controls a
        auto third = rewriter.create<CxOp>(loc, second.getBOut(), second.getAOut());  // a controls b again
        rewriter.replaceOp(op, {third.getAOut(), third.getBOut()});
        return success();
    }
};

struct CpLowering : OpRewritePattern<CpOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(CpOp op, PatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Value half = rewriter.create<arith::MulFOp>(loc, op.getLambda(), angle(rewriter, loc, 0.5));
        Value minusHalf = rewriter.create<arith::NegFOp>(loc, half);
        Value b = rewriter.create<RzOp>(loc, op.getB(), half);
        auto first = rewriter.create<CxOp>(loc, op.getA(), b);
        Value b2 = rewriter.create<RzOp>(loc, first.getBOut(), minusHalf);
        auto second = rewriter.create<CxOp>(loc, first.getAOut(), b2);
        Value a = rewriter.create<RzOp>(loc, second.getAOut(), half);
        rewriter.replaceOp(op, {a, second.getBOut()});
        return success();
    }
};

struct Decompose : PassWrapper<Decompose, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Decompose)

    StringRef getArgument() const override { return "qnova-decompose"; }
    StringRef getDescription() const override
    {
        return "Rewrite logical gates as a neutral-atom machine's native ones: r, rz and cz";
    }
    void getDependentDialects(DialectRegistry& registry) const override { registry.insert<arith::ArithDialect>(); }

    void runOnOperation() override
    {
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<HLowering, CxLowering, SwapLowering, CpLowering>(context);
        patterns.add<AsR<XOp>>(context, M_PI, 0.0);
        patterns.add<AsR<YOp>>(context, M_PI, M_PI / 2);
        patterns.add<AsRz<ZOp>>(context, M_PI);
        patterns.add<AsRz<SOp>>(context, M_PI / 2);
        patterns.add<AsRz<TOp>>(context, M_PI / 4);
        patterns.add<RotationAsR<RxOp>>(context, 0.0);
        patterns.add<RotationAsR<RyOp>>(context, M_PI / 2);
        // Only these patterns, not the ops' canonicalizations: canonicalize is its own step of --qnova-to-native,
        // so the stats between the two show what each does
        GreedyRewriteConfig config;
        config.fold = false;
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) signalPassFailure();

        // Every logical gate must be gone
        getOperation().walk([&](Operation* op) {
            if (isa<HOp, XOp, YOp, ZOp, SOp, TOp, RxOp, RyOp, CxOp, SwapOp, CpOp>(op))
            {
                op->emitError("not decomposed into native gates");
                signalPassFailure();
            }
        });
    }
};
}  // namespace

std::unique_ptr<Pass> qnova::createDecomposePass() { return std::make_unique<Decompose>(); }
