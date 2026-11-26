// --qnova-stats: one line per function with a circuit, to stderr:
//
//   qnova-stats @main: 4 qubits, 21 gates (cz 3, r 9, rz 9), 3 two-qubit, 1 moves (1.414 sites), depth 12
//
// Depth is the scheduled layer count if --qnova-schedule ran, else the longest chain of gates (SSA def-use), which is
// the depth with unlimited parallelism. utils/qbench.py reads these lines.
// This file also registers the passes and the --qnova-to-native pipeline.
#include "QNova/QNovaPasses.h"

#include <map>

#include "QNova/QNovaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace qnova;

namespace
{
bool is_gate(Operation* op)
{
    return isa<HOp, XOp, YOp, ZOp, SOp, TOp, RxOp, RyOp, RzOp, ROp, CxOp, CzOp, CpOp, SwapOp>(op);
}

struct Stats : PassWrapper<Stats, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Stats)

    StringRef getArgument() const override { return "qnova-stats"; }
    StringRef getDescription() const override { return "Print each circuit's qubits, gates, moves and depth"; }

    void runOnOperation() override
    {
        func::FuncOp f = getOperation();
        int64_t qubits = 0, gates = 0, twoQubit = 0, moves = 0, longest = 0;
        std::map<std::string, int64_t> byKind;
        llvm::DenseMap<Operation*, int64_t> depth;  // the longest chain of gates ending at each op
        f.walk([&](Operation* op) {
            if (isa<AllocOp>(op)) qubits++;
            if (isa<MoveOp>(op)) moves++;
            if (!isa<QNovaDialect>(op->getDialect())) return;
            int64_t d = 0;
            for (Value v : op->getOperands())
                if (Operation* def = v.getDefiningOp(); def && depth.count(def)) d = std::max(d, depth[def]);
            if (is_gate(op))
            {
                gates++;
                byKind[op->getName().stripDialect().str()]++;
                if (op->getNumResults() == 2) twoQubit++;
                d++;
            }
            depth[op] = d;
            longest = std::max(longest, d);
        });
        if (qubits == 0) return;
        if (auto scheduled = f->getAttrOfType<IntegerAttr>("qnova.depth")) longest = scheduled.getInt();

        llvm::errs() << "qnova-stats @" << f.getSymName() << ": " << qubits << " qubits, " << gates << " gates (";
        bool first = true;
        for (auto& [kind, count] : byKind)
        {
            llvm::errs() << (first ? "" : ", ") << kind << " " << count;
            first = false;
        }
        llvm::errs() << "), " << twoQubit << " two-qubit, " << moves << " moves";
        if (auto distance = f->getAttrOfType<FloatAttr>("qnova.move_distance"))
            llvm::errs() << " (" << llvm::format("%.3f", distance.getValueAsDouble()) << " sites)";
        llvm::errs() << ", depth " << longest << "\n";
    }
};
}  // namespace

std::unique_ptr<Pass> qnova::createStatsPass() { return std::make_unique<Stats>(); }

// canonicalize the logical circuit (h h, cx cx and cp cp cancel or merge), decompose, canonicalize again (the native
// gates decomposing leaves), then CSE and schedule
void qnova::buildToNativePipeline(OpPassManager& pm)
{
    pm.addPass(createCanonicalizerPass());  // first on the logical gates, where a cancellation is easiest to see
    pm.addPass(createDecomposePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(createSchedulePass());
}

void qnova::registerPasses()
{
    registerPass([] { return createStatsPass(); });
    registerPass([] { return createDecomposePass(); });
    registerPass([] { return createSchedulePass(); });
    PassPipelineRegistration<>("qnova-to-native",
                               "Compile a circuit for a neutral-atom machine: decompose, canonicalize, schedule",
                               buildToNativePipeline);
}
