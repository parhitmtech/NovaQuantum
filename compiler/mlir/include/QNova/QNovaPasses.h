// The qnova dialect's passes (lib/QNova): compiling a logical circuit for a neutral-atom machine
#ifndef QNOVA_QNOVAPASSES_H
#define QNOVA_QNOVAPASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace qnova
{
// --qnova-decompose: each logical gate (h, x, cx, cp, swap, ...) as native ones: r, rz and cz
std::unique_ptr<mlir::Pass> createDecomposePass();

// --qnova-schedule: places the atoms on a grid, moves an atom next to its partner before each cz that needs it, and
// gives each op the hardware layer it runs in ({layer = n}), under the machine's constraints
std::unique_ptr<mlir::Pass> createSchedulePass();

// --qnova-stats: prints each function's qubits, gates by kind, two-qubit gates, moves and depth
std::unique_ptr<mlir::Pass> createStatsPass();

// The --qnova-to-native pipeline: decompose, canonicalize (cancel and merge gates), schedule
void buildToNativePipeline(mlir::OpPassManager& pm);

// Registers the passes and the pipeline with nova-opt's command line
void registerPasses();
}  // namespace qnova

#endif  // QNOVA_QNOVAPASSES_H
