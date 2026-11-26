// The passes to and on the atom dialect (lib/Atom)
#ifndef ATOM_ATOMPASSES_H
#define ATOM_ATOMPASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace atom
{
// --qnova-to-atom: a scheduled qnova circuit as timed hardware instructions, under a timing model (its options)
std::unique_ptr<mlir::Pass> createQNovaToAtomPass();

// --atom-timing: prints each program's instructions, layers and run time, by kind of instruction
std::unique_ptr<mlir::Pass> createTimingPass();

// Registers the passes with nova-opt's command line
void registerPasses();
}  // namespace atom

#endif  // ATOM_ATOMPASSES_H
