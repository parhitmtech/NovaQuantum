// Lowering to MLIR (roadmap step C, docs/DESIGN.md C1): the checked tree becomes a module of nova, arith, scf and
// func ops in SSA form. It reads the same check-pass result as lower.cc, the interpreter's lowering, and like it
// keeps no tables by name: every name and type comes from the check pass.
#ifndef NOVA_LOWERTREE_H
#define NOVA_LOWERTREE_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

class ParseTree;     // the front end (compiler/include): parse_tree.h
struct CheckResult;  // check.h

namespace nova
{
// The program as a module, or null when lowering met something it does not cover yet: that is reported to the
// diagnostics engine (E0601), and its message is in 'error' too. 'file' names the source in the ops' locations.
mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(mlir::MLIRContext& context, const ParseTree& tree,
                                                const CheckResult& checked, const std::string& file,
                                                std::string& error);

// The program in the file at path, read, checked and lowered: the front end and lower_to_mlir, then MLIR's verifier.
// Null with the errors in 'error', one per line.
// A file's program as a verified module, or null: the program's errors are in the diagnostics engine
// (diagnostics.h), and 'error' says what failed when it was not the program (a file that cannot be read, a bug)
mlir::OwningOpRef<mlir::ModuleOp> lower_file(mlir::MLIRContext& context, const std::string& path, std::string& error);
}  // namespace nova

#endif  // NOVA_LOWERTREE_H
