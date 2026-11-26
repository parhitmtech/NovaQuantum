// The nova dialect's ops (nova::MatmulOp, ...), as TableGen writes them from NovaOps.td
#ifndef NOVA_NOVAOPS_H
#define NOVA_NOVAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"  // canonicalize methods take a PatternRewriter
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Nova/NovaDialect.h"
#include "Nova/NovaTypes.h"

namespace nova
{
// The program's runtime errors, as a side-effect resource (NovaOps.td, Nova_RuntimeError): an op that can stop the
// program with one writes it, so passes keep the op even when nothing uses its result, and never move it to where
// it would not have run
struct RuntimeErrorResource : mlir::SideEffects::Resource::Base<RuntimeErrorResource>
{
    llvm::StringRef getName() final { return "nova.runtime_error"; }
};
}  // namespace nova

#define GET_OP_CLASSES
#include "Nova/NovaOps.h.inc"

#endif  // NOVA_NOVAOPS_H
