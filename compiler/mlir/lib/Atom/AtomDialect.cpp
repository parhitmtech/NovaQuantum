// The atom dialect: TableGen's generated definitions, the registration of its instructions, and their verifier
#include "Atom/AtomDialect.h"

#include <cstdlib>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Atom/AtomOps.h"

using namespace mlir;
using namespace atom;

#include "Atom/AtomOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Atom/AtomOps.cpp.inc"

void AtomDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Atom/AtomOps.cpp.inc"
        >();
}

// A Rydberg pulse's pairs: four numbers each, two neighbouring sites (within the interaction radius)
LogicalResult RydbergOp::verify()
{
    ArrayRef<int64_t> p = getPairs();
    if (p.empty() || p.size() % 4 != 0) return emitOpError("needs pairs of sites, four numbers each");
    for (size_t k = 0; k < p.size(); k += 4)
    {
        if (std::max(std::abs(p[k] - p[k + 2]), std::abs(p[k + 1] - p[k + 3])) != 1)
            return emitOpError("pairs atoms that are not neighbours: (") << p[k] << ", " << p[k + 1] << ") and ("
                                                                         << p[k + 2] << ", " << p[k + 3] << ")";
    }
    return success();
}
