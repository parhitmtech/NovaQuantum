// The qnova dialect: TableGen's generated definitions, and the registration of its type and ops
#include "QNova/QNovaDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "QNova/QNovaOps.h"

using namespace mlir;
using namespace qnova;

#include "QNova/QNovaOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "QNova/QNovaOpsTypes.cpp.inc"

void QNovaDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "QNova/QNovaOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "QNova/QNovaOpsTypes.cpp.inc"
        >();
}
