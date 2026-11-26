// The OpenQASM translations, as mlir-translate registers its own: nova-translate --qnova-to-openqasm and
// --openqasm-to-qnova
#include "QASM/OpenQASM.h"

#include "QNova/QNovaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

void qasm::registerTranslations()
{
    mlir::TranslateFromMLIRRegistration exporter(
        "qnova-to-openqasm", "Write a qnova circuit as OpenQASM 3",
        [](mlir::Operation* op, llvm::raw_ostream& os) {
            return exportOpenQASM(llvm::cast<mlir::ModuleOp>(op), os);
        },
        [](mlir::DialectRegistry& registry) {
            registry.insert<qnova::QNovaDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect>();
        });
    mlir::TranslateToMLIRRegistration importer(
        "openqasm-to-qnova", "Read an OpenQASM 3 (or 2) circuit as qnova",
        [](llvm::StringRef source, mlir::MLIRContext* context) -> mlir::OwningOpRef<mlir::Operation*> {
            return importOpenQASM(source, "<openqasm>", context);
        });
}
