// nova-translate: Nova's mlir-translate, between MLIR and other formats. So far OpenQASM 3, the format quantum
// toolchains exchange circuits in:
//   nova-translate --qnova-to-openqasm circuit.mlir     a qnova circuit as OpenQASM 3
//   nova-translate --openqasm-to-qnova circuit.qasm     an OpenQASM 3 (or 2) circuit as qnova
#include "QASM/OpenQASM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char** argv)
{
    qasm::registerTranslations();
    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "Nova's translations")) ? 1 : 0;
}
