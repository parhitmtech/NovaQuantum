// nova-lower: reads a Nova program, runs the front end (the scanner, the parser and the check pass, as the
// interpreter does), lowers the checked tree to MLIR (LowerTree.cpp) and prints the module. Its output is
// nova-opt's input: nova-lower prog.nova | nova-opt --canonicalize. MLIR's printing flags work here too:
// --mlir-print-debuginfo writes each op's Nova line, which a later tool reading the text then keeps.
#include <string>

#include "Nova/LowerTree.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::desc("<file.nova>"), llvm::cl::Required);

int main(int argc, char** argv)
{
    mlir::registerAsmPrinterCLOptions();  // --mlir-print-debuginfo and the other printing flags
    mlir::registerMLIRContextCLOptions();
    llvm::cl::ParseCommandLineOptions(argc, argv, "nova-lower: a Nova program as MLIR\n");

    mlir::MLIRContext context;
    std::string error;
    mlir::OwningOpRef<mlir::ModuleOp> module = nova::lower_file(context, input, error);
    if (!module)
    {
        llvm::errs() << error << "\n";
        return 1;
    }
    module->print(llvm::outs(), mlir::OpPrintingFlags());  // the flags from the command line apply
    return 0;
}
