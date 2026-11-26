// nova-compile: a Nova program to a native executable, through MLIR and LLVM, in one process.
//
//   nova-compile prog.nova -o prog              the executable (clang links it with Nova's C runtime)
//   nova-compile prog.nova --emit=mlir          the program as nova-lower writes it
//   nova-compile prog.nova --emit=llvm-dialect  after --nova-to-llvm
//   nova-compile prog.nova --emit=llvm-ir       the LLVM IR clang compiles
//
// It is nova-lower | nova-opt --nova-to-llvm | mlir-translate --mlir-to-llvmir | clang, without the text in between:
// the module stays in memory, so every op keeps the Nova line it came from, which the runtime's error messages
// print. clang then optimizes (-O2 unless -O0 is given) and links.
#include <string>

#include "Nova/LowerTree.h"
#include "Nova/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;

static cl::opt<std::string> input(cl::Positional, cl::desc("<file.nova>"), cl::Required);
static cl::opt<std::string> output("o", cl::desc("The executable to write"), cl::value_desc("file"), cl::init("a.out"));
static cl::opt<std::string> emit("emit", cl::desc("Stop early and print: mlir, llvm-dialect or llvm-ir"),
                                 cl::init(""));
static cl::opt<bool> noOptimize("O0", cl::desc("Have clang compile without optimizing"));

int main(int argc, char** argv)
{
    cl::ParseCommandLineOptions(argc, argv, "nova-compile: a Nova program to a native executable\n");

    mlir::MLIRContext context;
    std::string error;
    mlir::OwningOpRef<mlir::ModuleOp> module = nova::lower_file(context, input, error);
    if (!module)
    {
        llvm::errs() << error << "\n";
        return 1;
    }
    if (emit == "mlir")
    {
        module->print(llvm::outs());
        return 0;
    }

    // nova, arith, scf and func to the llvm dialect: the --nova-to-llvm pipeline, as nova-opt runs it. The pass
    // manager is anchored on the module, so the pipeline's passes run on this module, not on modules nested in it.
    mlir::PassManager pm(&context, mlir::ModuleOp::getOperationName());
    nova::buildNovaToLLVMPipeline(pm);
    if (mlir::failed(pm.run(*module)))
    {
        llvm::errs() << "Error: lowering to LLVM failed\n";
        return 1;
    }
    if (emit == "llvm-dialect")
    {
        module->print(llvm::outs());
        return 0;
    }

    // The llvm dialect to LLVM IR, as mlir-translate --mlir-to-llvmir does
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> ir = mlir::translateModuleToLLVMIR(*module, llvmContext, input);
    if (!ir)
    {
        llvm::errs() << "Error: translating to LLVM IR failed\n";
        return 1;
    }
    if (emit == "llvm-ir")
    {
        ir->print(llvm::outs(), nullptr);
        return 0;
    }
    if (!emit.empty())
    {
        llvm::errs() << "Error: --emit takes mlir, llvm-dialect or llvm-ir\n";
        return 1;
    }

    // clang compiles the IR and links it with the runtime. The IR goes to a temporary file next to the output.
    std::string ll = output + ".ll";
    {
        std::error_code ec;
        llvm::raw_fd_ostream file(ll, ec, llvm::sys::fs::OF_Text);
        if (ec)
        {
            llvm::errs() << "Error: cannot write '" << ll << "': " << ec.message() << "\n";
            return 1;
        }
        ir->print(file, nullptr);
    }
    llvm::StringRef clang = NOVA_CLANG;  // set by CMake: LLVM's clang, and the runtime library this build made
    llvm::SmallVector<llvm::StringRef> args{clang, noOptimize ? "-O0" : "-O2", "-Wno-override-module", ll,
                                            NOVA_RUNTIME, "-o", output};
    std::string message;
    int status = llvm::sys::ExecuteAndWait(clang, args, std::nullopt, {}, 0, 0, &message);
    llvm::sys::fs::remove(ll);
    if (status != 0)
    {
        llvm::errs() << "Error: clang failed" << (message.empty() ? "" : ": " + message) << "\n";
        return 1;
    }
    return 0;
}
