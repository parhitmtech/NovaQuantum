// nova-compile: a Nova program, or a circuit written in MLIR, to a native executable, through MLIR and LLVM, in one
// process.
//
//   nova-compile prog.nova -o prog              the executable (clang links it with Nova's C runtime)
//   nova-compile circuit.mlir -o circuit        a qnova circuit: compiled for a neutral-atom machine (decomposed into
//                                               its native gates, optimized, routed and scheduled), then run on the
//                                               runtime's simulator; --logical runs it as written instead
//   nova-compile prog.nova --emit=mlir          the program as nova-lower writes it
//   nova-compile prog.nova --emit=llvm-dialect  after --nova-to-llvm
//   nova-compile prog.nova --emit=llvm-ir       the LLVM IR clang compiles
//   --qnova-stats                               print each circuit's gates, moves and depth as compiled
//   --hardware                                  compile a circuit down to the machine's timed instructions (the atom
//                                               dialect), which the simulator runs by site; with --qnova-stats, it
//                                               prints their run time too
//
// It is nova-lower | nova-opt --nova-to-llvm | mlir-translate --mlir-to-llvmir | clang, without the text in between:
// the module stays in memory, so every op keeps the Nova line it came from, which the runtime's error messages
// print. clang then optimizes (-O2 unless -O0 is given) and links.
#include <string>

#include "Nova/LowerTree.h"
#include "Nova/NovaDialect.h"
#include "Nova/Passes.h"
#include "QNova/QNovaDialect.h"
#include "QNova/QNovaPasses.h"
#include "Atom/AtomDialect.h"
#include "Atom/AtomPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Parser/Parser.h"
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

static cl::opt<std::string> input(cl::Positional, cl::desc("<file.nova or file.mlir>"), cl::Required);
static cl::opt<std::string> output("o", cl::desc("The executable to write"), cl::value_desc("file"), cl::init("a.out"));
static cl::opt<std::string> emit("emit", cl::desc("Stop early and print: mlir, llvm-dialect or llvm-ir"),
                                 cl::init(""));
static cl::opt<bool> noOptimize("O0", cl::desc("Have clang compile without optimizing"));
static cl::opt<bool> logical("logical", cl::desc("Run a circuit as written, without compiling it for the machine"));
static cl::opt<bool> stats("qnova-stats", cl::desc("Print each circuit's gates, moves and depth as compiled"));
static cl::opt<bool> hardware("hardware", cl::desc("Compile a circuit down to the machine's timed instructions"));

int main(int argc, char** argv)
{
    cl::ParseCommandLineOptions(argc, argv, "nova-compile: a Nova program to a native executable\n");

    // A .nova program goes through the front end; a .mlir file (a qnova circuit, or any nova program as MLIR) is
    // parsed as it is
    mlir::DialectRegistry registry;
    registry.insert<nova::NovaDialect, qnova::QNovaDialect, atom::AtomDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::ub::UBDialect>();
    mlir::MLIRContext context(registry);
    std::string error;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (llvm::StringRef(input).ends_with(".mlir"))
    {
        module = mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
        if (!module) return 1;  // the parser has reported why
    }
    else
    {
        module = nova::lower_file(context, input, error);
        if (!module)
        {
            llvm::errs() << error << "\n";
            return 1;
        }
    }

    // A circuit compiled for the machine: decomposed, optimized, routed and scheduled (--qnova-to-native)
    mlir::PassManager native(&context, mlir::ModuleOp::getOperationName());
    if (!logical) qnova::buildToNativePipeline(native);
    if (stats) native.addNestedPass<mlir::func::FuncOp>(qnova::createStatsPass());
    if (hardware && !logical)
    {
        native.addNestedPass<mlir::func::FuncOp>(atom::createQNovaToAtomPass());
        if (stats) native.addNestedPass<mlir::func::FuncOp>(atom::createTimingPass());
    }
    if (mlir::failed(native.run(*module)))
    {
        llvm::errs() << "Error: compiling the circuit failed\n";
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
    llvm::SmallVector<llvm::StringRef> args{clang,        noOptimize ? "-O0" : "-O2", "-Wno-override-module", ll,
                                            NOVA_RUNTIME, "-lm",  "-o",                       output};
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
