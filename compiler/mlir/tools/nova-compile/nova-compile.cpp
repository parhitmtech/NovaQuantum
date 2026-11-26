// nova-compile: a Nova program, or a circuit written in MLIR, to a native executable, through MLIR and LLVM, in one
// process.
//
//   nova-compile prog.nova -o prog              the executable (clang links it with Nova's C runtime)
//   nova-compile circuit.qasm -o circuit        an OpenQASM 3 (or 2) circuit, read as qnova, then as below
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
//   --run                                       compile in memory (LLVM's JIT) and run at once: no file written, no
//                                               clang (the IDE's sandbox runs programs this way)
//   --gpu                                       link with the GPU runtime (runtime/gpu: make there first), so the
//                                               circuits run on the GPU simulator; NOVA_GPU_RUNTIME and CUDA_HOME
//                                               override where it and CUDA are
//
// It is nova-lower | nova-opt --nova-to-llvm | mlir-translate --mlir-to-llvmir | clang, without the text in between:
// the module stays in memory, so every op keeps the Nova line it came from, which the runtime's error messages
// print. clang then optimizes (-O2 unless -O0 is given) and links.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "Nova/LowerTree.h"
#include "Nova/NovaDialect.h"
#include "Nova/Passes.h"
#include "QNova/QNovaDialect.h"
#include "QNova/QNovaPasses.h"
#include "QASM/OpenQASM.h"
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
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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
static cl::opt<bool> run("run", cl::desc("Compile in memory (JIT) and run, writing nothing"));
static cl::opt<bool> gpu("gpu", cl::desc("Link with the GPU runtime (runtime/gpu): circuits run on the GPU"));

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
    else if (llvm::StringRef(input).ends_with(".qasm"))  // an OpenQASM circuit, from another toolchain
    {
        auto buffer = llvm::MemoryBuffer::getFile(input);
        if (!buffer)
        {
            llvm::errs() << "Error: cannot open '" << input << "'\n";
            return 1;
        }
        module = qasm::importOpenQASM((*buffer)->getBuffer(), input, &context);
        if (!module) return 1;  // the importer has reported why
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
        native.addNestedPass<mlir::func::FuncOp>(atom::createQNovaToAtomPass(/*skipUnscheduled=*/true));
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

    // --run: compile the module to machine code in this process (LLVM's ORC JIT, through MLIR's ExecutionEngine) and
    // call its main. The runtime's functions are linked into this program and exported (-rdynamic), so the JIT finds
    // them in the process. Nothing is written to disk and no compiler or linker runs: the IDE's sandbox, a read-only
    // container whose /tmp cannot hold executables, can run it.
    if (run)
    {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::ExecutionEngineOptions options;
        options.transformer = mlir::makeOptimizingTransformer(noOptimize ? 0 : 2, 0, nullptr);
        auto engine = mlir::ExecutionEngine::create(*module, options);
        if (!engine)
        {
            llvm::errs() << "Error: compiling to machine code failed: " << llvm::toString(engine.takeError()) << "\n";
            return 1;
        }
        int32_t status = 0;
        llvm::SmallVector<void*> packed{&status};  // the packed wrapper's arguments: main has none, then its result
        if (llvm::Error e = (*engine)->invokePacked("main", packed))
        {
            llvm::errs() << "Error: running main failed: " << llvm::toString(std::move(e)) << "\n";
            return 1;
        }
        fflush(stdout);
        return status;
    }
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
    llvm::SmallVector<std::string> args{clang.str(), noOptimize ? "-O0" : "-O2", "-Wno-override-module", ll};
    if (gpu)  // the GPU runtime (runtime/gpu, built by its Makefile) and CUDA's runtime library
    {
        const char* runtime = std::getenv("NOVA_GPU_RUNTIME");
        const char* cuda = std::getenv("CUDA_HOME");
        args.push_back(runtime ? runtime : NOVA_GPU_RUNTIME);
        args.push_back(std::string("-L") + (cuda ? cuda : "/usr/local/cuda") + "/lib64");
        args.push_back("-lcudart");
        args.push_back("-lstdc++");
    }
    else
        args.push_back(NOVA_RUNTIME);
    for (const char* a : {"-lm", "-o"}) args.push_back(a);
    args.push_back(output);
    llvm::SmallVector<llvm::StringRef> argRefs(args.begin(), args.end());
    std::string message;
    int status = llvm::sys::ExecuteAndWait(clang, argRefs, std::nullopt, {}, 0, 0, &message);
    llvm::sys::fs::remove(ll);
    if (status != 0)
    {
        llvm::errs() << "Error: clang failed" << (message.empty() ? "" : ": " + message) << "\n";
        return 1;
    }
    return 0;
}
