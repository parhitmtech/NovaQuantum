# nova-compile: a Nova program to a native executable, through MLIR and LLVM, or run at once (--run, LLVM's JIT). It
# runs LLVM's clang to compile the IR and link it with Nova's runtime; their paths are fixed when it is built.
add_llvm_executable(nova-compile nova-compile.cpp)
llvm_update_compile_flags(nova-compile)
target_link_libraries(nova-compile PRIVATE
    NovaLower
    NovaConversion
    MLIRQNova
    MLIRAtom
    NovaQASM
    MLIRParser
    MLIRPass
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
    # --run: the runtime inside this program, every function of it, exported so the JIT finds them in the process
    -Wl,--whole-archive nova_runtime -Wl,--no-whole-archive
)
# The runtime's functions exported from the executable, so the JIT finds them in the process: -rdynamic on Linux;
# on Windows (MinGW) the executable exports them as a DLL would
if(WIN32)
  target_link_options(nova-compile PRIVATE -Wl,--export-all-symbols)
else()
  target_link_options(nova-compile PRIVATE -rdynamic)
endif()
target_compile_definitions(nova-compile PRIVATE
    NOVA_CLANG="${LLVM_TOOLS_BINARY_DIR}/clang"
    NOVA_RUNTIME="${CMAKE_BINARY_DIR}/lib/libnova_runtime.a"
    NOVA_GPU_RUNTIME="${PROJECT_SOURCE_DIR}/runtime/gpu/libnova_runtime_gpu.a"
)
add_dependencies(nova-compile nova_runtime)
mlir_check_all_link_libraries(nova-compile)
