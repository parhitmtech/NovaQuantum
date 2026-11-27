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
# The runtime's functions exported from the executable, so the JIT finds them in the process: -rdynamic on Linux.
# On Windows (MinGW) the executable exports them as a DLL would, through a .def file listing the runtime's symbols
# (read from the library at build time): exporting everything instead (--export-all-symbols) would also export
# LLVM's and MLIR's ~110,000 symbols, past the 65,535 a Windows executable can export.
if(WIN32)
  set(NOVA_RUNTIME_DEF ${CMAKE_CURRENT_BINARY_DIR}/nova_runtime_exports.def)
  add_custom_command(OUTPUT ${NOVA_RUNTIME_DEF}
      COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DARCHIVE=$<TARGET_FILE:nova_runtime> -DOUT=${NOVA_RUNTIME_DEF}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime_exports.cmake
      DEPENDS nova_runtime ${CMAKE_CURRENT_SOURCE_DIR}/runtime_exports.cmake
      COMMENT "The runtime's functions, exported from nova-compile for the JIT")
  add_custom_target(nova-compile-exports DEPENDS ${NOVA_RUNTIME_DEF})
  add_dependencies(nova-compile nova-compile-exports)
  target_link_options(nova-compile PRIVATE ${NOVA_RUNTIME_DEF})
  set_property(TARGET nova-compile APPEND PROPERTY LINK_DEPENDS ${NOVA_RUNTIME_DEF})
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
