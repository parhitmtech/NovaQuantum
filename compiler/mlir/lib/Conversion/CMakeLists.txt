# Lowering the nova dialect toward LLVM (NovaToRuntime.cpp): the --convert-nova-to-runtime pass and the
# --nova-to-llvm pipeline
add_mlir_library(NovaConversion
    NovaToRuntime.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Nova

    DEPENDS
    MLIRNovaOpsIncGen

    LINK_LIBS PUBLIC
    MLIRNova
    MLIRQNova
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRUBDialect
    MLIRLLVMDialect
    MLIRFuncTransforms
    MLIRSCFTransforms
    MLIRTransformUtils
    MLIRTransforms
    MLIRPass
    # MLIR's own conversions, which --nova-to-llvm runs after Nova's
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRUBToLLVM
    MLIRReconcileUnrealizedCasts
)
