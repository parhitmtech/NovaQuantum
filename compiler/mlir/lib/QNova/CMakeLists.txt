# The qnova dialect and its passes, MLIRQNova: decompose, schedule and stats (docs/QUANTUM.md)
add_mlir_dialect_library(MLIRQNova
    QNovaDialect.cpp
    QNovaOps.cpp
    Decompose.cpp
    Schedule.cpp
    Stats.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QNova

    DEPENDS
    MLIRQNovaOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIRTransformUtils
)
