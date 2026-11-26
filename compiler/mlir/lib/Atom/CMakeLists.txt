# The atom dialect (timed hardware instructions) and the passes to and on it: --qnova-to-atom, --atom-timing
add_mlir_dialect_library(MLIRAtom
    AtomDialect.cpp
    QNovaToAtom.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Atom

    DEPENDS
    MLIRAtomOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRQNova
    MLIRFuncDialect
    MLIRPass
    MLIRSideEffectInterfaces
)
