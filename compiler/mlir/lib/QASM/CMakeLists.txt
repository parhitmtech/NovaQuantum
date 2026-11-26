# OpenQASM 3 to and from the qnova dialect (Export.cpp, Import.cpp), and their nova-translate registrations
add_mlir_library(NovaQASM
    Export.cpp
    Import.cpp
    Translations.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QASM

    DEPENDS
    MLIRQNovaOpsIncGen

    LINK_LIBS PUBLIC
    MLIRQNova
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRTranslateLib
)
