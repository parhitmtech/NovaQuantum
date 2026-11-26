# Nova's C runtime, a static library a compiled Nova program links with: tensors and printing (nova_runtime.c), and
# the statevector simulator compiled circuits run on (nova_quantum.c; it needs libm)
add_library(nova_runtime STATIC nova_runtime.c nova_quantum.c)
set_target_properties(nova_runtime PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
