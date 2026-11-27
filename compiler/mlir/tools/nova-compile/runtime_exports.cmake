# Writes a module-definition (.def) file exporting every global symbol the runtime library defines, so that on
# Windows nova-compile exports the runtime's functions (and only them) for the JIT to find in the process.
# Usage: cmake -DNM=<nm> -DARCHIVE=<libnova_runtime.a> -DOUT=<file.def> -P runtime_exports.cmake
execute_process(COMMAND "${NM}" -g --defined-only "${ARCHIVE}"
                OUTPUT_VARIABLE symbols RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "${NM} could not read ${ARCHIVE}")
endif()
# nm's lines: "<address> <type> <name>"; T is code, D, B and R are data
string(REGEX MATCHALL "[0-9a-fA-F]+ [TDBR] [A-Za-z_][A-Za-z0-9_]*" lines "${symbols}")
set(names)
foreach(line IN LISTS lines)
  string(REGEX REPLACE "^[0-9a-fA-F]+ [TDBR] " "" name "${line}")
  list(APPEND names "${name}")
endforeach()
list(REMOVE_DUPLICATES names)
list(LENGTH names count)
if(count EQUAL 0)
  message(FATAL_ERROR "no symbols found in ${ARCHIVE}")
endif()
set(text "EXPORTS\n")
foreach(name IN LISTS names)
  string(APPEND text "    ${name}\n")
endforeach()
file(WRITE "${OUT}" "${text}")
message(STATUS "nova-compile exports the runtime's ${count} symbols")
