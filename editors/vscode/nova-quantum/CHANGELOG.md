# Changelog

All notable changes to the Nova Quantum extension. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).
Odd minor versions (0.1, 0.3) are pre-releases.

## [0.3.0] (pre-release)

### Added

- Each circuit's compiled numbers above it (native gates, depth, atom moves, time on the machine), updated as you
  type, for Nova programs with qubits and OpenQASM circuits.
- Commands: Run on Simulator, Run on Simulator as Written, Compile for Hardware and Run, Show the Compiled Circuit
  (as written, native gates, hardware instructions), Export as OpenQASM.
- OpenQASM 3 highlighting, and its errors as you type, from nova-compile.
- `qubit`, `qpu` and the gates highlighted in Nova files.
- Settings for the paths of nova-compile and nova-translate; Workspace Trust (no program runs in an untrusted
  folder).

### Changed

- Needs VS Code 1.91 or later.

## [0.1.0] (pre-release)

### Added

- `.qasm` files, and files whose first line is `OPENQASM`, are recognized as OpenQASM.
- Language configuration for OpenQASM: `//` and `/* */` comments, bracket matching and auto-closing, and indentation
  inside `{ }`.
- Depends on the Nova Compiler extension, which installs with it.
