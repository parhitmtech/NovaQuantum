# Changelog

All notable changes to the Nova Quantum extension. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] (pre-release)

### Added

- `.qasm` files, and files whose first line is `OPENQASM`, are recognized as OpenQASM.
- Language configuration for OpenQASM: `//` and `/* */` comments, bracket matching and auto-closing, and indentation
  inside `{ }`.
- Depends on the Nova Programming Language extension, which installs with it.
