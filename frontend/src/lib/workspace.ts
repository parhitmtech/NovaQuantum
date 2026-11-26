export const EXAMPLES = [
  {
    name: 'hello.nova',
    title: 'Hello, Nova',
    description: 'Your first program',
    code: 'ignite:\n    print("Hello, Nova!")\n',
  },
  {
    name: 'functions.nova',
    title: 'Functions',
    description: 'Types, parameters, and return values',
    code: 'def add(int a, int b) -> int:\n    return a + b\n\nignite:\n    print(add(3, 4))\n',
  },
  {
    name: 'loops.nova',
    title: 'Loops',
    description: 'Add the numbers from 1 to 5',
    code: 'ignite:\n    int sum = 0\n    int i = 0\n    for (i = 1 ; i < 6 ; i += 1):\n        sum += i\n    print(sum)\n',
  },
  {
    name: 'input.nova',
    title: 'Standard input',
    description: 'Try input values 3 and 4',
    code: 'ignite:\n    int a = input()\n    int b = input()\n    print(a + b)\n    print(a * b)\n',
  },
  {
    name: 'teleport.nova',
    title: 'Quantum teleportation',
    description: 'Qubits and gates, compiled for a neutral-atom machine',
    code:
      '# The state of a moves to c, through an entangled pair and two measurements.\n' +
      '# Compiled through MLIR and LLVM and run on a statevector simulator.\n' +
      'ignite:\n' +
      '    qubit a = qpu.alloc()\n    qubit b = qpu.alloc()\n    qubit c = qpu.alloc()\n' +
      '    qpu.ry(a, 2.0943951023931953)   # |1> with probability 0.75\n' +
      '    qpu.h(b)\n    qpu.cx(b, c)\n    qpu.cx(a, b)\n    qpu.h(a)\n' +
      '    bool m1 = qpu.measure(a)\n    bool m2 = qpu.measure(b)\n' +
      '    if (m2):\n        qpu.x(c)\n    if (m1):\n        qpu.z(c)\n' +
      '    qpu.probs(c)                     # |1>: 0.7500, whatever was measured\n',
  },
  {
    name: 'ghz.qasm',
    title: 'OpenQASM circuit',
    description: 'A GHZ state, compiled down to hardware instructions',
    code:
      '// OpenQASM 3, as Qiskit or CUDA-Q write it: compiled to a neutral-atom machine\'s timed instructions\n' +
      'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nh q[0];\ncx q[0], q[1];\ncx q[1], q[2];\n',
  },
]
export const isBinaryFile = (name: string) => /\.(tar\.gz|gz|bin|pt|pth|nbc|zip)$/i.test(name)
export const validFileName = (name: string) =>
  /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(name) && !name.includes('..')
export function errorMessage(error: unknown, fallback: string) {
  return (
    (error as { response?: { data?: { error?: string } } })?.response?.data?.error ??
    (error instanceof Error ? error.message : fallback)
  )
}
export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
