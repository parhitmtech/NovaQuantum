// Reading the quantum tools' output, in plain Node (no VS Code)
import * as assert from 'assert';
import { describe as describeStats, formatTime, isQuantumNova, lineOfFunction, parseErrors, parseStats } from '../../src/tools';

describe('parseStats', () => {
  it("reads a circuit's numbers and its time on the machine", () => {
    const stats = parseStats(
      'qnova-stats @main: 3 qubits, 15 gates (cz 2, r 8, rz 5), 2 two-qubit, 1 moves (1.000 sites), depth 7\n' +
      'atom-timing @main: 7 layers, 14 instructions, 1027.8 us (raman 2.0 us, rydberg 0.8 us, move 25.0 us, readout 1000.0 us)\n');
    const s = stats.get('main');
    assert.ok(s);
    assert.deepStrictEqual(
      [s.qubits, s.gates, s.byKind, s.twoQubit, s.moves, s.depth, s.layers, s.instructions, s.timeUs],
      [3, 15, 'cz 2, r 8, rz 5', 2, 1, 7, 7, 14, 1027.8]);
  });

  it('reads several circuits, and one without timing (not scheduled)', () => {
    const stats = parseStats(
      'qnova-stats @main: 3 qubits, 15 gates (cz 2, r 8, rz 5), 2 two-qubit, 0 moves, depth 7\n' +
      'qnova-stats @bell: 2 qubits, 5 gates (cz 1, r 4), 1 two-qubit, 0 moves, depth 4\n');
    assert.strictEqual(stats.size, 2);
    assert.strictEqual(stats.get('main')?.timeUs, undefined);
    assert.strictEqual(stats.get('bell')?.gates, 5);
  });

  it('ignores everything else', () => {
    assert.strictEqual(parseStats('|0>: 0.2500\nsomething else\n').size, 0);
  });
});

describe('describe and formatTime', () => {
  it('says what a circuit costs, in a line', () => {
    assert.strictEqual(describeStats({ fn: 'main', qubits: 3, gates: 15, byKind: '', twoQubit: 2, moves: 1, depth: 7, timeUs: 1027.8 }),
                       '3 qubits · 15 native gates · depth 7 · 1 atom move · 1.03 ms on the machine');
    assert.strictEqual(describeStats({ fn: 'main', qubits: 1, gates: 2, byKind: '', twoQubit: 0, moves: 0, depth: 2 }),
                       '1 qubit · 2 native gates · depth 2');
    assert.strictEqual(formatTime(80), '80.0 µs');
  });
});

describe('parseErrors', () => {
  it("reads MLIR's and Nova's errors", () => {
    const errors = parseErrors(
      "/tmp/c.qasm:4:1: error: the gate 'foo' is not read yet\nfoo q[1];\n^\n" +
      "prog.nova:3:11: error[E0406]: undeclared variable 'y'\n" +
      'C:\\work\\c.qasm:2:5: warning: something\n');
    assert.deepStrictEqual(errors.map(e => [e.file, e.line, e.column, e.severity, e.code ?? '']), [
      ['/tmp/c.qasm', 4, 1, 'error', ''],
      ['prog.nova', 3, 11, 'error', 'E0406'],
      ['C:\\work\\c.qasm', 2, 5, 'warning', ''],
    ]);
    assert.strictEqual(errors[0].message, "the gate 'foo' is not read yet");
  });
});

describe('lineOfFunction and isQuantumNova', () => {
  const nova = 'def bell(int k):\n    pass\n\nignite:\n    qubit a = qpu.alloc()\n';
  it("finds main (the ignite block) and a def", () => {
    assert.strictEqual(lineOfFunction(nova, 'main', false), 3);
    assert.strictEqual(lineOfFunction(nova, 'bell', false), 0);
    assert.strictEqual(lineOfFunction(nova, 'nowhere', false), -1);
  });
  it("puts an OpenQASM circuit's numbers on its OPENQASM line", () => {
    assert.strictEqual(lineOfFunction('// a comment\nOPENQASM 3.0;\nqubit q;\n', 'main', true), 1);
  });
  it('tells a quantum Nova program from another', () => {
    assert.ok(isQuantumNova(nova));
    assert.ok(!isQuantumNova('ignite:\n    print(1)\n'));
  });
});
