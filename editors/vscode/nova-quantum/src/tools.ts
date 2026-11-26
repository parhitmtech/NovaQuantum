// What the extension reads from Nova's quantum tools (nova-compile, nova-translate): no VS Code here, so the unit
// tests run it in plain Node.

// One circuit's numbers, as nova-compile --qnova-stats prints them (stderr):
//   qnova-stats @main: 3 qubits, 15 gates (cz 2, r 8, rz 5), 2 two-qubit, 1 moves (1.000 sites), depth 7
//   atom-timing @main: 7 layers, 14 instructions, 1027.8 us (raman 2.0 us, rydberg 0.8 us, move 25.0 us, ...)
export interface CircuitStats {
  fn: string;            // the circuit's function: main is the ignite block
  qubits: number;
  gates: number;
  byKind: string;        // "cz 2, r 8, rz 5"
  twoQubit: number;
  moves: number;
  depth: number;
  layers?: number;       // with --hardware, when the circuit could be scheduled
  instructions?: number;
  timeUs?: number;
}

const STATS = /^qnova-stats @(\S+): (\d+) qubits, (\d+) gates \(([^)]*)\), (\d+) two-qubit, (\d+) moves(?: \([^)]*\))?, depth (\d+)\s*$/;
const TIMING = /^atom-timing @(\S+): (\d+) layers, (\d+) instructions, ([\d.]+) us\b/;

export function parseStats(stderr: string): Map<string, CircuitStats> {
  const out = new Map<string, CircuitStats>();
  for (const line of stderr.split(/\r?\n/)) {
    const s = STATS.exec(line);
    if (s) {
      out.set(s[1], { fn: s[1], qubits: +s[2], gates: +s[3], byKind: s[4], twoQubit: +s[5], moves: +s[6], depth: +s[7] });
      continue;
    }
    const t = TIMING.exec(line);
    if (t) {
      const prev = out.get(t[1]);
      if (prev) Object.assign(prev, { layers: +t[2], instructions: +t[3], timeUs: +t[4] });
    }
  }
  return out;
}

// "15 gates · 2 two-qubit · depth 7 · 1027.8 µs on the machine"
export function describe(s: CircuitStats): string {
  const parts = [`${s.qubits} qubit${s.qubits === 1 ? '' : 's'}`, `${s.gates} native gates`, `depth ${s.depth}`];
  if (s.moves > 0) parts.push(`${s.moves} atom move${s.moves === 1 ? '' : 's'}`);
  if (s.timeUs !== undefined) parts.push(`${formatTime(s.timeUs)} on the machine`);
  return parts.join(' · ');
}

export function formatTime(us: number): string {
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(1)} µs`;
}

// An error a tool printed: "file:line:col: error: message" (MLIR's, and Nova's with a code: "error[E0406]")
export interface ToolError {
  file: string;
  line: number;     // 1-based
  column: number;   // 1-based
  severity: 'error' | 'warning';
  code?: string;
  message: string;
}

const ERROR = /^(.+?):(\d+):(\d+): (error|warning)(?:\[([A-Z]\d{4})\])?: (.*)$/;

export function parseErrors(stderr: string): ToolError[] {
  const out: ToolError[] = [];
  for (const line of stderr.split(/\r?\n/)) {
    const m = ERROR.exec(line);
    if (!m) continue;
    out.push({ file: m[1], line: +m[2], column: +m[3], severity: m[4] as 'error' | 'warning',
               ...(m[5] ? { code: m[5] } : {}), message: m[6] });
  }
  return out;
}

// Where a circuit's function starts in its source, for the line its numbers are shown above: main is a .nova file's
// ignite block (or def main), and an OpenQASM file's first statement; any other is a def. -1: not found.
export function lineOfFunction(text: string, fn: string, openqasm: boolean): number {
  const lines = text.split(/\r?\n/);
  if (openqasm) {
    if (fn !== 'main') return -1;
    const k = lines.findIndex(l => /^\s*OPENQASM\b/.test(l));
    return k >= 0 ? k : 0;
  }
  const pattern = fn === 'main' ? /^(ignite\s*:|def\s+main\s*\()/ : new RegExp(`^\\s*def\\s+${fn.replace(/\W/g, '')}\\s*\\(`);
  return lines.findIndex(l => pattern.test(l));
}

// A Nova file is a quantum program when it has a qubit or calls a qpu gate
export function isQuantumNova(text: string): boolean {
  return /\bqpu\s*\.\s*\w+\s*\(/.test(text) || /^\s*qubit\s+\w+/m.test(text);
}
