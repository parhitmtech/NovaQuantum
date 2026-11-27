// Nova Quantum: quantum programs in Nova and OpenQASM 3, through Nova's quantum compiler (nova-compile).
//
// - OpenQASM files get their errors as you type (nova-compile reads the circuit), and both kinds of program show
//   their compiled numbers above each circuit (CodeLens): native gates, depth, atom moves, time on the machine.
// - Commands run a program on the simulator, compile it for the neutral-atom machine and run it, show the circuit
//   at each stage of compilation, and export a Nova circuit as OpenQASM.
// The Nova extension (a dependency) provides the Nova language itself.
import { execFile } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import { CircuitStats, describe, isQuantumNova, lineOfFunction, parseErrors, parseStats } from './tools';

const exe = (name: string) => (process.platform === 'win32' ? `${name}.exe` : name);
const CIRCUIT_SCHEME = 'nova-circuit';

let output: vscode.LogOutputChannel;
let diagnostics: vscode.DiagnosticCollection;

// A tool: the path in the settings, else the one shipped in the extension (bin/), else the one on PATH
function findTool(context: vscode.ExtensionContext, setting: string, name: string): string | undefined {
  const configured = vscode.workspace.getConfiguration('novaQuantum').get<string>(setting, '').trim();
  if (configured) return fs.existsSync(configured) ? configured : undefined;
  const bundled = context.asAbsolutePath(path.join('bin', exe(name)));
  if (fs.existsSync(bundled)) {
    // A package unzipped without its file modes leaves the program not executable: mark it again
    if (process.platform !== 'win32') {
      try { fs.accessSync(bundled, fs.constants.X_OK); } catch { try { fs.chmodSync(bundled, 0o755); } catch { /* reported when it fails */ } }
    }
    return bundled;
  }
  for (const dir of (process.env.PATH ?? '').split(path.delimiter)) {
    if (dir && fs.existsSync(path.join(dir, exe(name)))) return path.join(dir, exe(name));
  }
  return undefined;
}

const isQuantum = (doc: vscode.TextDocument) =>
  doc.languageId === 'openqasm' || (doc.languageId === 'nova' && isQuantumNova(doc.getText()));

interface ToolResult { code: number; stdout: string; stderr: string; }

function runTool(tool: string, args: string[], timeoutMs = 30000): Promise<ToolResult> {
  return new Promise(resolve => {
    execFile(tool, args, { timeout: timeoutMs, maxBuffer: 64 * 1024 * 1024, windowsHide: true }, (error, stdout, stderr) => {
      const code = error ? (typeof error.code === 'number' ? error.code : 1) : 0;
      resolve({ code, stdout: String(stdout), stderr: String(stderr) });
    });
  });
}

// The document as a file the tools can read: the file itself when it is saved, else a copy of its text in a
// temporary folder (with the same name, so errors name it). cleanup() removes the copy.
function asFile(doc: vscode.TextDocument): { file: string; cleanup: () => void } {
  if (!doc.isUntitled && !doc.isDirty && doc.uri.scheme === 'file') return { file: doc.uri.fsPath, cleanup: () => {} };
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-quantum-'));
  const name = doc.isUntitled ? (doc.languageId === 'openqasm' ? 'circuit.qasm' : 'program.nova') : path.basename(doc.uri.fsPath);
  const file = path.join(dir, name);
  fs.writeFileSync(file, doc.getText());
  return { file, cleanup: () => fs.rmSync(dir, { recursive: true, force: true }) };
}

// ─────────────────────────────────────────────────────────────────────────────
// The compiled numbers (CodeLens) and OpenQASM's errors
// ─────────────────────────────────────────────────────────────────────────────

interface Compiled { version: number; stats: Map<string, CircuitStats>; failed: boolean; }

class CircuitLens implements vscode.CodeLensProvider {
  private compiled = new Map<string, Compiled>();
  private pending = new Map<string, NodeJS.Timeout>();
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.changed.event;
  private warnedMissing = false;

  constructor(private context: vscode.ExtensionContext) {}

  // Compiles the document for its numbers (and, for OpenQASM, its errors), after the typing stops
  schedule(doc: vscode.TextDocument, delayMs = 400) {
    if (!isQuantum(doc)) return;
    const key = doc.uri.toString();
    clearTimeout(this.pending.get(key));
    this.pending.set(key, setTimeout(() => { this.pending.delete(key); void this.compile(doc); }, delayMs));
  }

  forget(doc: vscode.TextDocument) {
    this.compiled.delete(doc.uri.toString());
    diagnostics.delete(doc.uri);
  }

  async compile(doc: vscode.TextDocument): Promise<void> {
    const tool = findTool(this.context, 'compiler.path', 'nova-compile');
    if (!tool) {
      if (!this.warnedMissing) {
        this.warnedMissing = true;
        output.warn('nova-compile was not found: set novaQuantum.compiler.path');
      }
      return;
    }
    const version = doc.version;
    const { file, cleanup } = asFile(doc);
    let result: ToolResult;
    try {
      result = await runTool(tool, [file, '--hardware', '--qnova-stats', '--emit=mlir']);
    } finally {
      cleanup();
    }
    if (doc.version !== version) return;  // typed since: the next compile answers
    const stats = parseStats(result.stderr);
    this.compiled.set(doc.uri.toString(), { version, stats, failed: result.code !== 0 });
    // Nova's own errors come from the Nova extension's language server; OpenQASM's are read here
    if (doc.languageId === 'openqasm') {
      const found = parseErrors(result.stderr).filter(e => path.basename(e.file) === path.basename(file));
      diagnostics.set(doc.uri, found.map(e => {
        const line = Math.max(0, Math.min(e.line - 1, doc.lineCount - 1));
        const text = doc.lineAt(line);
        const range = new vscode.Range(line, Math.max(0, e.column - 1), line, Math.max(text.range.end.character, e.column));
        const d = new vscode.Diagnostic(range, e.message, e.severity === 'error' ? vscode.DiagnosticSeverity.Error
                                                                                 : vscode.DiagnosticSeverity.Warning);
        d.source = 'nova-compile';
        return d;
      }));
    }
    this.changed.fire();
  }

  provideCodeLenses(doc: vscode.TextDocument): vscode.CodeLens[] {
    if (!isQuantum(doc) || !vscode.workspace.getConfiguration('novaQuantum').get<boolean>('codeLens.enabled', true)) return [];
    const openqasm = doc.languageId === 'openqasm';
    const compiled = this.compiled.get(doc.uri.toString());
    if (!compiled || compiled.version !== doc.version) this.schedule(doc, 0);
    const lenses: vscode.CodeLens[] = [];
    const mainLine = lineOfFunction(doc.getText(), 'main', openqasm);
    const runs = (line: number) => {
      const at = new vscode.Range(line, 0, line, 0);
      lenses.push(new vscode.CodeLens(at, { title: '▷ Run on simulator', command: 'novaQuantum.runOnSimulator', arguments: [doc.uri] }));
      lenses.push(new vscode.CodeLens(at, { title: 'Compile for hardware', command: 'novaQuantum.compileForHardware', arguments: [doc.uri] }));
      lenses.push(new vscode.CodeLens(at, { title: 'Show circuit', command: 'novaQuantum.showCircuit', arguments: [doc.uri] }));
    };
    if (compiled && compiled.version === doc.version) {
      for (const s of compiled.stats.values()) {
        const line = lineOfFunction(doc.getText(), s.fn, openqasm);
        if (line < 0) continue;
        lenses.push(new vscode.CodeLens(new vscode.Range(line, 0, line, 0), {
          title: `⚛ ${describe(s)}`,
          tooltip: `As compiled for the neutral-atom machine: ${s.gates} native gates (${s.byKind}), ${s.twoQubit} two-qubit` +
                   (s.layers !== undefined ? `, ${s.layers} layers of ${s.instructions} timed instructions` : ''),
          command: 'novaQuantum.showCircuit',
          arguments: [doc.uri, 'hardware'],
        }));
      }
    }
    if (mainLine >= 0) runs(mainLine);
    return lenses;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Commands
// ─────────────────────────────────────────────────────────────────────────────

function quote(arg: string): string {
  if (process.platform === 'win32') return /[\s&()^]/.test(arg) ? `"${arg}"` : arg;
  return /[^A-Za-z0-9_./=:-]/.test(arg) ? `'${arg.replace(/'/g, "'\\''")}'` : arg;
}

// The document a command acts on: the one it was given (a CodeLens), else the active editor's
async function target(uri?: vscode.Uri): Promise<vscode.TextDocument | undefined> {
  const doc = uri ? await vscode.workspace.openTextDocument(uri) : vscode.window.activeTextEditor?.document;
  if (!doc || (doc.languageId !== 'nova' && doc.languageId !== 'openqasm')) {
    vscode.window.showInformationMessage('Open a Nova program with qubits, or an OpenQASM circuit (.qasm).');
    return undefined;
  }
  return doc;
}

function compilerOrComplain(context: vscode.ExtensionContext): string | undefined {
  const tool = findTool(context, 'compiler.path', 'nova-compile');
  if (!tool) {
    void vscode.window.showErrorMessage(
      "Nova's quantum compiler (nova-compile) was not found. Build it (see the extension's README) and set " +
        'novaQuantum.compiler.path.', 'Open Settings').then(choice => {
      if (choice) vscode.commands.executeCommand('workbench.action.openSettings', 'novaQuantum.compiler.path');
    });
  }
  return tool;
}

// Runs the program with nova-compile --run (and args) in the Nova Quantum terminal
async function runInTerminal(context: vscode.ExtensionContext, uri: vscode.Uri | undefined, args: string[]) {
  if (!vscode.workspace.isTrusted) {
    vscode.window.showWarningMessage('Nova Quantum runs programs in trusted workspaces only: trust this folder to run it.');
    return;
  }
  const doc = await target(uri);
  if (!doc) return;
  if (doc.isUntitled) {
    vscode.window.showInformationMessage('Save the program first (.nova or .qasm).');
    return;
  }
  if (doc.isDirty && !(await doc.save())) return;
  const tool = compilerOrComplain(context);
  if (!tool) return;
  const terminal = vscode.window.terminals.find(t => t.name === 'Nova Quantum') ?? vscode.window.createTerminal({ name: 'Nova Quantum' });
  terminal.show(true);
  const prefix = process.platform === 'win32' && vscode.env.shell.toLowerCase().includes('powershell') ? '& ' : '';
  terminal.sendText(`${prefix}${[tool, doc.uri.fsPath, '--run', ...args].map(quote).join(' ')}`);
}

// The circuit at a stage of compilation, as MLIR, in a read-only document
const STAGES: { [stage: string]: { label: string; detail: string; flags: string[] } } = {
  logical: { label: 'As written', detail: "The qnova circuit, before it is compiled (--logical)", flags: ['--logical'] },
  native: { label: 'Native gates', detail: "Decomposed into the machine's gates, optimized, routed and scheduled", flags: [] },
  hardware: { label: 'Hardware instructions', detail: "The atom dialect: the machine's timed instructions (--hardware)", flags: ['--hardware'] },
};

class CircuitDocuments implements vscode.TextDocumentContentProvider {
  private contents = new Map<string, string>();
  private changed = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.changed.event;
  set(uri: vscode.Uri, text: string) {
    this.contents.set(uri.toString(), text);
    this.changed.fire(uri);
  }
  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? '';
  }
}

async function showCircuit(context: vscode.ExtensionContext, circuits: CircuitDocuments, uri?: vscode.Uri, stage?: string) {
  const doc = await target(uri);
  if (!doc) return;
  if (!stage) {
    const pick = await vscode.window.showQuickPick(
      Object.entries(STAGES).map(([key, s]) => ({ label: s.label, detail: s.detail, key })),
      { title: 'Show the circuit', placeHolder: 'Which stage of compilation?' });
    if (!pick) return;
    stage = pick.key;
  }
  const tool = compilerOrComplain(context);
  if (!tool) return;
  const { file, cleanup } = asFile(doc);
  let result: ToolResult;
  try {
    result = await runTool(tool, [file, ...STAGES[stage].flags, '--qnova-stats', '--emit=mlir']);
  } finally {
    cleanup();
  }
  if (result.code !== 0) {
    output.error(result.stderr);
    vscode.window.showErrorMessage('The program did not compile: see its errors in the Problems view.');
    return;
  }
  const name = path.basename(doc.uri.fsPath || 'circuit').replace(/\.(nova|qasm)$/, '');
  const view = vscode.Uri.from({ scheme: CIRCUIT_SCHEME, path: `/${name}.${stage}.mlir` });
  const stats = result.stderr.split(/\r?\n/).filter(l => /^(qnova-stats|atom-timing) @/.test(l)).map(l => `// ${l}`).join('\n');
  circuits.set(view, `// ${STAGES[stage].label}: ${STAGES[stage].detail}\n${stats ? stats + '\n' : ''}\n${result.stdout}`);
  const shown = await vscode.workspace.openTextDocument(view);
  const languages = await vscode.languages.getLanguages();
  if (languages.includes('mlir')) await vscode.languages.setTextDocumentLanguage(shown, 'mlir');
  await vscode.window.showTextDocument(shown, { preview: true, viewColumn: vscode.ViewColumn.Beside });
}

async function exportOpenQASM(context: vscode.ExtensionContext, uri?: vscode.Uri) {
  const doc = await target(uri);
  if (!doc) return;
  if (doc.languageId === 'openqasm') {
    vscode.window.showInformationMessage('This is OpenQASM already.');
    return;
  }
  const compiler = compilerOrComplain(context);
  const translate = findTool(context, 'translate.path', 'nova-translate');
  if (!compiler) return;
  if (!translate) {
    vscode.window.showErrorMessage("nova-translate was not found: set novaQuantum.translate.path.");
    return;
  }
  const { file, cleanup } = asFile(doc);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-quantum-'));
  try {
    const circuit = await runTool(compiler, [file, '--logical', '--emit=mlir']);
    if (circuit.code !== 0) {
      output.error(circuit.stderr);
      vscode.window.showErrorMessage('The program did not compile: see its errors in the Problems view.');
      return;
    }
    const mlir = path.join(dir, 'circuit.mlir');
    fs.writeFileSync(mlir, circuit.stdout);
    const qasm = await runTool(translate, ['--qnova-to-openqasm', mlir]);
    if (qasm.code !== 0) {
      output.error(qasm.stderr);
      vscode.window.showErrorMessage(`The circuit could not be written as OpenQASM: ${qasm.stderr.split('\n')[0]}`);
      return;
    }
    const shown = await vscode.workspace.openTextDocument({ language: 'openqasm', content: qasm.stdout });
    await vscode.window.showTextDocument(shown, { viewColumn: vscode.ViewColumn.Beside });
  } finally {
    cleanup();
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

export function activate(context: vscode.ExtensionContext) {
  output = vscode.window.createOutputChannel('Nova Quantum', { log: true });
  diagnostics = vscode.languages.createDiagnosticCollection('nova-quantum');
  const lens = new CircuitLens(context);
  const circuits = new CircuitDocuments();
  context.subscriptions.push(
    output,
    diagnostics,
    vscode.languages.registerCodeLensProvider([{ language: 'nova' }, { language: 'openqasm' }], lens),
    vscode.workspace.registerTextDocumentContentProvider(CIRCUIT_SCHEME, circuits),
    vscode.workspace.onDidOpenTextDocument(doc => lens.schedule(doc, 0)),
    vscode.workspace.onDidChangeTextDocument(e => lens.schedule(e.document)),
    vscode.workspace.onDidCloseTextDocument(doc => lens.forget(doc)),
    vscode.commands.registerCommand('novaQuantum.runOnSimulator', (uri?: vscode.Uri) => runInTerminal(context, uri, ['--qnova-stats'])),
    vscode.commands.registerCommand('novaQuantum.runLogical', (uri?: vscode.Uri) => runInTerminal(context, uri, ['--logical'])),
    vscode.commands.registerCommand('novaQuantum.compileForHardware',
                                    (uri?: vscode.Uri) => runInTerminal(context, uri, ['--hardware', '--qnova-stats'])),
    vscode.commands.registerCommand('novaQuantum.showCircuit',
                                    (uri?: vscode.Uri, stage?: string) => showCircuit(context, circuits, uri, stage)),
    vscode.commands.registerCommand('novaQuantum.exportOpenQASM', (uri?: vscode.Uri) => exportOpenQASM(context, uri)),
  );
  for (const doc of vscode.workspace.textDocuments) lens.schedule(doc, 0);
  return { compile: (doc: vscode.TextDocument) => lens.compile(doc) };  // for the extension's tests
}

export function deactivate() {}
