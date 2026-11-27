// Nova Quantum in a real VS Code, with the real nova-compile and nova-translate
import * as assert from 'assert';
import * as path from 'path';
import * as vscode from 'vscode';

const fixtures = path.resolve(__dirname, '..', '..', '..', 'fixtures');
const exe = (name: string) => (process.platform === 'win32' ? `${name}.exe` : name);

async function until<T>(what: string, check: () => T | undefined | Promise<T | undefined>, ms = 30000): Promise<T> {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const value = await check();
    if (value) return value;
    await new Promise(r => setTimeout(r, 150));
  }
  throw new Error(`timed out waiting for ${what}`);
}

async function open(name: string): Promise<vscode.TextDocument> {
  const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(fixtures, name)));
  await vscode.window.showTextDocument(doc);
  return doc;
}

const lensTitles = async (doc: vscode.TextDocument) => {
  const lenses = await vscode.commands.executeCommand<vscode.CodeLens[]>('vscode.executeCodeLensProvider', doc.uri, 100);
  return (lenses ?? []).map(l => l.command?.title ?? '');
};

suite('Nova Quantum', () => {
  suiteSetup(async () => {
    const bin = process.env.NOVA_QUANTUM_BIN ?? '';
    const config = vscode.workspace.getConfiguration('novaQuantum');
    // A package under test (NOVA_TEST_BUNDLED) must find its own tools: the settings are cleared
    if (process.env.NOVA_TEST_BUNDLED) {
      for (const key of ['compiler.path', 'translate.path']) await config.update(key, undefined, vscode.ConfigurationTarget.Global);
      return;
    }
    await config.update('compiler.path', path.join(bin, exe('nova-compile')), vscode.ConfigurationTarget.Global);
    await config.update('translate.path', path.join(bin, exe('nova-translate')), vscode.ConfigurationTarget.Global);
  });

  test('a .qasm file is OpenQASM, and opening one starts the extension, with Nova', async () => {
    const doc = await open('bell.qasm');
    assert.strictEqual(doc.languageId, 'openqasm');
    const ext = vscode.extensions.getExtension('parhitmathur.nova-quantum');
    assert.ok(ext);
    await until('the extension to activate', () => ext.isActive);
    assert.ok(vscode.extensions.getExtension('parhitmathur.nova-programming-language'), 'the Nova extension it depends on is there');
    const commands = await vscode.commands.getCommands(true);
    for (const c of ['novaQuantum.runOnSimulator', 'novaQuantum.compileForHardware', 'novaQuantum.showCircuit', 'novaQuantum.exportOpenQASM']) {
      assert.ok(commands.includes(c), c);
    }
  });

  test("an OpenQASM circuit's errors show as you type, from nova-compile", async () => {
    const doc = await open('broken.qasm');
    const diags = await until('diagnostics', () => {
      const d = vscode.languages.getDiagnostics(doc.uri);
      return d.length ? d : undefined;
    });
    assert.strictEqual(diags[0].range.start.line, 4);
    assert.match(diags[0].message, /the gate 'foo' is not read yet/);
    assert.strictEqual(diags[0].source, 'nova-compile');
  });

  test("a circuit's compiled numbers show above it, with Run on simulator", async () => {
    const doc = await open('bell.qasm');
    const titles = await until('the circuit numbers', async () => {
      const t = await lensTitles(doc);
      return t.some(x => x.startsWith('⚛')) ? t : undefined;
    });
    const numbers = titles.find(t => t.startsWith('⚛')) ?? '';
    assert.match(numbers, /2 qubits · \d+ native gates · depth \d+/);
    assert.match(numbers, /on the machine/);  // a straight-line circuit is scheduled: it has a time
    assert.ok(titles.includes('▷ Run on simulator'));
  });

  test('a Nova quantum program gets its numbers too', async () => {
    const doc = await open('ghz.nova');
    const titles = await until('the circuit numbers', async () => {
      const t = await lensTitles(doc);
      return t.some(x => x.startsWith('⚛')) ? t : undefined;
    });
    assert.ok(titles.some(t => /^⚛ 3 qubits/.test(t)), titles.join(' | '));
  });

  test('Show the compiled circuit: the hardware instructions, in a view beside', async () => {
    const doc = await open('bell.qasm');
    await vscode.commands.executeCommand('novaQuantum.showCircuit', doc.uri, 'hardware');
    const view = await until('the circuit view', () =>
      vscode.window.visibleTextEditors.find(e => e.document.uri.scheme === 'nova-circuit')?.document);
    assert.match(view.getText(), /atom\./);
    assert.match(view.getText(), /^\/\/ Hardware instructions/);
  });

  test('Export as OpenQASM: a Nova circuit, written as OpenQASM 3', async () => {
    const doc = await open('ghz.nova');
    await vscode.commands.executeCommand('novaQuantum.exportOpenQASM', doc.uri);
    const qasm = await until('the exported circuit', () =>
      vscode.workspace.textDocuments.find(d => d.languageId === 'openqasm' && d.isUntitled));
    assert.match(qasm.getText(), /OPENQASM 3/);
    assert.match(qasm.getText(), /cx /);
  });

  test('Run on simulator runs it in the Nova Quantum terminal', async () => {
    const doc = await open('bell.qasm');
    await vscode.commands.executeCommand('novaQuantum.runOnSimulator', doc.uri);
    assert.ok(await until('the terminal', () => vscode.window.terminals.find(t => t.name === 'Nova Quantum')));
  });
});
