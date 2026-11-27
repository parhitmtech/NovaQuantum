// Starts VS Code with Nova Quantum and the Nova extension it depends on (both from this repository), opens
// test/fixtures, and runs the tests in suite/. NOVA_QUANTUM_BIN is the folder that holds nova-compile and
// nova-translate (an MLIR build's bin/); VSCODE_VERSION picks the VS Code (default: the oldest supported).
import * as path from 'path';
import { runTests } from '@vscode/test-electron';

async function main() {
  const root = path.resolve(__dirname, '..', '..', '..');   // editors/vscode/nova-quantum (from test/out/test)
  const nova = path.resolve(root, '..', 'nova');
  const pkg = require(path.join(root, 'package.json'));
  const version = process.env.VSCODE_VERSION || String(pkg.engines.vscode).replace(/^\^/, '');
  const bin = process.env.NOVA_QUANTUM_BIN || path.resolve(root, '..', '..', '..', 'build', 'mlir', 'bin');
  try {
    const code = await runTests({
      version,
      // NOVA_TEST_EXTENSION_PATH: an unpacked package to test instead of this folder
      extensionDevelopmentPath: [process.env.NOVA_TEST_EXTENSION_PATH || root, nova],
      extensionTestsPath: path.join(__dirname, 'suite', 'index'),
      launchArgs: [path.join(root, 'test', 'fixtures'), '--disable-extensions', '--disable-workspace-trust'],
      extensionTestsEnv: { NOVA_QUANTUM_BIN: bin, NOVA_TEST_BUNDLED: process.env.NOVA_TEST_BUNDLED },
    });
    process.exit(code);
  } catch (e) {
    console.error('The extension tests did not run:', e);
    process.exit(1);
  }
}

main();
