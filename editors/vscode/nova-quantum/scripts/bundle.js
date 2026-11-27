#!/usr/bin/env node
// Puts Nova's quantum tools into the extension before a platform's package is made (vsce package --target):
// bin/nova-compile and bin/nova-translate from an MLIR build. The extension finds them there first, so the package
// works with nothing else installed.
//   node scripts/bundle.js <target> <the MLIR build's bin directory>     target: linux-x64, win32-x64, ...
'use strict';
const fs = require('fs');
const path = require('path');

const [target, binDir] = process.argv.slice(2);
if (!target || !binDir) {
  console.error('usage: node scripts/bundle.js <target> <the MLIR build\'s bin directory>');
  process.exit(1);
}
const root = path.resolve(__dirname, '..');
const windows = target.startsWith('win32');
fs.rmSync(path.join(root, 'bin'), { recursive: true, force: true });
fs.mkdirSync(path.join(root, 'bin'));
for (const tool of ['nova-compile', 'nova-translate']) {
  const name = windows ? `${tool}.exe` : tool;
  const from = path.join(path.resolve(binDir), name);
  if (!fs.existsSync(from)) {
    console.error(`missing ${from}: build it first (ninja nova-compile nova-translate)`);
    process.exit(1);
  }
  fs.copyFileSync(from, path.join(root, 'bin', name));
  if (!windows) fs.chmodSync(path.join(root, 'bin', name), 0o755);
}
console.log(`bundled nova-compile and nova-translate for ${target} from ${binDir}`);
