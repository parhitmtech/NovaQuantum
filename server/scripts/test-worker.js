'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const { once } = require('node:events');
const { execFile } = require('node:child_process');
const execFileAsync = require('node:util').promisify(execFile);
const { createExecutor } = require('../src/services/execution');

(async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-worker-'));
  try {
    const source = path.join(dir, 'program.nova');
    fs.writeFileSync(source, 'ignite: print(42)\n');
    const executor = createExecutor({ mode: 'docker', nodeEnv: 'production' });
    const child = executor.spawnCompiler([source], { cwd: dir });
    let stdout = '', stderr = '';
    child.stdout.on('data', data => { stdout += data });
    child.stderr.on('data', data => { stderr += data });
    const [code] = await once(child, 'close');
    assert.equal(code, 0, stderr);
    assert.equal(stdout.trim(), '42');
    console.log('Production worker compiled and ran a real Nova program.');

    // --emit-cuda writes program.cu next to the source: the worker must be able to write into its job directory
    fs.writeFileSync(source, 'ignite:\n    Tensor t = ones(2, 2)\n    print(tensor_sum(t))\n');
    const output = path.join(dir, 'program.cu');
    const compilation = executor.spawnCompiler([source, '--emit-cuda'], { cwd: dir, userId: 'worker-test' });
    let errors = '';
    compilation.stdout.resume(); compilation.stderr.on('data', data => { errors += data; });
    const [compiled] = await once(compilation, 'close');
    assert.equal(compiled, 0, errors);
    assert.ok(fs.statSync(output).size > 0, 'worker must write its output file into the job directory');
    console.log('Production worker wrote an output file into its job directory.');

    // A quantum program runs through the native compiler, in the same sandbox: compiled through MLIR and LLVM in
    // memory (--run) and run on the simulator. Teleportation: c always ends in a's state, |1> with probability 0.75.
    const quantum = [
      'ignite:',
      '    qubit a = qpu.alloc()', '    qubit b = qpu.alloc()', '    qubit c = qpu.alloc()',
      '    qpu.ry(a, 2.0943951023931953)', '    qpu.h(b)', '    qpu.cx(b, c)', '    qpu.cx(a, b)', '    qpu.h(a)',
      '    bool m1 = qpu.measure(a)', '    bool m2 = qpu.measure(b)',
      '    if (m2):', '        qpu.x(c)', '    if (m1):', '        qpu.z(c)',
      '    qpu.probs(c)', '',
    ].join('\n');
    fs.writeFileSync(source, quantum);
    const run = executor.spawnCompiler([source], { cwd: dir, userId: 'worker-test', native: true });
    let qout = '', qerr = '';
    run.stdout.on('data', data => { qout += data });
    run.stderr.on('data', data => { qerr += data });
    const [qcode] = await once(run, 'close');
    assert.equal(qcode, 0, qerr);
    assert.equal(qout.trim(), '|0>: 0.2500\n|1>: 0.7500');
    assert.match(qerr, /qnova-stats @main: 3 qubits/);
    console.log('Production worker compiled a quantum program with its JIT and ran it on the simulator.');

    // An OpenQASM circuit, compiled down to the neutral-atom machine's timed instructions
    const circuit = path.join(dir, 'program.qasm');
    fs.writeFileSync(circuit, 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n');
    const qasm = executor.spawnCompiler([circuit], { cwd: dir, userId: 'worker-test', native: true });
    let aout = '', aerr = '';
    qasm.stdout.on('data', data => { aout += data });
    qasm.stderr.on('data', data => { aerr += data });
    const [acode] = await once(qasm, 'close');
    assert.equal(acode, 0, aerr);
    assert.equal(aout.trim(), '|00>: 0.5000\n|11>: 0.5000');
    assert.match(aerr, /atom-timing @main/);
    console.log('Production worker ran an OpenQASM circuit as hardware instructions.');

    // Launch directly in detached mode: no executor/Node timer can stop this job.
    // Detached -i keeps stdin open; input() blocks without hitting Nova's instruction cap.
    fs.writeFileSync(source, 'ignite:\n    int x = input()\n    print(x)\n');
    const spec = executor.buildCommand([source], { cwd: dir, timeoutMs: 2000 });
    spec.args.splice(spec.args.indexOf('--rm'), 1); // retain state for timeout verification
    spec.args.splice(1, 0, '--detach');
    await execFileAsync(spec.command, spec.args, { env: spec.env, timeout: 30000 });
    try {
      const { stdout: running } = await execFileAsync('docker', ['inspect', '--format', '{{.State.Running}}', spec.containerName], { env: spec.env, timeout: 5000 });
      assert.equal(running.trim(), 'true', 'deadline fixture must start and remain running');
      const { stdout: exitCode } = await execFileAsync('docker', ['wait', spec.containerName], { env: spec.env, timeout: 10000 });
      assert.equal(exitCode.trim(), '137', 'container must receive SIGKILL from its internal deadline');
      console.log('Detached worker hit its internal deadline without a Node execution timer.');
    } finally {
      await execFileAsync('docker', ['rm', '-f', spec.containerName], { env: spec.env, timeout: 5000 }).catch(() => {});
    }
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
