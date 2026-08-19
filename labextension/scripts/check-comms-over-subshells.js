/**
 * Build-time guard for the one line in this extension that is load-bearing.
 *
 *     comm.commsOverSubshells = 'disabled';
 *
 * JupyterLab 4.6 defaults `commsOverSubshells: perCommTarget`, which delivers
 * comms on a SUBSHELL THREAD. With that default:
 *
 *   1. ordering against the execute_request stops being FIFO and becomes a
 *      race (measured 0.4-7.4ms lead, won 130/130 -- evidence that reads as a
 *      guarantee right up until a slower machine disagrees), and
 *   2. the kernel-side store in src/cash/notebook/live_cells.py, which is
 *      deliberately lock-free, becomes cross-thread mutable state.
 *
 * Nothing on the Python side can enforce this -- it is comment-only there. So
 * this script fails the build if the line is missing, both in the TypeScript
 * source and in the bundle that actually ships.
 *
 * Usage:
 *   node scripts/check-comms-over-subshells.js --source-only   (before tsc)
 *   node scripts/check-comms-over-subshells.js                 (after build)
 *
 * The same invariant is asserted from Python, without Node, by
 * tests/test_notebook/test_labextension_packaging.py.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SOURCE = path.join(ROOT, 'src', 'index.ts');
const OUTPUT_DIR = path.resolve(ROOT, '..', 'src', 'cash', 'labextension');

// Deliberately loose about whitespace and quote style, and about the receiver
// name, so that a rename or a reformat does not read as a removal. Minifiers
// keep property names and string literals, so the same pattern matches the
// built bundle.
const PATTERN = /commsOverSubshells\s*[=:]\s*["']disabled["']/;

const EXPLANATION = [
  '',
  "  `comm.commsOverSubshells = 'disabled'` is missing.",
  '',
  '  This is not a tuning knob. Without it JupyterLab 4.6 delivers the comm on',
  '  a subshell thread, which (a) turns the push-before-execute_request ordering',
  '  from a FIFO guarantee into a race, and (b) makes the lock-free store in',
  '  src/cash/notebook/live_cells.py cross-thread mutable state.',
  '',
  '  If you removed it on purpose, you are changing the design: update',
  '  labextension/src/index.ts, live_cells.py\'s thread-safety note, this script,',
  '  and tests/test_notebook/test_labextension_packaging.py together.',
  ''
].join('\n');

function fail(what, where) {
  console.error(`\n[cash] BUILD BLOCKED: ${what}\n  looked in: ${where}${EXPLANATION}`);
  process.exit(1);
}

function checkSource() {
  if (!fs.existsSync(SOURCE)) {
    fail('extension source not found', SOURCE);
  }
  if (!PATTERN.test(fs.readFileSync(SOURCE, 'utf8'))) {
    fail('the TypeScript source no longer forces the comm onto the main shell', SOURCE);
  }
  console.log(`[cash] ok: source forces commsOverSubshells='disabled' (${path.relative(ROOT, SOURCE)})`);
}

function collectJs(dir, acc) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      collectJs(full, acc);
    } else if (entry.name.endsWith('.js')) {
      acc.push(full);
    }
  }
  return acc;
}

function checkBundle() {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fail('the built labextension is missing (did `jupyter labextension build .` run?)', OUTPUT_DIR);
  }
  const files = collectJs(OUTPUT_DIR, []);
  if (files.length === 0) {
    fail('the built labextension contains no JavaScript', OUTPUT_DIR);
  }
  const hit = files.find(f => PATTERN.test(fs.readFileSync(f, 'utf8')));
  if (!hit) {
    fail(
      `the SHIPPED bundle does not force the comm onto the main shell ` +
        `(${files.length} .js files scanned)`,
      OUTPUT_DIR
    );
  }
  console.log(`[cash] ok: bundle forces commsOverSubshells='disabled' (${path.relative(ROOT, hit)})`);
}

checkSource();
if (!process.argv.includes('--source-only')) {
  checkBundle();
}
