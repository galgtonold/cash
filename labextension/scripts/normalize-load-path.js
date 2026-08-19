/**
 * Make the built extension portable off Windows.
 *
 * `jupyter labextension build` writes the entry point into the output
 * package.json using the HOST's path separator:
 *
 *     "_build": { "load": "static\\remoteEntry.<hash>.js" }
 *
 * JupyterLab serves that string to the browser as a URL, where a backslash is
 * not a separator -- so a bundle built on Windows fails to load anywhere. The
 * built output is COMMITTED (see README.md), which means whoever happens to run
 * the build decides whether the shipped wheel works. Normalising here removes
 * that from chance.
 *
 * tests/test_notebook/test_labextension_packaging.py asserts the committed
 * output is normalised, so a rebuild that skips this step is caught by the
 * ordinary pytest run.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const OUTPUT_PKG = path.resolve(__dirname, '..', '..', 'src', 'cash', 'labextension', 'package.json');

if (!fs.existsSync(OUTPUT_PKG)) {
  console.error(`\n[cash] BUILD BLOCKED: built package.json not found at ${OUTPUT_PKG}\n`);
  process.exit(1);
}

const raw = fs.readFileSync(OUTPUT_PKG, 'utf8');
const data = JSON.parse(raw);
const build = (data.jupyterlab || {})._build || {};

if (typeof build.load !== 'string') {
  console.error(`\n[cash] BUILD BLOCKED: jupyterlab._build.load missing from ${OUTPUT_PKG}\n`);
  process.exit(1);
}

const before = build.load;
build.load = before.split('\\').join('/');

if (before !== build.load) {
  // Preserve the builder's two-space formatting so the committed diff stays
  // readable and does not churn on every rebuild.
  fs.writeFileSync(OUTPUT_PKG, JSON.stringify(data, null, 2) + '\n', 'utf8');
  console.log(`[cash] normalised _build.load: ${before} -> ${build.load}`);
} else {
  console.log(`[cash] ok: _build.load already URL-shaped (${build.load})`);
}
