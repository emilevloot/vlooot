// Python's standard library for the page, compiled in advance.
//
// Pyodide's python_stdlib.zip holds the library as source (.py). Python
// can't save its compiled bytecode inside a zip, so every start compiled the
// ~180 modules Python needs again: seconds on a phone. This script lets
// Pyodide itself (the same Python 3.12 as in the browser; the bytecode must
// match it) compile every module once, and writes py/python_stdlib.b64.txt
// with only the bytecode (.pyc, "unchecked": used as they are).
//
//   node tools/compile_stdlib.cjs            (the original is kept as py/python_stdlib.src.b64.txt)
//
// After a new Pyodide: delete python_stdlib.src.b64.txt, copy the new
// library to python_stdlib.b64.txt, and run this again.

const fs = require('fs');
const os = require('os');
const path = require('path');

const PY = path.join(__dirname, '..', 'py');
const SRC = path.join(PY, 'python_stdlib.src.b64.txt');
const OUT = path.join(PY, 'python_stdlib.b64.txt');

(async () => {
  if (!fs.existsSync(SRC)) fs.copyFileSync(OUT, SRC);          // the source version, once
  const zipPath = path.join(os.tmpdir(), 'looot_stdlib_src.zip');
  fs.writeFileSync(zipPath, Buffer.from(fs.readFileSync(SRC, 'utf8').trim(), 'base64'));
  const { loadPyodide } = require(path.join(PY, 'pyodide.js'));
  const py = await loadPyodide({ indexURL: PY + path.sep, stdLibURL: zipPath });
  const t = Date.now();
  const report = py.runPython(`
import zipfile, importlib.util, importlib._bootstrap_external as ext
src = zipfile.ZipFile("/lib/python312.zip")
out = zipfile.ZipFile("/tmp/stdlib_pyc.zip", "w", zipfile.ZIP_DEFLATED)
n = bad = 0
for info in src.infolist():
    data = src.read(info)
    if info.filename.endswith(".py"):
        try:
            code = compile(data, "/lib/python312.zip/" + info.filename, "exec", dont_inherit=True)
        except SyntaxError:                  # (test files with errors on purpose)
            out.writestr(info, data); bad += 1
            continue
        pyc = ext._code_to_hash_pyc(code, importlib.util.source_hash(data), checked=False)
        out.writestr(info.filename[:-3] + ".pyc", bytes(pyc))
        n += 1
    else:
        out.writestr(info, data)
out.close()
"%d modules compiled, %d kept as source" % (n, bad)
`);
  const bin = py.FS.readFile('/tmp/stdlib_pyc.zip');
  fs.writeFileSync(OUT, Buffer.from(bin).toString('base64'));
  fs.unlinkSync(zipPath);
  console.log(report + ' in ' + ((Date.now() - t) / 1000).toFixed(1) + ' s; ' + OUT + ': ' + (bin.length / 2 ** 20).toFixed(1) + ' MB zip');
})().catch(e => { console.error('FAILED', e); process.exit(1); });
