// Runs the real HYDRA engine (Python) in the browser via Pyodide. Packages come from the Pyodide CDN.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let ready = null;
function init() {
  if (ready) return ready;
  ready = (async () => {
    importScripts(PYODIDE + "pyodide.js");
    postMessage({ type: "status", msg: "Loading Python (WebAssembly)…" });
    const py = await loadPyodide({ indexURL: PYODIDE });
    postMessage({ type: "status", msg: "Loading NumPy, SciPy, Pydantic…" });
    await py.loadPackage(["numpy", "scipy", "pydantic"]);
    postMessage({ type: "status", msg: "Loading HYDRA engine…" });
    const files = await (await fetch("engine.json")).json();
    for (const [p, src] of Object.entries(files)) {
      const dst = "/" + p, dir = dst.slice(0, dst.lastIndexOf("/"));
      if (dir) py.FS.mkdirTree(dir);
      py.FS.writeFile(dst, src);
    }
    py.runPython("import sys; sys.path.insert(0, '/')\nexec(open('/glue.py').read(), globals())");
    return py;
  })();
  return ready;
}
onmessage = async (e) => {
  try {
    const py = await init();
    postMessage({ type: "ready" });
    if (e.data.type !== "run") return;
    postMessage({ type: "status", msg: "Simulating…" });
    py.globals.set("_sc", JSON.stringify(e.data.scenario));
    const t0 = performance.now();
    const out = py.runPython("run(_sc)");
    postMessage({ type: "result", data: JSON.parse(out), seconds: (performance.now() - t0) / 1000 });
  } catch (err) {
    postMessage({ type: "error", msg: String(err.message || err).split("\n").slice(-6).join("\n") });
  }
};
