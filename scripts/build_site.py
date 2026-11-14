"""Assemble the browser-only site (GitHub Pages): the page, the in-browser engine, onnxruntime-web and
a web model bundle. Standard library only, so CI can run it without the training stack.

usage: python scripts/build_site.py --model <web bundle dir> --ort <onnxruntime-web dist dir> --out _site
The web bundle comes from `chordstream export-web models/vNN --out <dir>` (published as a release asset).
"""
import argparse
import hashlib
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "src" / "chordstream" / "web"
STATIC = ["sheet.js", "lyrics.js", "engine.js", "engine-worker.js"]
ORT = ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]
MODEL = ["meta.json", "model.onnx", "state.bin", "hmm.bin"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ort", default="tests/web/node_modules/onnxruntime-web/dist")
ap.add_argument("--out", default="_site")
a = ap.parse_args()
out = Path(a.out)
for sub in ("static", "ort", "model"):
    (out / sub).mkdir(parents=True, exist_ok=True)
html = (WEB / "index.html").read_text(encoding="utf-8")
(out / "index.html").write_text(html, encoding="utf-8")
(out / "bench.html").write_text((WEB / "bench.html").read_text(encoding="utf-8"), encoding="utf-8")
# Cache-busting: browsers (and Pages' 10-minute cache) must never pair a new page with an old engine.
ver = hashlib.sha1(b"".join((WEB / f).read_bytes() for f in STATIC) + (Path(a.model) / "meta.json").read_bytes()).hexdigest()[:10]
for f in STATIC:
    text = (WEB / f).read_text(encoding="utf-8")
    if f == "engine-worker.js":
        text = text.replace("'./engine.js'", f"'./engine.js?v={ver}'")
    (out / "static" / f).write_text(text, encoding="utf-8")
html = (out / "index.html").read_text(encoding="utf-8")
for f in STATIC:
    html = html.replace(f"static/{f}'", f"static/{f}?v={ver}'").replace(f'static/{f}"', f'static/{f}?v={ver}"')
(out / "index.html").write_text(html, encoding="utf-8")
bench = (out / "bench.html").read_text(encoding="utf-8").replace("'./static/engine.js'", f"'./static/engine.js?v={ver}'")
(out / "bench.html").write_text(bench, encoding="utf-8")
for f in ORT:
    shutil.copyfile(Path(a.ort) / f, out / "ort" / f)
if Path(a.model).resolve() != (out / "model").resolve():
    for f in MODEL:
        shutil.copyfile(Path(a.model) / f, out / "model" / f)
(out / ".nojekyll").write_text("")
print(f"site -> {out}  ({sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.1f} MB)")
