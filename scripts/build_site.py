"""Assemble the browser-only site (GitHub Pages): the page, the in-browser engine, onnxruntime-web and
a web model bundle. Standard library only, so CI can run it without the training stack.

usage: python scripts/build_site.py --model <web bundle dir> --ort <onnxruntime-web dist dir> --out _site
The web bundle comes from `chordstream export-web models/vNN --out <dir>` (published as a release asset).
"""
import argparse
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "src" / "chordstream" / "web"
STATIC = ["sheet.js", "engine.js", "engine-worker.js", "api-local.js"]
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
html = html.replace("<head>", "<head>\n<script>window.CHORDSTREAM_LOCAL = true;</script>", 1)
(out / "index.html").write_text(html, encoding="utf-8")
(out / "bench.html").write_text((WEB / "bench.html").read_text(encoding="utf-8"), encoding="utf-8")
for f in STATIC:
    shutil.copyfile(WEB / f, out / "static" / f)
for f in ORT:
    shutil.copyfile(Path(a.ort) / f, out / "ort" / f)
if Path(a.model).resolve() != (out / "model").resolve():
    for f in MODEL:
        shutil.copyfile(Path(a.model) / f, out / "model" / f)
(out / ".nojekyll").write_text("")
print(f"site -> {out}  ({sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.1f} MB)")
