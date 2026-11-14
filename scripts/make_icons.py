"""Rubato logo ("Wave": a triad riding a stretched line) -> SVG + PNG app icons.

Writes src/chordstream/web/icons/: icon.svg (tile), mark.svg (no tile, for the page header),
favicon-32.png, apple-touch-icon.png (180), icon-192.png, icon-512.png, icon-maskable-512.png, og.png.
Needs Pillow (dev only; the PNGs are committed). usage: python scripts/make_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "src" / "chordstream" / "web" / "icons"
BG, LINE, DOT, DOT2, FG = "#151a2e", "#6d8dff", "#6d8dff", "#a78bfa", "#eceef3"
# Geometry in a 132-unit box centred on 0 (the concept sketch): one cubic-Bezier pair, stroke 7.
P0, C1, C2, P1 = (-52, 18), (-30, 18), (-26, -22), (-6, -22)
C3, C4, P2 = (16, -22), (14, 26), (34, 26)
C5, C6, P3 = (44, 26), (48, 10), (54, 4)  # the upturn at the end: it reads as a wave, not a peak
STROKE, R = 7.5, 10.0


def bez(a, b, c, d, t):
    u = 1 - t
    return tuple(u**3 * a[i] + 3 * u * u * t * b[i] + 3 * u * t * t * c[i] + t**3 * d[i] for i in range(2))


CURVE = ([bez(P0, C1, C2, P1, i / 400) for i in range(401)] + [bez(P1, C3, C4, P2, i / 400) for i in range(1, 401)]
         + [bez(P2, C5, C6, P3, i / 400) for i in range(1, 401)])


def on_curve(x):  # the curve point nearest to x (dots sit on the line)
    return min(CURVE, key=lambda p: abs(p[0] - x))


DOTS = [(on_curve(-32), DOT), (P1, DOT), (on_curve(28), DOT2)]
# Centre the drawing: the curve spans x -52..34, y -22..26 (+ dot radius).
xs, ys = [p[0] for p in CURVE], [p[1] for p in CURVE]
CX, CY = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
SPAN = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * R


def svg(tile: bool) -> str:
    v = 132 if tile else SPAN + 4
    ox, oy = (66 - CX, 66 - CY) if tile else (v / 2 - CX, v / 2 - CY)
    pt = lambda p: f"{p[0] + ox:.2f} {p[1] + oy:.2f}"
    path = f"M{pt(P0)} C{pt(C1)} {pt(C2)} {pt(P1)} C{pt(C3)} {pt(C4)} {pt(P2)} C{pt(C5)} {pt(C6)} {pt(P3)}"
    body = (f'<path d="{path}" fill="none" stroke="{LINE if tile else "currentColor"}" stroke-width="{STROKE}" stroke-linecap="round"/>'
            + "".join(f'<circle cx="{p[0] + ox:.2f}" cy="{p[1] + oy:.2f}" r="{R}" fill="{c if tile else ("var(--accent-2, " + c + ")" if c == DOT2 else "currentColor")}"/>' for p, c in DOTS))
    bg = f'<rect width="132" height="132" rx="30" fill="{BG}"/>' if tile else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {v:.1f} {v:.1f}">{bg}{body}</svg>\n'


def png(size: int, content: float, radius: float | None, ss: int = 4) -> Image.Image:
    """content: fraction of the canvas the drawing spans; radius: tile corner (fraction), None = full bleed."""
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if radius is None:
        d.rectangle([0, 0, S, S], fill=BG)
    else:
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=radius * S, fill=BG)
    k = content * S / SPAN
    tx = lambda p: ((p[0] - CX) * k + S / 2, (p[1] - CY) * k + S / 2)
    pts = [tx(p) for p in CURVE]
    w = STROKE * k
    for x, y in pts:  # stroke = discs stamped along a dense curve: clean round edges and caps
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=LINE)
    for p, c in DOTS:
        x, y = tx(p)
        d.ellipse([x - R * k, y - R * k, x + R * k, y + R * k], fill=c)
    return img.resize((size, size), Image.LANCZOS)


def og() -> Image.Image:
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), BG)
    icon = png(300, 0.78, 0.23)
    img.paste(icon, (110, (H - 300) // 2), icon)
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 120)
        small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)
    except OSError:
        big = small = ImageFont.load_default()
    d.text((470, 205), "Rubato", font=big, fill=FG)
    d.text((474, 355), "Live chords, in your browser", font=small, fill="#8b90a0")
    return img


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "icon.svg").write_text(svg(True), encoding="utf-8")
(OUT / "mark.svg").write_text(svg(False), encoding="utf-8")
png(32, 0.86, 0.22).save(OUT / "favicon-32.png")
png(180, 0.66, None).save(OUT / "apple-touch-icon.png")
png(192, 0.72, 0.23).save(OUT / "icon-192.png")
png(512, 0.72, 0.23).save(OUT / "icon-512.png")
png(512, 0.56, None).save(OUT / "icon-maskable-512.png")  # content inside the 80% safe zone
og().save(OUT / "og.png")
print("icons ->", OUT)
