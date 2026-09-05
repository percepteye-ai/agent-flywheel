#!/usr/bin/env python3
"""The mark that spins beside the README title: an asymmetric ∞ in a wheel.

THE MARK IS THE PITCH, which is why it is not the 🎡 emoji. A flywheel (the
ring) enclosing a loop that never ends (the lemniscate), whose second lobe is
visibly larger than its first: the agent you deployed, then the agent you got
back. The bright node rides the loop small-lobe-first, so the whole thing reads
as growth rather than as a lopsided symbol.

Growth is stated THREE times, deliberately, because one signal does not survive
a 44px downscale: the improved lobe is bigger (:data:`RATIO`), the stroke
through it is thicker, and the travelling node grows as it enters it.

THE LOBES ARE A REAL BERNOULLI LEMNISCATE, each UNIFORMLY scaled. That detail
is load-bearing and cost two rewrites to find. The obvious construction --
two closed ovals sharing a point, rotated about it -- does NOT make an infinity:
rotating a closed loop about a point on its own boundary just swings the lobe
around, so the halves come out either kissing (two tangent circles) or
overlapping (a Venn diagram), never crossing. Scaling each lobe of a true
lemniscate works because a UNIFORM scale leaves that lobe's ±45° tangent at the
origin untouched -- so the two halves still meet at a real X and the curve stays
smooth through the centre while the lobes differ in size.

THE COLOURS ARE THE PRODUCT'S, not a guess. ``--brand-cyan`` #2DD9C9 through
``--brand-green`` #1FE072 is ``--brand-grad``, the gradient Mission Control's
dashboard is built on (``frontend/src/styles/dashboard.css``). Sweeping it
around the path and then turning the ring means the motion and the identity are
one decision rather than two.

SEAMLESS BY CONSTRUCTION. The ring turns exactly one revolution and the gradient
advances exactly one cycle across the frame run, so the last frame is one step
short of the first. Nothing is eased -- easing is what puts a stutter at the seam.

TWO THINGS THAT LOOK LIKE STYLE AND ARE NOT:

* THE NODE HAS A DARK RING. The node is ``--brand-teal``, which sits between
  the gradient's two stops -- so on the stretch where the stroke behind it is
  also teal-ish it would disappear into its own path. The deep ring is what
  separates them, and it is also what keeps the node visible on GitHub's light
  theme, where an unringed node loses its edge against the gradient.
* THE MATTE IS MID-GREY. GIF transparency is ONE BIT -- there is no alpha blend,
  so a half-covered edge pixel is either fully drawn or fully gone. Drawing at
  :data:`SS`x and downsampling gives real antialiasing in RGB, and cutting the
  alpha against mid-grey leaves surviving edge pixels leaning neither white nor
  black, so the mark picks up no halo on either theme.

Run through ``render.sh spinner``, which owns the ffmpeg half. Needs Pillow --
use the repo venv (``PY=.venv/bin/python``), not a bare ``python3``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw

#: 48 frames at 24fps is a 2s revolution -- fast enough to read as motion in
#: peripheral vision beside a title, slow enough not to strobe.
FPS = 24
FRAMES = 48

SIZE = 144                  # rendered; displayed at 44, so 3x for retina
SS = 8                      # supersample factor

#: ``--brand-grad``: linear-gradient(135deg, #2DD9C9 0%, #1FE072 100%).
CYAN = (45, 217, 201)
GREEN = (31, 224, 114)
#: ``--brand-teal`` #1FCBA8 -- the third brand stop, so the node belongs to the
#: same palette as the loop it rides rather than reading as a white highlight.
CORE = (31, 203, 168)
#: Deep brand teal. Two jobs: separate the node from the cyan/green stroke it
#: sits on, and make it visible on a WHITE ground, where an unringed node in
#: any brand colour loses its edge against the gradient behind it.
RIMINK = (7, 74, 63)
MATTE = (128, 128, 128)

#: The incumbent lobe as a fraction of the improved one.
RATIO = 0.52
#: Lobe roundness. A bare lemniscate is flat; >1 fattens both lobes.
YSCALE = 1.60
#: Degrees of ring left open. The gap is what makes the wheel visibly TURN --
#: a closed ring rotating is indistinguishable from a static one.
GAP = 90


def _mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _grad(t):
    """cyan -> green -> cyan, so the sweep closes on itself with no seam."""
    u = (t % 1.0) * 2
    return _mix(CYAN, GREEN, u) if u <= 1 else _mix(GREEN, CYAN, u - 1)


def _lem(s):
    """The path. ``s`` in [0,1) runs the SMALL lobe first, then the LARGE."""
    t = s * 2 * math.pi + math.pi          # +pi so s=0 starts on the small lobe
    d = 1 + math.sin(t) ** 2
    x = math.cos(t) / d
    y = math.sin(t) * math.cos(t) / d
    k = 1.0 if x >= 0 else RATIO
    return x * k, y * k * YSCALE


def _ring(d, c, r, w, t, segs=220):
    box = [c - r, c - r, c + r, c + r]
    span = 360 - GAP
    a0 = 360 * t
    for i in range(segs):
        f = i / segs
        d.arc(box, a0 + span * f, a0 + span * (i + 1) / segs + 0.9,
              fill=_grad(t + f), width=int(w))
    # Rounded ends. Pillow's arc has butt caps only, and square ends are the
    # single biggest thing that makes a mark like this look cheap.
    for f, ang in ((0.0, a0), (1.0, a0 + span)):
        rad = math.radians(ang)
        rr = w / 2
        x, y = c + math.cos(rad) * r, c + math.sin(rad) * r
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=_grad(t + f))


def _frame(t: float) -> Image.Image:
    n = SIZE * SS
    img = Image.new("RGBA", (n, n), MATTE + (0,))
    d = ImageDraw.Draw(img)
    c = n / 2

    _ring(d, c, r=n * 0.443, w=n * 0.045, t=t)

    a = n * 0.335
    off = (1 - RATIO) / 2 * a          # centre the asymmetric figure in the ring
    base = n * 0.052

    # THE LOOP, drawn as overlapping discs. Round caps and smooth joins come
    # free, and the per-point width lets the stroke thicken through the
    # improved lobe.
    pts = 2200
    for i in range(pts):
        s = i / pts
        x, y = _lem(s)
        r = base / 2 * (1.16 if x >= 0 else 0.84)
        px, py = c + x * a - off, c + y * a
        d.ellipse([px - r, py - r, px + r, py + r], fill=_grad(s + t))

    # THE NODE. Grows as it enters the improved lobe; see the module docstring
    # for why it carries a dark ring.
    x, y = _lem(t % 1.0)
    r = base * (0.74 if x < 0 else 0.95)
    px, py = c + x * a - off, c + y * a
    d.ellipse([px - r, py - r, px + r, py + r], fill=CORE,
              outline=RIMINK, width=int(base * 0.30))

    small = img.resize((SIZE, SIZE), Image.LANCZOS)

    # ONE-BIT ALPHA, cut once, on the DOWNSAMPLED image so the threshold lands
    # on a real coverage value rather than on a hard supersampled edge.
    rgb = Image.new("RGB", small.size, MATTE)
    rgb.paste(small, mask=small.split()[3])
    alpha = small.split()[3].point(lambda v: 255 if v > 127 else 0)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def main() -> None:
    out = Path(os.environ.get("FRAMES", "frames_spinner"))
    out.mkdir(parents=True, exist_ok=True)
    for i in range(FRAMES):
        _frame(i / FRAMES).save(out / f"f{i:04d}.png")
    print(f"{FRAMES} frames -> {out}")


if __name__ == "__main__":
    main()
