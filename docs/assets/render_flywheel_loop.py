"""Render the PerceptEye Agent Flywheel LOOP as PNG frames.

One round of the loop, a station at a time, drawn as a ring across a trust
boundary -- the animated twin of ``flywheel-architecture.svg``:

1. capture -- the agent's real conversations are recorded where it runs: by our
   SDK, by the OpenTelemetry traces it already exports, or read from the
   observability provider they already go to;
2. learn   -- once the customer allows it, they go out, and a new version is
   learned from them;
3. test    -- the new version is tested against the agent's current best, on
   conversations it has never seen;
4. approve -- a version that did better waits in Decisions, and a person
   approves it or not;
5. repeat  -- the agent runs the winner, and the next round starts from it as
   new conversations arrive.

Practice tasks are an optional extra, not the loop, so they get a footnote.

Design constraints:
  * Every connection is opened from the customer's side. The figure says so in
    its footer, and draws no arrow from the platform that the customer's side
    did not ask for.
  * The platform is ONE box. No internal service names, no backend
    architecture: ``tests/test_oss_boundary.py`` scans every tracked file,
    this one included. The customer can still watch every round in the
    dashboard, and the box says so.
  * There is no score and no bar chart. "Did better" is the rule the product
    applies; a number here would be one nobody measured.
  * Customer words only, and the loop is described rather than summarised as
    "we train a model".

ONE FIGURE FOR BOTH SDKs. The loop is the same for every way of connecting an
agent, so both READMEs carry these same bytes, and
``tests/test_node_figures_match.py`` holds the Node repo's copy to them.

Usage:
    FRAMES=/tmp/frames python render_flywheel_loop.py
    ...then encode with ffmpeg -- see render.sh alongside this file.
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 18
DUR = 21.0
NFRAMES = int(DUR * FPS)
SS = 2  # supersample factor for crisp edges

OUT = os.environ.get("FRAMES", "")

# ── palette: the light set the overview figure uses, so the two match ───────
BG = (255, 255, 255)
CARD = (252, 253, 254)
LINE = (196, 205, 214)
LINE_SOFT = (226, 232, 238)
TXT = (9, 14, 20)
MUTED = (74, 84, 96)
DIM = (114, 124, 137)
BLUE = (0, 82, 204)
PURPLE = (88, 36, 190)
GREEN = (2, 112, 48)
RED = (186, 24, 24)
DOT_BASE = (205, 212, 220)
ZL_FILL = (248, 250, 252)
ZL_LABEL = (88, 98, 110)
ZR_FILL = (250, 248, 255)
ZR_EDGE = (206, 192, 234)
ZR_LABEL = (88, 36, 190)
BD_DASH = (168, 148, 206)
BD_FILL = (243, 238, 253)
BD_EDGE = (190, 170, 224)
BD_TXT = (88, 36, 190)
FRAME_EDGE = (154, 165, 177)
FOOT = (112, 96, 148)
HRULE = (222, 227, 233)

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
MONO_PATH = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, weight: str = "Regular", mono: bool = False):
    f = ImageFont.truetype(MONO_PATH if mono else FONT_PATH, size * SS)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F_TITLE = font(24, "Semibold")
F_H = font(14, "Semibold")
F_ZONE = font(11, "Bold")
F_BOX = font(18, "Semibold")
F_SUB = font(13, "Regular")
F_TINY = font(11, "Medium")
F_LANE = font(12, "Semibold")
F_CAP = font(18, "Semibold")
F_CAPSUB = font(14, "Regular")
F_CAPNUM = font(13, "Bold")
F_CHIP = font(11, "Semibold")
F_ROW = font(13, "Regular", mono=True)

# ── geometry (unscaled logical coords) ───────────────────────────────────────
LZ = (40, 92, 612, 612)          # your infrastructure
BOUND_X = 640
RZ = (668, 92, 1240, 612)        # PerceptEye
AGENT = (72, 134, 580, 314)      # top-left: your agent
DECIDE = (72, 410, 580, 584)     # bottom-left: Decisions
LEARN = (700, 134, 1208, 314)    # top-right: learn
TEST = (700, 410, 1208, 584)     # bottom-right: test
OUT_Y = 236                      # conversations, out
BACK_Y = 498                     # the winner, back
L_MID = (AGENT[0] + AGENT[2]) / 2
R_MID = (LEARN[0] + LEARN[2]) / 2

SOURCES = ("our SDK", "OpenTelemetry", "your observability provider")

# ── timeline (seconds): (key, t0, t1, badge, verb, caption) ──────────────────
STAGES = [
    ("today",   0.0,  2.4, "0", "your agent, today",
     "In your process, on your credentials, behind your firewall."),
    ("capture", 2.4,  5.6, "1", "capture",
     "Its real conversations, recorded where it runs. Nothing inside it changes."),
    ("allow",   5.6,  8.2, "2", "with your permission",
     "None of its conversations are kept until you allow it. Outbound only."),
    ("learn",   8.2, 11.2, "3", "learn",
     "A new version, learned from your agent's own conversations."),
    ("test",   11.2, 14.2, "4", "test",
     "Against your agent's current best, on conversations it has never seen."),
    ("approve", 14.2, 17.2, "5", "you approve the winner",
     "A version that did better waits for you in Decisions."),
    ("repeat", 17.2, 21.0, "6", "repeat",
     "Your agent runs the winner. The next round starts from it."),
]


def stage_at(t: float):
    for s in STAGES:
        if s[1] <= t < s[2]:
            return s
    return STAGES[-1]


def prog(t: float, a: float, b: float) -> float:
    return 0.0 if t <= a else 1.0 if t >= b else (t - a) / (b - a)


def ease(x: float) -> float:
    return x * x * (3 - 2 * x)


def at(t: float, a: float, b: float) -> float:
    return ease(prog(t, a, b))


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(round(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def fade(c, a):
    return mix(BG, c, a)


# ── drawing helpers (logical coords, scaled internally) ──────────────────────
def S(v):
    return tuple(x * SS for x in v) if isinstance(v, (tuple, list)) else v * SS


def rrect(d, box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(S(box), radius=radius * SS, fill=fill, outline=outline,
                        width=max(1, int(width * SS)))


def dashed_v(d, x, y0, y1, color, dash=7, gap=6, width=1):
    y = y0
    while y < y1:
        d.line([S(x), S(y), S(x), S(min(y + dash, y1))], fill=color,
               width=max(1, int(width * SS)))
        y += dash + gap


def dashed_h(d, y, x0, x1, color, dash=6, gap=5, width=1):
    x = x0
    while x < x1:
        d.line([S(x), S(y), S(min(x + dash, x1)), S(y)], fill=color,
               width=max(1, int(width * SS)))
        x += dash + gap


def text(d, xy, s, f, fill, anchor="la"):
    d.text(S(xy), s, font=f, fill=fill, anchor=anchor)


def tw(s, f) -> float:
    return f.getbbox(s)[2] / SS


def arrowhead(d, x, y, color, size=6, direction=(1, 0)):
    dx, dy = direction
    px, py = -dy, dx
    pts = [(x - dx * size + px * size * .75, y - dy * size + py * size * .75),
           (x, y),
           (x - dx * size - px * size * .75, y - dy * size - py * size * .75)]
    d.polygon([S(p) for p in pts], fill=color)


def packet(d, x, y, color, w=16, h=7, alpha=1.0, trail=3, direction=1):
    """A chip travelling a lane, with a fading trail behind it."""
    for i in range(trail, -1, -1):
        a = alpha * (1.0 - i / (trail + 1.4))
        if a <= 0.02:
            continue
        ox = x - i * (w * 0.85) * direction
        rrect(d, (ox - w / 2, y - h / 2, ox + w / 2, y + h / 2), h / 2,
              fill=fade(color, a))


def pill(d, x, y, label, color, on, a=1.0, base=CARD, pad=10, hgt=22, dot=True):
    """A labelled pill; `on` in 0..1 lights it, `a` is its card's visibility."""
    wpx = tw(label, F_CHIP) + pad * 2 + (10 if dot else 0)
    if a <= .01:
        return wpx
    rrect(d, (x, y, x + wpx, y + hgt), hgt / 2,
          fill=fade(mix(base, mix(BG, color, 0.20), on), a),
          outline=fade(mix(LINE, color, on), a), width=1)
    tx = x + pad
    if dot:
        d.ellipse(S((x + pad - 3, y + hgt / 2 - 2, x + pad + 1, y + hgt / 2 + 2)),
                  fill=fade(mix(DIM, color, on), a))
        tx = x + pad + 8
    text(d, (tx, y + hgt / 2 + 1), label, F_CHIP, fade(mix(MUTED, TXT, on), a),
         anchor="lm")
    return wpx


def flywheel_icon(d, cx, cy, r, angle, color):
    """THE PRODUCT'S MARK: the same lemniscate-in-a-wheel as the README title
    (docs/assets/render_spinner.py), so the figure does not disown it."""
    ratio, yscale = 0.52, 1.60
    deg = math.degrees(angle) % 360.0
    d.arc(S((cx - r, cy - r, cx + r, cy + r)), deg, deg + 268.0,
          fill=color, width=max(1, int(1.7 * SS)))
    a = r * 0.60
    off = (1.0 - ratio) / 2.0 * a
    rr = r * 0.085
    for i in range(180):
        tt = i / 180.0 * 2.0 * math.pi + math.pi
        dd = 1.0 + math.sin(tt) ** 2
        x, y = math.cos(tt) / dd, math.sin(tt) * math.cos(tt) / dd
        k = 1.0 if x >= 0 else ratio
        px, py = cx + x * k * a - off, cy + y * k * yscale * a
        d.ellipse(S((px - rr, py - rr, px + rr, py + rr)), fill=color)


def glow(d, box, color, a, radius=12, rings=3):
    for i in range(rings, 0, -1):
        k = a * (0.42 / i)
        if k > 0.02:
            rrect(d, (box[0] - i * 2, box[1] - i * 2, box[2] + i * 2, box[3] + i * 2),
                  radius + i * 2, outline=fade(color, k), width=1)


def station(d, box, a, accent, lit):
    """A card that fades in without moving, its accent bar lit when active."""
    if a <= 0.01:
        return
    fill = mix(fade(CARD, a), fade(mix(BG, accent, 0.10), a), lit)
    rrect(d, box, 12, fill=fill, outline=mix(fade(LINE, a), fade(accent, a), lit * .8),
          width=1)
    rrect(d, (box[0], box[1] + 16, box[0] + 3, box[3] - 16), 1.5,
          fill=fade(accent, a * (0.35 + 0.65 * lit)))
    if lit > .01:
        glow(d, box, accent, lit * .7)


def hlane(d, y, x0, x1, color, a, dashed=False, direction=1):
    """A lane crossing the boundary, its arrowhead at ``x1``."""
    if a <= .01:
        return
    c = fade(color, a)
    lo, hi = min(x0, x1), max(x0, x1)
    if dashed:
        dashed_h(d, y, lo, hi, c, width=1.6)
    else:
        d.line([S(lo), S(y), S(hi), S(y)], fill=c, width=max(1, int(1.6 * SS)))
    arrowhead(d, x1, y, c, size=7, direction=(direction, 0))


def vlane(d, x, y0, y1, color, a, draw=1.0, direction=1):
    if a <= .01 or draw <= .01:
        return
    end = y0 + (y1 - y0) * draw
    c = fade(color, a)
    d.line([S(x), S(y0), S(x), S(end)], fill=c, width=max(1, int(1.6 * SS)))
    if draw > .98:
        arrowhead(d, x, y1, c, size=7, direction=(0, direction))


def label(d, xy, s, color, a):
    x, y = xy
    if a <= .01:
        return
    w = tw(s, F_LANE) + 18
    rrect(d, (x - w / 2, y - 11, x + w / 2, y + 11), 11, fill=BG)
    text(d, (x, y + 1), s, F_LANE, fade(color, a), anchor="mm")


# ── the frame ────────────────────────────────────────────────────────────────
def draw_frame(i: int) -> Image.Image:
    t = i / FPS
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    name, t0, t1, num, verb, cap = stage_at(t)
    q = prog(t, t0, t1)
    seam = 1.0 - ease(prog(t, DUR - 0.30, DUR))   # a clean restart

    a_l = at(t, 0.0, 0.6) * seam                  # your side
    a_r = at(t, 5.8, 6.6) * seam                  # PerceptEye
    a_b = at(t, 5.6, 6.3) * seam                  # the boundary

    # ── header ──────────────────────────────────────────────────────────────
    flywheel_icon(d, 58, 42, 15, t * 1.5, fade(BLUE, .75 * seam))
    text(d, (84, 42), "the loop", F_TITLE, fade(TXT, seam), anchor="lm")
    text(d, (84 + tw("the loop", F_TITLE) + 14, 44),
         "your agent gets better from its own conversations", F_H,
         fade(DIM, seam), anchor="lm")
    idx = STAGES.index(stage_at(t))
    dx0 = W - 40 - (len(STAGES) - 1) * 17
    for k, st in enumerate(STAGES):
        live, done = k == idx, st[2] <= t
        c = BLUE if live else (mix(DOT_BASE, BLUE, .42) if done else DOT_BASE)
        cx = dx0 + k * 17
        if live:
            d.ellipse(S((cx - 7, 34, cx + 7, 48)), outline=fade(BLUE, .45 * seam),
                      width=max(1, int(SS)))
        d.ellipse(S((cx - 4, 37, cx + 4, 45)), fill=fade(c, seam))

    # ── zones ───────────────────────────────────────────────────────────────
    if a_l > .01:
        rrect(d, LZ, 14, fill=fade(ZL_FILL, a_l), outline=fade(LINE, a_l))
        text(d, (LZ[0] + 20, LZ[1] + 17), "YOUR INFRASTRUCTURE", F_ZONE,
             fade(ZL_LABEL, a_l))
    if a_r > .01:
        rrect(d, RZ, 14, fill=fade(ZR_FILL, a_r), outline=fade(ZR_EDGE, a_r))
        text(d, (RZ[0] + 20, RZ[1] + 17), "PERCEPTEYE", F_ZONE, fade(ZR_LABEL, a_r))
        text(d, (RZ[2] - 20, RZ[1] + 18), "every round visible in the dashboard",
             F_TINY, fade(DIM, a_r), anchor="ra")
    if a_b > .01:
        dashed_v(d, BOUND_X, 98, 98 + (628 - 98) * at(t, 5.6, 6.6),
                 fade(BD_DASH, a_b), width=2)
        bw = tw("TRUST BOUNDARY", F_ZONE) + 22
        rrect(d, (BOUND_X - bw / 2, 66, BOUND_X + bw / 2, 88), 11,
              fill=fade(BD_FILL, a_b), outline=fade(BD_EDGE, a_b))
        text(d, (BOUND_X, 78), "TRUST BOUNDARY", F_ZONE, fade(BD_TXT, a_b), anchor="mm")

    # ── top-left: your agent ────────────────────────────────────────────────
    lit_agent = (at(t, 2.4, 2.9) * (1 - at(t, 7.8, 8.4))
                 + at(t, 17.4, 18.0)) * seam
    station(d, AGENT, a_l, BLUE, min(1.0, lit_agent))
    if a_l > .01:
        text(d, (AGENT[0] + 24, AGENT[1] + 32), "your agent", F_BOX, fade(TXT, a_l),
             anchor="lm")
        pill(d, AGENT[0] + 24 + tw("your agent", F_BOX) + 12, AGENT[1] + 21,
             "unchanged", GREEN, .9, a=a_l, dot=False)
        text(d, (AGENT[0] + 24, AGENT[1] + 58),
             "its real conversations, recorded where it runs", F_SUB,
             fade(MUTED, a_l), anchor="lm")
        x = AGENT[0] + 24
        for k, s in enumerate(SOURCES):
            on = at(t, 2.8 + k * .55, 3.2 + k * .55) * seam
            x += pill(d, x, AGENT[1] + 80, s, BLUE, on, a=a_l) + 8
        # what the agent runs: it changes only once a person approves
        swap = at(t, 17.6, 18.3) * seam
        ry = AGENT[1] + 138
        text(d, (AGENT[0] + 24, ry), "runs", F_TINY, fade(DIM, a_l), anchor="lm")
        if swap > .5:
            k2 = (swap - .5) * 2
            rrect(d, (AGENT[0] + 70, ry - 14, AGENT[0] + 290, ry + 14), 7,
                  fill=fade(mix(CARD, GREEN, .14), a_l * k2))
            text(d, (AGENT[0] + 82, ry), "the version you approved", F_ROW,
                 fade(GREEN, a_l * k2), anchor="lm")
        else:
            text(d, (AGENT[0] + 82, ry), "your current best", F_ROW,
                 fade(MUTED, a_l * (1 - swap * 2)), anchor="lm")

    # ── the permission, on your side ────────────────────────────────────────
    allow = at(t, 6.2, 6.9) * seam
    if a_l > .01 and t >= 5.6:
        a_sw = at(t, 5.6, 6.0) * seam
        x, y = AGENT[2] - 200, AGENT[1] + 128
        rrect(d, (x, y, x + 34, y + 20), 10,
              fill=mix(fade(LINE_SOFT, a_sw), fade(GREEN, a_sw), allow))
        kx = x + 10 + 14 * allow
        d.ellipse(S((kx - 7, y + 3, kx + 7, y + 17)), fill=fade(CARD, a_sw))
        text(d, (x + 44, y + 10), "learn from its conversations", F_TINY,
             mix(fade(DIM, a_sw), fade(TXT, a_sw), allow), anchor="lm")

    # ── top-right: learn ────────────────────────────────────────────────────
    lit_learn = at(t, 8.6, 9.2) * (1 - at(t, 11.0, 11.6)) * seam
    station(d, LEARN, a_r, PURPLE, lit_learn)
    if a_r > .01:
        text(d, (LEARN[0] + 24, LEARN[1] + 32), "learn", F_BOX, fade(TXT, a_r),
             anchor="lm")
        text(d, (LEARN[0] + 24, LEARN[1] + 58),
             "a new version, from your agent's own conversations", F_SUB,
             fade(MUTED, a_r), anchor="lm")
        x = LEARN[0] + 24
        for k, s in enumerate(("what the model saw", "what it did", "how it went on")):
            on = at(t, 9.0 + k * .5, 9.4 + k * .5) * seam
            x += pill(d, x, LEARN[1] + 80, s, PURPLE, on, a=a_r) + 8
        text(d, (LEARN[0] + 24, LEARN[1] + 140),
             "personal data is found, and never learned from", F_TINY,
             fade(DIM, a_r), anchor="lm")

    # ── bottom-right: test ──────────────────────────────────────────────────
    a_test = at(t, 6.2, 7.0) * seam
    lit_test = at(t, 11.5, 12.1) * (1 - at(t, 14.0, 14.6)) * seam
    station(d, TEST, a_test, PURPLE, lit_test)
    if a_test > .01:
        text(d, (TEST[0] + 24, TEST[1] + 32), "test", F_BOX, fade(TXT, a_test),
             anchor="lm")
        text(d, (TEST[0] + 24, TEST[1] + 58),
             "on conversations it has never seen", F_SUB, fade(MUTED, a_test),
             anchor="lm")
        x = TEST[0] + 24
        x += pill(d, x, TEST[1] + 80, "your current best", DIM,
                  at(t, 11.9, 12.3) * seam, a=a_test) + 8
        text(d, (x + 4, TEST[1] + 91), "vs", F_TINY, fade(DIM, a_test), anchor="lm")
        x += 26
        pill(d, x, TEST[1] + 80, "the new version", PURPLE, at(t, 12.3, 12.7) * seam,
             a=a_test)
        better = at(t, 13.0, 13.5) * seam
        if better > .01:
            w = pill(d, TEST[0] + 24, TEST[1] + 124, "did better", GREEN, 1.0, a=better)
            text(d, (TEST[0] + 24 + w + 10, TEST[1] + 135), "only a winner goes on",
                 F_TINY, fade(DIM, better), anchor="lm")

    # ── bottom-left: Decisions ──────────────────────────────────────────────
    a_dec = at(t, 6.2, 7.0) * seam
    lit_dec = at(t, 14.8, 15.4) * (1 - at(t, 17.4, 18.0)) * seam
    station(d, DECIDE, a_dec, GREEN, lit_dec)
    if a_dec > .01:
        text(d, (DECIDE[0] + 24, DECIDE[1] + 32), "you approve the winner", F_BOX,
             fade(TXT, a_dec), anchor="lm")
        text(d, (DECIDE[0] + 24, DECIDE[1] + 58),
             "until you do, your agent keeps what it has", F_SUB,
             fade(MUTED, a_dec), anchor="lm")
        x = DECIDE[0] + 24
        x += pill(d, x, DECIDE[1] + 80, "Decisions", GREEN, .6, a=a_dec, dot=False) + 8
        x += pill(d, x, DECIDE[1] + 80, "approve", GREEN, at(t, 15.6, 16.1) * seam,
                  a=a_dec) + 8
        pill(d, x, DECIDE[1] + 80, "reject", RED, 0.0, a=a_dec)
        text(d, (DECIDE[0] + 24, DECIDE[1] + 140),
             "a person decides; a reply in a chat never approves anything",
             F_TINY, fade(DIM, a_dec), anchor="lm")

    # ── the lanes ───────────────────────────────────────────────────────────
    x0, x1 = AGENT[2] + 6, LEARN[0] - 8
    mid = (x0 + x1) / 2
    a_out = at(t, 5.8, 6.4) * seam
    hlane(d, OUT_Y, x0, x1, BLUE, a_out, dashed=allow < .99)
    label(d, (mid, OUT_Y - 24), "its conversations", BLUE, a_out)
    if a_out > .01:
        text(d, (mid, OUT_Y + 20), "once you allow it", F_TINY, fade(DIM, a_out),
             anchor="mm")
    if name == "learn" and q < .85:
        for k in range(3):
            packet(d, x0 + (x1 - x0) * (((t * .8) - k * .2) % 1), OUT_Y, BLUE,
                   alpha=seam)

    vlane(d, R_MID, LEARN[3] + 8, TEST[1] - 10, PURPLE, a_test,
          draw=at(t, 11.2, 11.8) if t < 11.8 else 1.0)
    if t >= 11.2:
        label(d, (R_MID, (LEARN[3] + TEST[1]) / 2), "a new version", PURPLE,
              at(t, 11.3, 11.8) * seam)

    xb0, xb1 = TEST[0] - 8, DECIDE[2] + 6
    a_back = at(t, 6.4, 7.0) * seam
    hlane(d, BACK_Y, xb0, xb1, PURPLE, a_back, direction=-1)
    label(d, (mid, BACK_Y - 24), "the winner", PURPLE, a_back)
    if a_back > .01:
        text(d, (mid, BACK_Y + 20), "waits for your approval", F_TINY,
             fade(DIM, a_back), anchor="mm")
    if name == "approve" and q < .55:
        packet(d, xb0 + (xb1 - xb0) * min(1.0, q / .5), BACK_Y, PURPLE,
               alpha=seam, direction=-1)

    vlane(d, L_MID, DECIDE[1] - 10, AGENT[3] + 8, GREEN, a_dec,
          draw=at(t, 17.2, 17.8) if t < 17.8 else 1.0, direction=-1)
    if t >= 17.2:
        label(d, (L_MID, (AGENT[3] + DECIDE[1]) / 2),
              "runs the winner, and repeats", GREEN, at(t, 17.3, 17.8) * seam)

    # ── footnote and caption ───────────────────────────────────────────────
    if a_r > .01:
        text(d, (RZ[2] - 20, RZ[3] - 14),
             "optional: practice tasks we write from your agent's own tools",
             F_TINY, fade(DIM, a_r), anchor="rs")
    d.line([S(40), S(632), S(W - 40), S(632)], fill=fade(HRULE, seam),
           width=max(1, int(SS)))
    intro = at(t, t0, t0 + .22)
    rrect(d, (44, 652, 66, 674), 11, fill=fade(mix(BG, BLUE, .20), seam),
          outline=fade(mix(BG, BLUE, .5), seam))
    text(d, (55, 664), num, F_CAPNUM, fade(mix(BG, BLUE, .95), seam), anchor="mm")
    text(d, (78, 663), verb, F_CAP, fade(TXT, seam * intro), anchor="lm")
    text(d, (78 + tw(verb, F_CAP) + 16, 665), cap, F_CAPSUB,
         fade(MUTED, seam * intro), anchor="lm")
    text(d, (W - 44, 664), "every connection opened from your side", F_TINY,
         fade(FOOT, seam), anchor="rm")

    d.rectangle(S((0, 0, W - 1, H - 1)), outline=FRAME_EDGE,
                width=max(1, int(1.0 * SS)))
    return img.resize((W, H), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i in range(NFRAMES):
        draw_frame(i).save(os.path.join(OUT, f"f{i:04d}.png"))
        if i % 40 == 0:
            print("frame", i, "/", NFRAMES, flush=True)
    print("done", NFRAMES)
