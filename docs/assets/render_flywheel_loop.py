"""Render the PerceptEye Agent Flywheel diagram as PNG frames.

The animation is one continuous shot in two halves. The first half is the
INTEGRATION, told in the order a developer lives it: an agent that already
exists, then `pip install`, then one import and one call, then the outbound
connection to the control plane. The second half is the LOOP that connection
carries -- claim, route, record, report, score, improve.

Design constraints:
  * Every arrow that crosses the trust boundary ORIGINATES on the customer side.
    Responses return along the same lane, dimmer and thinner -- never as an
    independent arrow drawn from the platform. That is the load-bearing visual
    claim of the whole picture and nothing may contradict it.
  * The platform is ONE box. No internal service names, no backend
    architecture. `tests/test_oss_boundary.py` scans every tracked file for
    internal system names, this one included.
    That constraint is about not disclosing OUR internals -- it is not a claim
    that the platform is opaque to the customer, who can log into the dashboard
    and watch every lap of this loop. The box is unlabelled inside; the work it
    does is not hidden. An earlier label read "opaque to you" and said the
    opposite of what is true.
  * The agent box is on screen from the first frame and never changes shape.
    The picture's claim is that integration adds things AROUND your agent; a
    box that redrew itself would contradict it.
  * Dark canvas at GitHub's own dark-theme background, so it blends on dark and
    reads as a deliberate figure on light.

Usage:
    FRAMES=/tmp/frames python render_flywheel_loop.py
    ...then encode with ffmpeg -- see render.sh alongside this file.
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

#: WHICH SDK THIS FIGURE DESCRIBES. The same diagram serves both packages, and
#: the only thing that differs is the four strings a customer would actually
#: type. Shipping the Python frames in the Node README advertised
#: ``pip install`` and ``fw.serve()`` to somebody installing a harness plugin,
#: which is worse than having no figure at all.
FLAVOUR = os.environ.get("FLAVOUR", "python")
_F = {
    "python": {
        "cmd": "pip install percepteye-agent-flywheel",
        "dep": "+ httpx   \u2190 the only runtime dependency",
        "box": "flywheel harness",
        "l1": "import percepteye_agent_flywheel as fw",
        "l2": 'fw.serve("my.agent:build")',
        "foot": "1 import  \u00b7  1 call  \u00b7  MIT",
        "prod": "fw.apply_current_policy(...)",
    },
    "node": {
        "cmd": "openclaw plugins install percepteye-agent-flywheel",
        "dep": "+ nothing   \u2190 zero runtime dependencies",
        "box": "flywheel plugin",
        "l1": "export PERCEPTEYE_AGENT_MODE=training",
        "l2": "openclaw session          # unchanged",
        "foot": "0 deps  \u00b7  0 lines changed  \u00b7  MIT",
        # NOT an apply_current_policy() equivalent. `policyCurrent()` exists on
        # the Node transport and NOTHING CALLS IT, so the plugin does not fetch
        # a trained policy -- the customer points their harness at the served
        # endpoint. Drawing a call the package does not make would be inventing
        # a capability in the one figure people trust to be literal.
        "prod": "PERCEPTEYE_AGENT_MODE=production",
    },
}[FLAVOUR]

W, H = 1280, 720
FPS = 18
DUR = 26.4
NFRAMES = int(DUR * FPS)
SS = 2  # supersample factor for crisp edges

OUT = os.environ["FRAMES"]

# ── palette ──────────────────────────────────────────────────────────────────
#: WHICH GROUND THIS FIGURE IS DRAWN ON. ``THEME=light`` (the default) is what
#: both READMEs ship: the dark original looked handsome in isolation and was
#: genuinely hard to READ inline in a README -- pale text on near-black, at
#: whatever size the page gives it, next to white page chrome. A diagram whose
#: job is to be grasped in three seconds loses that argument.
#:
#: Every accent below is DARKENED rather than merely inverted. A colour tuned to
#: glow on #0d1117 is washed out on white -- #58a6ff blue and #a371f7 purple both
#: drop under 3:1 against white -- so the light set uses deep, saturated stops
#: that stay distinguishable from each other at small sizes.
THEME = os.environ.get("THEME", "light")

_DARK = dict(
    BG=(13, 17, 23), CARD=(22, 27, 34), CARD_HI=(27, 33, 42),
    LINE=(48, 54, 61), LINE_SOFT=(33, 38, 45),
    TXT=(230, 237, 243), MUTED=(125, 133, 144), DIM=(88, 96, 105),
    BLUE=(88, 166, 255), PURPLE=(163, 113, 247), GREEN=(63, 185, 80),
    RED=(248, 81, 73), AMBER=(210, 153, 34), CYAN=(57, 197, 207),
    DOT_BASE=(44, 50, 58),
    ZL_FILL=(16, 20, 27), ZL_LABEL=(122, 132, 145),
    ZR_FILL=(18, 18, 28), ZR_EDGE=(58, 46, 84), ZR_LABEL=(150, 118, 214),
    BD_DASH=(72, 60, 96), BD_FILL=(28, 24, 40), BD_EDGE=(78, 64, 104),
    BD_TXT=(168, 142, 224),
    TERM=((72, 78, 88), (60, 66, 76), (52, 58, 68)),
    CODE=(126, 176, 226), OKTXT=(94, 160, 110), LBL=(102, 110, 122),
    CHIP_ON=(214, 196, 248),
    PF_FILL=(26, 24, 40), PF_EDGE=(96, 76, 132), PF_TITLE=(186, 156, 240),
    PF_RULE=(66, 56, 88), PF_SUB=(108, 92, 140),
    PF_ROW=(56, 50, 74), PF_ROWTX=(96, 92, 112), PF_CHIPBG=(26, 24, 40),
    RAIL=(60, 66, 75), FOOT=(128, 110, 160), HRULE=(26, 31, 38),
)

_LIGHT = dict(
    BG=(255, 255, 255), CARD=(252, 253, 254), CARD_HI=(246, 248, 251),
    LINE=(196, 205, 214), LINE_SOFT=(226, 232, 238),
    TXT=(9, 14, 20), MUTED=(74, 84, 96), DIM=(114, 124, 137),
    BLUE=(0, 82, 204), PURPLE=(88, 36, 190), GREEN=(2, 112, 48),
    RED=(186, 24, 24), AMBER=(148, 86, 0), CYAN=(0, 112, 124),
    DOT_BASE=(205, 212, 220),
    ZL_FILL=(248, 250, 252), ZL_LABEL=(88, 98, 110),
    ZR_FILL=(250, 248, 255), ZR_EDGE=(206, 192, 234), ZR_LABEL=(88, 36, 190),
    BD_DASH=(168, 148, 206), BD_FILL=(243, 238, 253), BD_EDGE=(190, 170, 224),
    BD_TXT=(88, 36, 190),
    TERM=((188, 196, 206), (202, 209, 217), (216, 222, 229)),
    CODE=(0, 82, 204), OKTXT=(2, 112, 48), LBL=(96, 105, 116),
    CHIP_ON=(88, 36, 190),
    PF_FILL=(250, 248, 255), PF_EDGE=(196, 180, 226), PF_TITLE=(80, 30, 180),
    PF_RULE=(214, 202, 236), PF_SUB=(118, 102, 152),
    PF_ROW=(226, 216, 244), PF_ROWTX=(112, 120, 132), PF_CHIPBG=(246, 242, 254),
    RAIL=(176, 184, 194), FOOT=(112, 96, 148), HRULE=(222, 227, 233),
)

_T = _LIGHT if THEME == "light" else _DARK

BG = _T['BG']
CARD = _T['CARD']
CARD_HI = _T['CARD_HI']
LINE = _T['LINE']
LINE_SOFT = _T['LINE_SOFT']
TXT = _T['TXT']
MUTED = _T['MUTED']
DIM = _T['DIM']
BLUE = _T['BLUE']
PURPLE = _T['PURPLE']
GREEN = _T['GREEN']
RED = _T['RED']
AMBER = _T['AMBER']
CYAN = _T['CYAN']
DOT_BASE = _T['DOT_BASE']
ZL_FILL = _T['ZL_FILL']
ZL_LABEL = _T['ZL_LABEL']
ZR_FILL = _T['ZR_FILL']
ZR_EDGE = _T['ZR_EDGE']
ZR_LABEL = _T['ZR_LABEL']
BD_DASH = _T['BD_DASH']
BD_FILL = _T['BD_FILL']
BD_EDGE = _T['BD_EDGE']
BD_TXT = _T['BD_TXT']
TERM = _T['TERM']
CODE = _T['CODE']
OKTXT = _T['OKTXT']
LBL = _T['LBL']
CHIP_ON = _T['CHIP_ON']
PF_FILL = _T['PF_FILL']
PF_EDGE = _T['PF_EDGE']
PF_TITLE = _T['PF_TITLE']
PF_RULE = _T['PF_RULE']
PF_SUB = _T['PF_SUB']
PF_ROW = _T['PF_ROW']
PF_ROWTX = _T['PF_ROWTX']
PF_CHIPBG = _T['PF_CHIPBG']
RAIL = _T['RAIL']
FOOT = _T['FOOT']
HRULE = _T['HRULE']

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
MONO_PATH = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, weight: str = "Regular", mono: bool = False):
    f = ImageFont.truetype(MONO_PATH if mono else FONT_PATH, size * SS)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F_TITLE   = font(24, "Semibold")
F_H       = font(14, "Semibold")
F_ZONE    = font(11, "Bold")
F_BOX     = font(17, "Semibold")
F_SUB     = font(13, "Regular")
F_TINY    = font(11, "Medium")
F_LANE    = font(12, "Semibold")
F_CAP     = font(18, "Semibold")
F_CAPSUB  = font(14, "Regular")
F_CAPNUM  = font(13, "Bold")
F_CODE    = font(13, "Regular", mono=True)
F_CODES   = font(12, "Regular", mono=True)
F_CHIP    = font(11, "Semibold")

# ── geometry (unscaled logical coords) ───────────────────────────────────────
LZ = (40, 92, 780, 628)          # your infrastructure zone
BOUND_X = 800
RZ = (828, 92, 1244, 628)        # percepteye zone

INSTALL  = (72, 134, 440, 206)
HARNESS  = (72, 226, 440, 338)
AGENT    = (72, 368, 440, 520)
PROD     = (72, 536, 440, 616)
PLATFORM = (858, 226, 1214, 600)

LANE_X0 = 440          # right edge of the customer cards
LANE_X1 = 858          # left edge of the platform box
LANE_MID = (LANE_X0 + LANE_X1) / 2

LANES = {
    "claim":  282,
    "route":  406,
    "report": 478,
    "policy": 576,
}

LANE_LABEL = {
    "claim":  ("claim a rollout", "long-poll, leased"),
    "route":  ("completions", "per-rollout endpoint"),
    "report": ("result + outcomes", "final text, artifacts"),
    "policy": ("current policy", "an endpoint you can call"),
}

# ── timeline (seconds) ───────────────────────────────────────────────────────
# (key, t0, t1, badge, verb, caption)
STAGES = [
    ("before",    0.00,  2.20, "0", "your agent today",
     "In your process, on your credentials, behind your firewall. We start here."),
    ("install",   2.20,  4.60, "1", "install",
     "One runtime dependency. Nothing to containerise, no GPU to provision."),
    ("integrate", 4.60,  7.40, "2", "integrate",
     "One import, one call. Zero lines change inside your agent."),
    ("connect",   7.40,  9.80, "3", "connect",
     "Your side dials out. No inbound port, no public DNS, no TLS to terminate."),
    ("claim",     9.80, 12.20, "4", "claim",
     "Your harness long-polls for a rollout. Nothing listens on your side."),
    ("route",    12.20, 14.90, "5", "route",
     "Your agent's completions go out through a per-rollout endpoint."),
    ("record",   14.90, 17.70, "6", "record",
     "Every tool call lands as ok / failed / unknown. Unknown is never rounded up."),
    ("report",   17.70, 19.90, "7", "report",
     "Result, artifacts and outcomes go back — a crash reports as a typed exclusion."),
    ("score",    19.90, 22.30, "8", "score & train",
     "Percepteye grades the rollout and trains the policy. This is the part you don't build."),
    ("improve",  22.30, 26.40, "9", "improve",
     "Your production agent picks up what training produced — as an endpoint it can call."),
]

# When each element of the picture arrives. (fade-in start, fade-in end)
REVEAL = {
    "zone_l":   (0.00, 0.55),
    "agent":    (0.25, 1.05),
    "install":  (2.30, 2.90),
    "harness":  (4.70, 5.30),
    "spawn":    (6.35, 7.00),
    "boundary": (7.50, 8.15),
    "zone_r":   (7.75, 8.45),
    "platform": (8.15, 8.90),
    "rails":    (8.90, 9.60),
    "footer":   (8.60, 9.40),
    "prod":     (22.40, 23.05),
}


def stage_at(t: float):
    for s in STAGES:
        if s[1] <= t < s[2]:
            return s
    return STAGES[-1]


def prog(t: float, a: float, b: float) -> float:
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    return (t - a) / (b - a)


def ease(x: float) -> float:
    return x * x * (3 - 2 * x)


def reveal(t: float, key: str) -> float:
    a, b = REVEAL[key]
    return ease(prog(t, a, b))


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(round(lerp(c1[i], c2[i], t)) for i in range(3))


# ── drawing helpers (all take LOGICAL coords, scale internally) ──────────────
def S(v):
    if isinstance(v, (tuple, list)):
        return tuple(x * SS for x in v)
    return v * SS


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
    """Logical width of a string in font f."""
    return f.getbbox(s)[2] / SS


def arrowhead(d, x, y, color, size=6, direction=1):
    p = [(x, y - size * 0.75), (x + size * direction, y), (x, y + size * 0.75)]
    d.polygon([S(pt) for pt in p], fill=color)


def packet(d, x, y, color, w=17, h=7, alpha=1.0, trail=3):
    """A data chip travelling a lane, with a fading trail behind it."""
    for i in range(trail, -1, -1):
        a = alpha * (1.0 - i / (trail + 1.4))
        if a <= 0.02:
            continue
        c = mix(BG, color, a)
        ox = x - i * (w * 0.85)
        rrect(d, (ox - w / 2, y - h / 2, ox + w / 2, y + h / 2), h / 2, fill=c)


def pill(d, x, y, label, f, color, on: float, base=CARD, pad=11, hgt=22, dot=True):
    """A labelled pill. `on` in 0..1 fades it up from the dim rail."""
    wpx = tw(label, f) + pad * 2 + (11 if dot else 0)
    rrect(d, (x, y, x + wpx, y + hgt), hgt / 2,
          fill=mix(base, mix(BG, color, 0.22), on),
          outline=mix(LINE, color, on), width=1)
    tx = x + pad
    if dot:
        d.ellipse(S((x + pad - 3, y + hgt / 2 - 2, x + pad + 1, y + hgt / 2 + 2)),
                  fill=mix(DIM, color, on))
        tx = x + pad + 9
    text(d, (tx, y + hgt / 2 + 1), label, f, mix(MUTED, TXT, on), anchor="lm")
    return wpx


def flywheel_icon(d, cx, cy, r, angle, color, spokes=6):
    d.ellipse(S((cx - r, cy - r, cx + r, cy + r)), outline=color,
              width=max(1, int(1.6 * SS)))
    for i in range(spokes):
        a = angle + i * (2 * math.pi / spokes)
        d.line([S(cx + math.cos(a) * r * 0.30), S(cy + math.sin(a) * r * 0.30),
                S(cx + math.cos(a) * r * 0.82), S(cy + math.sin(a) * r * 0.82)],
               fill=color, width=max(1, int(1.4 * SS)))
    d.ellipse(S((cx - r * 0.20, cy - r * 0.20, cx + r * 0.20, cy + r * 0.20)),
              fill=color)


def glow(d, box, color, a, radius=12, rings=3):
    """A soft focus ring around a card. Cheap, and reads as attention."""
    for i in range(rings, 0, -1):
        k = a * (0.42 / i)
        if k <= 0.02:
            continue
        rrect(d, (box[0] - i * 2, box[1] - i * 2, box[2] + i * 2, box[3] + i * 2),
              radius + i * 2, outline=mix(BG, color, k), width=1)


def card(d, box, a, accent=None, accent_a=0.0, radius=10):
    """A surface that fades in without moving. Nothing in this picture slides:
    a box that changed position would imply your agent moved, and it does not."""
    if a <= 0.01:
        return
    fill = mix(BG, CARD, a)
    edge = mix(BG, LINE, a)
    if accent is not None and accent_a > 0:
        fill = mix(fill, mix(BG, accent, 0.14), accent_a)
        edge = mix(edge, accent, accent_a * 0.85)
    rrect(d, box, radius, fill=fill, outline=edge, width=1)


def typed(s: str, p: float) -> str:
    """Reveal `s` a character at a time. Whole words look typed; whole
    characters look like a terminal, which is what this is."""
    n = round(len(s) * max(0.0, min(1.0, p)))
    return s[:n]


# ── the frame ────────────────────────────────────────────────────────────────
def draw_frame(i: int) -> Image.Image:
    t = i / FPS
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    name, t0, t1, num, verb, cap = stage_at(t)
    p = ease(prog(t, t0, t1))
    praw = prog(t, t0, t1)

    # global fade at the seam so the loop restarts clean
    seam = 1.0 - ease(prog(t, DUR - 0.30, DUR))

    a_zl = reveal(t, "zone_l") * seam
    a_ag = reveal(t, "agent") * seam
    a_in = reveal(t, "install") * seam
    a_hn = reveal(t, "harness") * seam
    a_sp = reveal(t, "spawn") * seam
    a_bd = reveal(t, "boundary") * seam
    a_zr = reveal(t, "zone_r") * seam
    a_pf = reveal(t, "platform") * seam
    a_rl = reveal(t, "rails") * seam
    a_ft = reveal(t, "footer") * seam
    a_pr = reveal(t, "prod") * seam

    # ── header ───────────────────────────────────────────────────────────────
    flywheel_icon(d, 58, 42, 15, t * 1.5, mix(BG, BLUE, 0.75 * seam))
    text(d, (84, 42), "the agent flywheel", F_TITLE, mix(BG, TXT, seam), anchor="lm")
    text(d, (84 + tw("the agent flywheel", F_TITLE) + 14, 44),
         "integrate in one call  ·  close the loop outbound",
         F_H, mix(BG, DIM, seam), anchor="lm")

    n_st = len(STAGES)
    dots_r = 4
    dx0 = W - 40 - (n_st - 1) * 17
    for k, st in enumerate(STAGES):
        live = st[0] == name
        done = st[2] <= t
        base = mix(DOT_BASE, BLUE, 0.42) if done else DOT_BASE
        c = mix(base, BLUE, 1.0 if live else 0.0)
        cx = dx0 + k * 17
        if live:
            d.ellipse(S((cx - dots_r - 3, 41 - dots_r - 3, cx + dots_r + 3, 41 + dots_r + 3)),
                      outline=mix(BG, BLUE, 0.45 * seam), width=max(1, int(1 * SS)))
        d.ellipse(S((cx - dots_r, 41 - dots_r, cx + dots_r, 41 + dots_r)),
                  fill=mix(BG, c, seam))
    idx = STAGES.index(stage_at(t))
    text(d, (dx0 - 18, 41), "integration" if idx <= 3 else "the loop",
         F_TINY, mix(BG, DIM, seam), anchor="rm")

    # ── zones ────────────────────────────────────────────────────────────────
    if a_zl > 0.01:
        rrect(d, LZ, 14, fill=mix(BG, ZL_FILL, a_zl),
              outline=mix(BG, LINE, a_zl), width=1)
        text(d, (LZ[0] + 20, LZ[1] + 17), "YOUR INFRASTRUCTURE", F_ZONE,
             mix(BG, ZL_LABEL, a_zl))
        text(d, (LZ[0] + 20 + tw("YOUR INFRASTRUCTURE", F_ZONE) + 12, LZ[1] + 18),
             "your process  ·  your credentials  ·  your firewall", F_TINY,
             mix(BG, DIM, a_zl))

    if a_zr > 0.01:
        rrect(d, RZ, 14, fill=mix(BG, ZR_FILL, a_zr),
              outline=mix(BG, ZR_EDGE, a_zr), width=1)
        text(d, (RZ[0] + 20, RZ[1] + 17), "PERCEPTEYE", F_ZONE,
             mix(BG, ZR_LABEL, a_zr))

    # ── trust boundary ───────────────────────────────────────────────────────
    if a_bd > 0.01:
        grow = prog(t, REVEAL["boundary"][0], REVEAL["boundary"][1] + 0.35)
        dashed_v(d, BOUND_X, 70, 70 + (642 - 70) * ease(grow),
                 mix(BG, BD_DASH, a_bd), dash=8, gap=7, width=2)
        bl = "TRUST BOUNDARY"
        bw = tw(bl, F_ZONE) + 22
        rrect(d, (BOUND_X - bw / 2, 40, BOUND_X + bw / 2, 62), 11,
              fill=mix(BG, BD_FILL, a_bd),
              outline=mix(BG, BD_EDGE, a_bd), width=1)
        text(d, (BOUND_X, 52), bl, F_ZONE, mix(BG, BD_TXT, a_bd), anchor="mm")

    # ── 1 · install ──────────────────────────────────────────────────────────
    if a_in > 0.01:
        live_in = name == "install"
        card(d, INSTALL, a_in, BLUE, 0.55 * p if live_in else 0.0)
        if live_in:
            glow(d, INSTALL, BLUE, p * 0.8)
        # three window dots, because a terminal reads as a terminal
        for k, cc in enumerate(TERM):
            d.ellipse(S((INSTALL[0] + 16 + k * 12, INSTALL[1] + 15,
                         INSTALL[0] + 22 + k * 12, INSTALL[1] + 21)),
                      fill=mix(BG, cc, a_in))
        cmd = _F["cmd"]
        tp = prog(t, 2.60, 4.00) if live_in else 1.0
        text(d, (INSTALL[0] + 18, INSTALL[1] + 44), "$", F_CODES,
             mix(BG, GREEN, 0.85 * a_in), anchor="lm")
        text(d, (INSTALL[0] + 31, INSTALL[1] + 44), typed(cmd, tp), F_CODES,
             mix(BG, TXT, a_in), anchor="lm")
        if live_in and tp >= 1.0 and (t * 3) % 1 < 0.6:
            cw = tw(cmd, F_CODES)
            d.rectangle(S((INSTALL[0] + 34 + cw, INSTALL[1] + 37,
                           INSTALL[0] + 40 + cw, INSTALL[1] + 51)),
                        fill=mix(BG, MUTED, 0.7 * a_in))
        ok = ease(prog(t, 4.05, 4.45)) if live_in else 1.0
        if ok > 0.01:
            text(d, (INSTALL[0] + 31, INSTALL[1] + 62),
                 _F["dep"], F_CODES,
                 mix(BG, mix(GREEN, MUTED, 0.35), ok * a_in), anchor="lm")

    # ── 2 · the harness ──────────────────────────────────────────────────────
    if a_hn > 0.01:
        live_hn = name == "integrate"
        card(d, HARNESS, a_hn, BLUE, 0.5 * p if live_hn else 0.0)
        if live_hn:
            glow(d, HARNESS, BLUE, p * 0.8)
        text(d, (HARNESS[0] + 18, HARNESS[1] + 26), _F["box"], F_BOX,
             mix(BG, TXT, a_hn), anchor="lm")
        l1 = _F["l1"]
        l2 = _F["l2"]
        p1 = prog(t, 4.85, 5.75) if live_hn else 1.0
        p2 = prog(t, 5.80, 6.45) if live_hn else 1.0
        text(d, (HARNESS[0] + 18, HARNESS[1] + 56), typed(l1, p1), F_CODE,
             mix(BG, CODE, a_hn), anchor="lm")
        text(d, (HARNESS[0] + 18, HARNESS[1] + 80), typed(l2, p2), F_CODE,
             mix(BG, CODE, a_hn), anchor="lm")
        text(d, (HARNESS[0] + 18, HARNESS[1] + 99),
             _F["foot"], F_TINY, mix(BG, DIM, a_hn), anchor="lm")

    # spawn connector, harness -> agent
    if a_sp > 0.01:
        grow = ease(prog(t, REVEAL["spawn"][0], REVEAL["spawn"][1]))
        dashed_v(d, HARNESS[0] + 30, HARNESS[3],
                 HARNESS[3] + (AGENT[1] - HARNESS[3]) * grow,
                 mix(BG, LINE, a_sp), dash=5, gap=4, width=2)
        text(d, (HARNESS[0] + 42, (HARNESS[3] + AGENT[1]) / 2), "spawns", F_TINY,
             mix(BG, DIM, a_sp), anchor="lm")

    # ── your agent: on screen from the first frame, never redrawn ────────────
    live_ag = name in ("route", "record")
    card(d, AGENT, a_ag, CYAN, 0.42 * p if live_ag else 0.0)
    if live_ag:
        glow(d, AGENT, CYAN, p * 0.7)
    text(d, (AGENT[0] + 18, AGENT[1] + 28), "your agent", F_BOX,
         mix(BG, TXT, a_ag), anchor="lm")
    # the "unchanged" claim only earns its place once the harness exists
    unch = ease(prog(t, 6.55, 7.20)) * seam
    if unch > 0.01:
        text(d, (AGENT[0] + 18 + tw("your agent", F_BOX) + 10, AGENT[1] + 29),
             "unchanged  ·  0 lines", F_TINY, mix(BG, OKTXT, unch),
             anchor="lm")
    # Two labels never crossfade in place -- one leaves before the other
    # arrives, because overlapping glyphs read as a rendering fault.
    out = 1.0 - ease(prog(t, 6.55, 6.90))
    inn = ease(prog(t, 6.95, 7.35))
    if out > 0.01:
        text(d, (AGENT[0] + 18, AGENT[1] + 52),
             "LangChain  ·  LangGraph  ·  ADK  ·  a CLI  ·  a binary", F_TINY,
             mix(BG, MUTED, a_ag * out), anchor="lm")
    if inn > 0.01:
        text(d, (AGENT[0] + 18, AGENT[1] + 52),
             "child process  ·  isolated env  ·  killable", F_TINY,
             mix(BG, MUTED, a_ag * inn), anchor="lm")

    # The beat this whole SDK exists for: before it is served, nothing has
    # observed whether your tool calls worked.
    pre = (1.0 - ease(prog(t, 7.00, 7.35))) * a_ag * seam
    if pre > 0.01:
        text(d, (AGENT[0] + 18, AGENT[1] + 86), "tool outcomes", F_ZONE,
             mix(BG, LBL, pre), anchor="lm")
        text(d, (AGENT[0] + 18, AGENT[1] + 110), "unobserved", F_SUB,
             mix(BG, DIM, pre), anchor="lm")

    # trajectory rail
    traj = ease(prog(t, 7.45, 8.05)) * seam
    if traj > 0.01:
        text(d, (AGENT[0] + 18, AGENT[1] + 86), "TRAJECTORY", F_ZONE,
             mix(BG, LBL, traj), anchor="lm")
        CHIPS = [("ok", GREEN, 15.15), ("ok", GREEN, 15.55),
                 ("unknown", AMBER, 15.95), ("failed", RED, 16.45),
                 ("ok", GREEN, 16.85)]
        cx = AGENT[0] + 18
        for label, col, at in CHIPS:
            on = ease(prog(t, at, at + 0.34))
            cx += pill(d, cx, AGENT[1] + 100, label, F_CHIP, col,
                       on * traj) + 8
        # the one callout the whole SDK exists for
        say = ease(prog(t, 16.20, 16.70)) * (1 - ease(prog(t, 17.35, 17.65)))
        if say > 0.01 and name == "record":
            text(d, (AGENT[0] + 18, AGENT[1] + 138),
                 "unknown is never rounded up to ok", F_TINY,
                 mix(BG, AMBER, 0.95 * say), anchor="lm")

    # ── 9 · your production agent ────────────────────────────────────────────
    if a_pr > 0.01:
        live_pr = name == "improve"
        card(d, PROD, a_pr, PURPLE, 0.5 * ease(prog(t, 24.05, 24.75)) if live_pr else 0.0)
        text(d, (PROD[0] + 18, PROD[1] + 25), "your production agent", F_BOX,
             mix(BG, TXT, a_pr), anchor="lm")
        text(d, (PROD[0] + 18, PROD[1] + 50), _F["prod"], F_CODE,
             mix(BG, CODE, a_pr), anchor="lm")
        ver_on = ease(prog(t, 24.15, 24.85)) * seam
        vlabel = "policy v2" if ver_on > 0.5 else "policy v1"
        vcol = PURPLE if ver_on > 0.5 else DIM
        vw = tw(vlabel, F_CHIP) + 20
        vx = PROD[2] - 18 - vw
        rrect(d, (vx, PROD[1] + 14, vx + vw, PROD[1] + 36), 11,
              fill=mix(BG, mix(CARD, mix(BG, vcol, 0.22), ver_on), a_pr),
              outline=mix(BG, mix(LINE, vcol, ver_on), a_pr), width=1)
        text(d, (vx + vw / 2, PROD[1] + 25), vlabel, F_CHIP,
             mix(BG, mix(MUTED, CHIP_ON, ver_on), a_pr), anchor="mm")
        if ver_on > 0.4:
            glow(d, PROD, PURPLE, (ver_on - 0.4) / 0.6 * 0.9)

    # ── the platform: ONE box, no internal names ─────────────────────────────
    if a_pf > 0.01:
        rrect(d, PLATFORM, 12, fill=mix(BG, PF_FILL, a_pf),
              outline=mix(BG, PF_EDGE, a_pf), width=1)
        text(d, (PLATFORM[0] + 24, PLATFORM[1] + 34), "Percepteye", F_BOX,
             mix(BG, TXT, a_pf), anchor="lm")
        text(d, (PLATFORM[0] + 24, PLATFORM[1] + 58), "control plane", F_BOX,
             mix(BG, PF_TITLE, a_pf), anchor="lm")
        dashed_h(d, PLATFORM[1] + 84, PLATFORM[0] + 24, PLATFORM[2] - 24,
                 mix(BG, PF_RULE, a_pf), width=1)
        text(d, (PLATFORM[2] - 24, PLATFORM[1] + 46), "visible in the dashboard",
             F_TINY, mix(BG, PF_SUB, a_pf), anchor="rm")

        rows = [
            (PLATFORM[1] + 130, "generates the tasks", 10.05),
            (PLATFORM[1] + 190, "scores every rollout", 20.15),
            (PLATFORM[1] + 250, "trains the policy", 21.05),
        ]
        for ry, label, at in rows:
            on = ease(prog(t, at, at + 0.55)) * a_pf * seam
            # once lit, a row stays lit -- the platform is doing all three
            rrect(d, (PLATFORM[0] + 16, ry - 17, PLATFORM[2] - 20, ry + 17), 8,
                  fill=mix(mix(BG, PF_FILL, a_pf),
                           mix(BG, PURPLE, 0.10), on),
                  outline=mix(mix(BG, PF_FILL, a_pf),
                              mix(BG, PURPLE, 0.55), on), width=1)
            d.ellipse(S((PLATFORM[0] + 32, ry - 3, PLATFORM[0] + 39, ry + 4)),
                      fill=mix(mix(BG, PF_ROW, a_pf), PURPLE, on))
            text(d, (PLATFORM[0] + 50, ry + 1), label, F_SUB,
                 mix(mix(BG, PF_ROWTX, a_pf), TXT, on), anchor="lm")

        cand = ease(prog(t, 21.60, 22.25)) * a_pf * seam
        if cand > 0.01:
            pill(d, PLATFORM[0] + 16, PLATFORM[1] + 302, "candidate policy ready",
                 F_CHIP, PURPLE, cand, base=PF_CHIPBG)

    # ── lanes ────────────────────────────────────────────────────────────────
    if a_rl > 0.01:
        for ln in LANES:
            dashed_h(d, LANES[ln], LANE_X0, LANE_X1,
                     mix(BG, LINE_SOFT, a_rl), dash=4, gap=5, width=1)

    def lane_labels(ln, active):
        # The zone fill is not the canvas colour, so text drawn AT the canvas
        # colour is a visible ghost rather than nothing. Skip, don't fade.
        if a_rl <= 0.02:
            return
        y = LANES[ln]
        a, b = LANE_LABEL[ln]
        text(d, (LANE_MID, y - 17), a, F_LANE,
             mix(BG, mix(DIM, TXT, 0.78 * active), a_rl), anchor="mm")
        text(d, (LANE_MID, y + 17), b, F_TINY,
             mix(BG, mix(RAIL, MUTED, active), a_rl), anchor="mm")

    def rail(ln, color, upto=1.0, width=1):
        y = LANES[ln]
        d.line([S(LANE_X0), S(y), S(LANE_X0 + (LANE_X1 - LANE_X0) * upto), S(y)],
               fill=color, width=max(1, int(width * SS)))
        if upto >= 0.995:
            arrowhead(d, LANE_X1 - 6, y, color, 6, 1)

    def flow(ln, phase, color, n=1, trail=3, back=False, alpha=1.0, clamp=False):
        """phase 0..1 along the lane. back=True draws the RESPONSE, which
        travels the same rail in reverse -- dimmer and thinner, so it can never
        be misread as an arrow the platform opened."""
        y = LANES[ln]
        span = LANE_X1 - LANE_X0
        if clamp and phase >= 1.0:
            n = 1  # arrived; a stack of parked packets reads as congestion
        for k in range(n):
            f = phase - k * 0.16
            # A one-shot lane holds its packet at the far end. Wrapping it back
            # to the start would draw a delivery that un-delivered itself.
            f = max(0.0, min(1.0, f)) if clamp else f % 1.0
            x = LANE_X1 - span * f if back else LANE_X0 + span * f
            packet(d, x, y, color, w=13 if back else 17, h=5 if back else 7,
                   alpha=(0.55 if back else 1.0) * alpha * seam, trail=trail)

    def note(ln, s, color, a, below=True):
        if a <= 0.02:
            return
        y = LANES[ln] + (40 if below else -40)
        text(d, (LANE_MID, y), s, F_TINY, mix(BG, color, 0.9 * a), anchor="mm")

    # 4 · claim
    if name == "claim":
        lane_labels("claim", 1.0)
        if praw < 0.55:
            q = praw / 0.55
            rail("claim", mix(BG, BLUE, 0.5 * seam), upto=q)
            flow("claim", q, BLUE, clamp=True)
        else:
            q = (praw - 0.55) / 0.45
            rail("claim", mix(BG, BLUE, 0.34 * seam), upto=1.0)
            flow("claim", min(1.0, q / 0.85), BLUE, back=True, n=2, clamp=True)
            note("claim", "← one task comes back on the connection you opened",
                 CYAN, ease(prog(q, 0.05, 0.4)))
    else:
        lane_labels("claim", 0.16 if a_rl > 0.5 else 0.0)

    # 5 · route
    if name == "route":
        lane_labels("route", 1.0)
        rail("route", mix(BG, BLUE, 0.42 * seam), upto=min(1.0, praw / 0.28))
        if praw > 0.20:
            flow("route", (t * 0.85) % 1.0, CYAN, n=3)
        note("route", "the gateway attributes the work and captures the wire",
             CYAN, ease(prog(praw, 0.42, 0.7)))
    else:
        lane_labels("route", 0.16 if a_rl > 0.5 else 0.0)

    # 7 · report
    if name == "report":
        lane_labels("report", 1.0)
        rail("report", mix(BG, GREEN, 0.38 * seam), upto=min(1.0, praw / 0.30))
        if praw > 0.22:
            flow("report", (t * 0.75) % 1.0, GREEN, n=2)
        note("report", "a rollout that crashed is still reported — and excluded, not scored",
             mix(GREEN, MUTED, 0.4), ease(prog(praw, 0.45, 0.75)))
    else:
        lane_labels("report", 0.16 if a_rl > 0.5 else 0.0)

    # 9 · policy
    if name == "improve":
        lane_labels("policy", 1.0)
        if praw < 0.34:
            q = praw / 0.34
            rail("policy", mix(BG, BLUE, 0.5 * seam), upto=q)
            flow("policy", q, BLUE, clamp=True)
            note("policy", "your agent asks — outbound, like everything else",
                 BLUE, ease(prog(q, 0.25, 0.6)))
        else:
            q = (praw - 0.34) / 0.66
            rail("policy", mix(BG, PURPLE, 0.34 * seam), upto=1.0)
            flow("policy", min(1.0, q / 0.72), PURPLE, back=True, n=2, clamp=True)
            note("policy", "← the better policy, as an endpoint you can call",
                 PURPLE, ease(prog(q, 0.12, 0.45)))
    else:
        lane_labels("policy", 0.16 if a_rl > 0.5 else 0.0)

    # ── footer: the claim the lanes are all making ───────────────────────────
    if a_ft > 0.01:
        text(d, (W - 44, 674),
             "outbound only  ·  no inbound port  ·  no public DNS", F_TINY,
             mix(BG, FOOT, a_ft * seam), anchor="rm")

    # ── caption strip ────────────────────────────────────────────────────────
    d.line([S(40), S(646), S(W - 40), S(646)], fill=mix(BG, HRULE, seam),
           width=max(1, int(1 * SS)))
    cnx = 44
    intro = ease(prog(t, t0, t0 + 0.22))
    rrect(d, (cnx, 662, cnx + 22, 684), 11,
          fill=mix(BG, mix(BG, BLUE, 0.20), seam),
          outline=mix(BG, mix(BG, BLUE, 0.5), seam), width=1)
    text(d, (cnx + 11, 674), num, F_CAPNUM, mix(BG, mix(BG, BLUE, 0.95), seam),
         anchor="mm")
    text(d, (cnx + 34, 673), verb, F_CAP, mix(BG, TXT, seam * intro), anchor="lm")
    text(d, (cnx + 34 + tw(verb, F_CAP) + 16, 674), cap, F_CAPSUB,
         mix(BG, MUTED, seam * intro), anchor="lm")

    return img.resize((W, H), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i in range(NFRAMES):
        draw_frame(i).save(os.path.join(OUT, f"f{i:04d}.png"))
        if i % 40 == 0:
            print("frame", i, "/", NFRAMES, flush=True)
    print("done", NFRAMES)
