"""Render the Percepteye Agent Flywheel OVERVIEW as PNG frames.

This is the customer's story, and it has ONE subject: their agent. The whole
animation happens *to* a single card. It is wrapped, it is observed, it sends
evidence out, and at the end one row of it changes -- which is the entire
product, in one beat.

Deliberately NOT an architecture drawing. ``render_flywheel_loop.py`` is that,
and it earns its four lanes and its trust zones. Here there are two objects and
two flows, drawn large, because a reader who has not decided to care yet will
not count lanes.

What the returning token is allowed to say, and why it says exactly that:
  ``/policy/current`` is read for five things and no others -- ``ttl_s``,
  ``serving_generation``, and ``base_url`` / ``api_key`` / ``model`` through
  ``endpoint_kwargs`` (see ``PolicySource._absorb``). A TRAINED MODEL therefore
  arrives, as an endpoint. A PROMPT does not: ``PolicySnapshot`` says so in its
  own docstring -- the system prompt is the thing the snapshot "does not carry",
  and ``serving_generation`` exists precisely so a fleet can notice it moved and
  cycle a replica. So the token names the endpoint, and names the generation as
  the signal that the served configuration moved. Drawing prompt text arriving
  on this wire would be a lie told in a picture, which is the hardest kind to
  correct later.

There is no score anywhere in this figure, and no bar chart. The repository
ships no benchmark numbers; a diagram is the easiest place to imply one nobody
measured. The superiority claim it does make is about PROVENANCE -- the policy
was trained on this agent's own observed outcomes -- which is true, checkable,
and the actual differentiator.

Usage:
    FRAMES=/tmp/frames python render_overview.py
    ...then encode with ffmpeg -- see render.sh alongside this file.
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

#: WHICH SDK THIS FIGURE DESCRIBES -- see render_flywheel_loop.py for why the
#: same picture cannot carry Python code into the Node README.
FLAVOUR = os.environ.get("FLAVOUR", "python")
_WRAP_TITLE = {"python": "one call", "node": "one plugin"}[FLAVOUR]
_WRAP_BODY = {
    "python": "fw.serve() wraps it where it already runs. Nothing inside it changes.",
    "node": "The plugin loads beside it in your harness. Nothing inside it changes.",
}[FLAVOUR]
_WRAP_LABEL = {
    "python": 'fw.serve("my.agent:build")',
    "node": "percepteye-agent-flywheel",
}[FLAVOUR]
#: The harnesses THIS package actually supports. The Python list in the Node
#: figure advertised LangChain and ADK to an OpenClaw customer.
_HARNESS_CHIP = {
    "python": "LangChain \u00b7 ADK \u00b7 a CLI",
    "node": "OpenClaw \u00b7 DeepSeek Harness",
}[FLAVOUR]

W, H = 1280, 720
FPS = 20
DUR = 23.4
NFRAMES = int(DUR * FPS)
SS = 2

OUT = os.environ["FRAMES"]

# ── palette ──────────────────────────────────────────────────────────────────
BG    = (13, 17, 23)
CARD  = (22, 27, 34)
LINE  = (48, 54, 61)
SOFT  = (33, 38, 45)
TXT   = (230, 237, 243)
MUTED = (125, 133, 144)
DIM   = (88, 96, 105)

BLUE   = (88, 166, 255)
CYAN   = (57, 197, 207)
PURPLE = (163, 113, 247)
GREEN  = (63, 185, 80)
RED    = (248, 81, 73)
AMBER  = (210, 153, 34)

FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size, weight="Regular", mono=False):
    f = ImageFont.truetype(MONO if mono else FONT, size * SS)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F_TITLE = font(25, "Semibold")
F_SUB   = font(14, "Semibold")
F_CARD  = font(21, "Semibold")
F_ROWK  = font(13, "Medium")
F_ROWV  = font(14, "Regular", mono=True)
F_BODY  = font(14, "Regular")
F_TINY  = font(11, "Medium")
F_ZONE  = font(11, "Bold")
F_CHIP  = font(11, "Semibold")
F_CAP   = font(19, "Semibold")
F_CAPS  = font(15, "Regular")
F_NUM   = font(13, "Bold")
F_CODE  = font(14, "Regular", mono=True)

# ── geometry ─────────────────────────────────────────────────────────────────
LZ = (40, 100, 604, 600)
BOUND_X = 662
RZ = (720, 100, 1240, 600)

AGENT = (74, 192, 540, 512)
PLAT = (792, 196, 1212, 436)
BA_Y = 548                      # the before/after lines, no box around them

CH_OUT, CH_BACK = 268, 372
CH_X0, CH_X1 = 568, 792   # outside the wrap ring, not under it

# ── beats ────────────────────────────────────────────────────────────────────
BEATS = [
    ("today",    0.00,  3.30, "1", "your agent, today",
     "Your prompt, your tools, your model, your credentials."),
    ("wrap",     3.30,  6.40, "2", _WRAP_TITLE, _WRAP_BODY),
    ("connect",  6.40,  8.90, "3", "outbound only",
     "Your side opens the connection. No inbound port, no container, no GPU."),
    ("work",     8.90, 12.60, "4", "it does the work",
     "Real tasks, in your process. Every tool call observed: ok, failed, unknown."),
    ("train",   12.60, 15.90, "5", "evidence, not transcripts",
     "Percepteye scores what your agent actually did, and trains on it."),
    ("deliver", 15.90, 19.10, "6", "a trained policy comes home",
     "base_url, model, and a marker that tells you when the served config moved."),
    ("better",  19.10, 23.40, "7", "same agent. same code. better policy.",
     "Trained on your agent's own observed work — not on a transcript that guessed."),
]

REVEAL = {
    "agent": (0.15, 1.10), "rows": (1.10, 2.20), "wrap": (3.45, 4.30),
    "bound": (6.55, 7.30), "rzone": (6.80, 7.60), "plat": (7.15, 8.00),
    "chan": (8.05, 8.70), "traj": (9.10, 9.70), "strip": (19.90, 20.80),
}

CHIPS = [("ok", GREEN, 9.90), ("ok", GREEN, 10.35), ("unknown", AMBER, 10.80),
         ("failed", RED, 11.30), ("ok", GREEN, 11.75)]

PLAT_ROWS = [("generates the tasks", 13.00),
             ("scores observed outcomes", 13.70),
             ("trains the policy", 14.40)]

#: What actually comes back -- the three fields ``_absorb`` reads and hands on.
#: Not a prompt: ``PolicySnapshot`` does not carry one.
TOKEN_LINE = "base_url  ·  model  ·  generation"


def beat_at(t):
    for b in BEATS:
        if b[1] <= t < b[2]:
            return b
    return BEATS[-1]


def prog(t, a, b):
    return 0.0 if t <= a else 1.0 if t >= b else (t - a) / (b - a)


def ease(x):
    return x * x * (3 - 2 * x)


def rev(t, k):
    return ease(prog(t, *REVEAL[k]))


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(round(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def fade(c, a):
    return mix(BG, c, a)


# ── helpers ──────────────────────────────────────────────────────────────────
def S(v):
    return tuple(x * SS for x in v) if isinstance(v, (tuple, list)) else v * SS


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(S(box), radius=r * SS, fill=fill, outline=outline,
                        width=max(1, int(width * SS)))


def dashed_v(d, x, y0, y1, c, dash=8, gap=7, w=1):
    y = y0
    while y < y1:
        d.line([S(x), S(y), S(x), S(min(y + dash, y1))], fill=c,
               width=max(1, int(w * SS)))
        y += dash + gap


def dashed_h(d, y, x0, x1, c, dash=5, gap=5, w=1):
    x = x0
    while x < x1:
        d.line([S(x), S(y), S(min(x + dash, x1)), S(y)], fill=c,
               width=max(1, int(w * SS)))
        x += dash + gap


def text(d, xy, s, f, c, anchor="la"):
    d.text(S(xy), s, font=f, fill=c, anchor=anchor)


def tw(s, f):
    return f.getbbox(s)[2] / SS


def arrow(d, x, y, c, size=6, direction=1):
    d.polygon([S((x, y - size * .75)), S((x + size * direction, y)),
               S((x, y + size * .75))], fill=c)


def glow(d, box, c, a, r=12, rings=4):
    for i in range(rings, 0, -1):
        k = a * (0.40 / i)
        if k > 0.02:
            rrect(d, (box[0] - i * 2.5, box[1] - i * 2.5, box[2] + i * 2.5,
                      box[3] + i * 2.5), r + i * 2.5, outline=fade(c, k), width=1)


def pill(d, x, y, s, f, c, on, base=CARD, h=22, pad=10, dot=True):
    w = tw(s, f) + pad * 2 + (10 if dot else 0)
    rrect(d, (x, y, x + w, y + h), h / 2, fill=mix(base, mix(BG, c, .22), on),
          outline=mix(LINE, c, on), width=1)
    tx = x + pad
    if dot:
        d.ellipse(S((x + pad - 2, y + h / 2 - 2, x + pad + 2, y + h / 2 + 2)),
                  fill=mix(DIM, c, on))
        tx = x + pad + 9
    text(d, (tx, y + h / 2 + 1), s, f, mix(MUTED, TXT, on), anchor="lm")
    return w


def packet(d, x, y, c, w=18, h=7, a=1.0, trail=3):
    for i in range(trail, -1, -1):
        k = a * (1 - i / (trail + 1.4))
        if k > 0.02:
            ox = x - i * (w * .85)
            rrect(d, (ox - w / 2, y - h / 2, ox + w / 2, y + h / 2), h / 2, fill=fade(c, k))


def wheel(d, cx, cy, r, ang, c, spokes=6):
    d.ellipse(S((cx - r, cy - r, cx + r, cy + r)), outline=c, width=max(1, int(1.6 * SS)))
    for i in range(spokes):
        a = ang + i * (2 * math.pi / spokes)
        d.line([S(cx + math.cos(a) * r * .30), S(cy + math.sin(a) * r * .30),
                S(cx + math.cos(a) * r * .82), S(cy + math.sin(a) * r * .82)],
               fill=c, width=max(1, int(1.4 * SS)))
    d.ellipse(S((cx - r * .2, cy - r * .2, cx + r * .2, cy + r * .2)), fill=c)


# ── the frame ────────────────────────────────────────────────────────────────
def draw_frame(i):
    t = i / FPS
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    name, t0, t1, num, verb, cap = beat_at(t)
    p = ease(prog(t, t0, t1))
    seam = 1.0 - ease(prog(t, DUR - .32, DUR))

    a_ag = rev(t, "agent") * seam
    a_rw = rev(t, "rows") * seam
    a_wr = rev(t, "wrap") * seam
    a_bd = rev(t, "bound") * seam
    a_rz = rev(t, "rzone") * seam
    a_pl = rev(t, "plat") * seam
    a_ch = rev(t, "chan") * seam
    a_tj = rev(t, "traj") * seam
    a_st = rev(t, "strip") * seam

    # ── header ──
    wheel(d, 56, 40, 15, t * 1.6, fade(BLUE, .75 * seam))
    text(d, (82, 40), "the agent flywheel", F_TITLE, fade(TXT, seam), anchor="lm")
    text(d, (82 + tw("the agent flywheel", F_TITLE) + 14, 42),
         "your agent, made better by its own work", F_SUB, fade(DIM, seam), anchor="lm")
    idx = BEATS.index(beat_at(t))
    x0 = W - 40 - (len(BEATS) - 1) * 17
    for k, b in enumerate(BEATS):
        live, done = k == idx, b[2] <= t
        c = BLUE if live else (mix((44, 50, 58), BLUE, .40) if done else (44, 50, 58))
        cx = x0 + k * 17
        if live:
            d.ellipse(S((cx - 7, 34, cx + 7, 48)), outline=fade(BLUE, .45 * seam),
                      width=max(1, int(SS)))
        d.ellipse(S((cx - 4, 37, cx + 4, 45)), fill=fade(c, seam))

    # ── zones ──
    if a_ag > .01:
        rrect(d, LZ, 14, fill=fade((16, 20, 27), a_ag), outline=fade(LINE, a_ag), width=1)
        text(d, (LZ[0] + 20, LZ[1] + 16), "YOUR INFRASTRUCTURE", F_ZONE,
             fade((122, 132, 145), a_ag))
    if a_rz > .01:
        rrect(d, RZ, 14, fill=fade((18, 18, 28), a_rz),
              outline=fade((58, 46, 84), a_rz), width=1)
        text(d, (RZ[0] + 20, RZ[1] + 16), "PERCEPTEYE", F_ZONE,
             fade((150, 118, 214), a_rz))
    if a_bd > .01:
        g = ease(prog(t, REVEAL["bound"][0], REVEAL["bound"][1] + .3))
        dashed_v(d, BOUND_X, 76, 76 + (612 - 76) * g, fade((72, 60, 96), a_bd), w=2)
        bw = tw("TRUST BOUNDARY", F_ZONE) + 22
        rrect(d, (BOUND_X - bw / 2, 46, BOUND_X + bw / 2, 68), 11,
              fill=fade((28, 24, 40), a_bd), outline=fade((78, 64, 104), a_bd), width=1)
        text(d, (BOUND_X, 58), "TRUST BOUNDARY", F_ZONE, fade((168, 142, 224), a_bd),
             anchor="mm")

    # ── 2 · the wrap: fw.serve() goes AROUND the card, never inside it ──
    if a_wr > .01:
        g = ease(prog(t, 3.45, 4.55))
        b = (AGENT[0] - 16, AGENT[1] - 16, AGENT[2] + 16, AGENT[3] + 16)
        rrect(d, b, 18, outline=fade(BLUE, .55 * a_wr), width=2)
        lab = _WRAP_LABEL
        lw = tw(lab, F_CODE) + 26
        rrect(d, (b[0] + 20, b[1] - 15, b[0] + 20 + lw, b[1] + 15), 15,
              fill=fade((18, 26, 38), a_wr), outline=fade(BLUE, .55 * a_wr), width=2)
        text(d, (b[0] + 33, b[1] + 1), lab[:max(0, int(len(lab) * g))], F_CODE,
             fade((150, 195, 245), a_wr), anchor="lm")
        z = ease(prog(t, 4.85, 5.55))
        if z > .01:
            w = tw("0 lines changed inside", F_CHIP) + 30
            pill(d, b[2] - 20 - w, b[3] - 14, "0 lines changed inside", F_CHIP, GREEN,
                 z * a_wr, base=(18, 26, 38))

    # ── THE SUBJECT: one card, and everything happens to it ──
    live = name in ("work", "better")
    fill = fade(CARD, a_ag)
    if name == "better":
        fill = mix(fill, fade(mix(CARD, PURPLE, .10), a_ag), p)
    rrect(d, AGENT, 12, fill=fill, outline=fade(LINE, a_ag), width=1)
    if live:
        glow(d, AGENT, PURPLE if name == "better" else CYAN, p * .75)

    text(d, (AGENT[0] + 22, AGENT[1] + 32), "your agent", F_CARD, fade(TXT, a_ag),
         anchor="lm")
    fw_on = ease(prog(t, .60, 1.30)) * seam
    if fw_on > .01:
        pill(d, AGENT[0] + 22 + tw("your agent", F_CARD) + 14, AGENT[1] + 21,
             _HARNESS_CHIP, F_CHIP, DIM, fw_on, dot=False)
    text(d, (AGENT[0] + 22, AGENT[1] + 58), "runs where it already runs", F_TINY,
         fade(MUTED, a_ag), anchor="lm")
    dashed_h(d, AGENT[1] + 80, AGENT[0] + 22, AGENT[2] - 22, fade(SOFT, a_ag))

    # three rows. Only the third ever changes -- that is the product.
    swap = ease(prog(t, 17.55, 18.25)) * seam
    rows = [("system prompt", "yours, unchanged", None),
            ("tools", "yours, unchanged", None),
            ("model", "your-model", "policy · gw.percepteye.ai/r/…")]
    for k, (key, val, newval) in enumerate(rows):
        ry = AGENT[1] + 112 + k * 40
        text(d, (AGENT[0] + 22, ry), key, F_ROWK, fade(DIM, a_rw), anchor="lm")
        if newval is None:
            text(d, (AGENT[0] + 168, ry), val, F_ROWV, fade(MUTED, a_rw), anchor="lm")
            continue
        # the upgrade beat: sweep, then replace -- never two strings crossfading
        # in one place, which reads as a rendering fault rather than a change
        if swap > .5:
            k2 = (swap - .5) * 2
            rrect(d, (AGENT[0] + 160, ry - 16, AGENT[2] - 18, ry + 16), 8,
                  fill=fade(mix(CARD, PURPLE, .16), a_rw * k2))
            text(d, (AGENT[0] + 168, ry), newval, F_ROWV,
                 fade((214, 196, 248), a_rw * k2), anchor="lm")
        else:
            text(d, (AGENT[0] + 168, ry), val, F_ROWV,
                 fade(MUTED, a_rw * max(0.0, 1 - swap * 2)), anchor="lm")
        if .02 < swap < .62:
            sx = AGENT[0] + 160 + (AGENT[2] - 178 - AGENT[0]) * (swap / .62)
            d.rectangle(S((sx - 3, ry - 16, sx + 3, ry + 16)), fill=fade(PURPLE, .85 * seam))

    # what the SDK observed, on the same card
    if a_tj > .01:
        text(d, (AGENT[0] + 22, AGENT[1] + 244), "OBSERVED THIS RUN", F_ZONE,
             fade((102, 110, 122), a_tj), anchor="lm")
        cx = AGENT[0] + 22
        for label, col, at in CHIPS:
            cx += pill(d, cx, AGENT[1] + 258, label, F_CHIP, col,
                       ease(prog(t, at, at + .32)) * a_tj) + 8
        say = ease(prog(t, 11.95, 12.35)) * (1 - ease(prog(t, 15.60, 15.90)))
        if say > .01:
            text(d, (AGENT[0] + 22, AGENT[1] + 296),
                 "unknown is never rounded up to ok", F_TINY,
                 fade(AMBER, .95 * say * seam), anchor="lm")

    # ── the other party: one box, no internal names ──
    if a_pl > .01:
        rrect(d, PLAT, 12, fill=fade((26, 24, 40), a_pl),
              outline=fade((96, 76, 132), a_pl), width=1)
        text(d, (PLAT[0] + 24, PLAT[1] + 34), "Percepteye", F_CARD, fade(TXT, a_pl),
             anchor="lm")
        text(d, (PLAT[0] + 24, PLAT[1] + 60), "control plane", F_CARD,
             fade((186, 156, 240), a_pl), anchor="lm")
        text(d, (PLAT[2] - 24, PLAT[1] + 47), "visible in the dashboard", F_TINY,
             fade((108, 92, 140), a_pl), anchor="rm")
        dashed_h(d, PLAT[1] + 86, PLAT[0] + 24, PLAT[2] - 24, fade((66, 56, 88), a_pl))
        for k, (label, at) in enumerate(PLAT_ROWS):
            on = ease(prog(t, at, at + .5)) * a_pl
            ry = PLAT[1] + 122 + k * 42
            d.ellipse(S((PLAT[0] + 28, ry - 4, PLAT[0] + 36, ry + 4)),
                      fill=mix(fade((56, 50, 74), a_pl), PURPLE, on))
            text(d, (PLAT[0] + 48, ry + 1), label, F_BODY,
                 mix(fade((96, 92, 112), a_pl), TXT, on), anchor="lm")

    # ── two flows, both opened from the left ──
    if a_ch > .01:
        for y, lab, sub in ((CH_OUT, "rollouts + observed outcomes", None),
                            (CH_BACK, "the trained policy", TOKEN_LINE)):
            dashed_h(d, y, CH_X0, CH_X1, fade(SOFT, a_ch), dash=4, gap=5)
            text(d, ((CH_X0 + CH_X1) / 2, y - 18), lab, F_TINY, fade(DIM, a_ch),
                 anchor="mm")
            if sub:
                on = ease(prog(t, 16.85, 17.55)) * seam
                if on > .01:
                    text(d, ((CH_X0 + CH_X1) / 2, y + 18), sub, F_TINY,
                         fade((186, 156, 240), on), anchor="mm")

    def rail(y, c, upto=1.0):
        d.line([S(CH_X0), S(y), S(CH_X0 + (CH_X1 - CH_X0) * upto), S(y)], fill=c,
               width=max(1, int(SS)))
        if upto > .99:
            arrow(d, CH_X1 - 6, y, c)

    if t >= BEATS[3][1] and name not in ("work", "train"):
        rail(CH_OUT, fade(SOFT, .9 * seam))
    if t >= BEATS[5][1] and name not in ("deliver", "better"):
        rail(CH_BACK, fade(SOFT, .9 * seam))

    if name == "work":
        q = prog(t, t0, t1)
        rail(CH_OUT, fade(BLUE, .34 * seam), min(1, q / .25))
        if q > .2:
            packet(d, CH_X1 - (CH_X1 - CH_X0) * ((t * .75) % 1), CH_BACK, CYAN,
                   w=14, h=5, a=.6 * seam, trail=2)
            text(d, ((CH_X0 + CH_X1) / 2, CH_BACK + 18),
                 "← a task, on your connection", F_TINY, fade(CYAN, .8 * seam),
                 anchor="mm")
    elif name == "train":
        q = prog(t, t0, t1)
        rail(CH_OUT, fade(GREEN, .40 * seam), min(1, q / .2))
        if q > .15:
            for k in range(3):
                packet(d, CH_X0 + (CH_X1 - CH_X0) * (((t * .8) - k * .17) % 1),
                       CH_OUT, GREEN, a=seam)
    elif name in ("deliver", "better"):
        q = prog(t, BEATS[5][1], BEATS[5][2])
        rail(CH_BACK, fade(PURPLE, .34 * seam), 1.0)
        if name == "deliver":
            packet(d, CH_X1 - (CH_X1 - CH_X0) * min(1, q / .7), CH_BACK, PURPLE,
                   a=seam, trail=3)
    # ── the payoff, stated as provenance rather than as a score ──
    if a_st > .01:
        text(d, (AGENT[0], BA_Y), "BEFORE", F_ZONE, fade(DIM, a_st), anchor="lm")
        text(d, (AGENT[0] + 74, BA_Y), "your prompt  ·  a general-purpose model",
             F_TINY, fade(MUTED, a_st), anchor="lm")
        text(d, (AGENT[0], BA_Y + 24), "AFTER", F_ZONE, fade((150, 118, 214), a_st),
             anchor="lm")
        text(d, (AGENT[0] + 74, BA_Y + 24),
             "your prompt  ·  a policy trained on this agent's own work", F_TINY,
             fade((196, 176, 232), a_st), anchor="lm")

    # ── caption ──
    d.line([S(40), S(626), S(W - 40), S(626)], fill=fade((26, 31, 38), seam),
           width=max(1, int(SS)))
    intro = ease(prog(t, t0, t0 + .22))
    rrect(d, (44, 646, 66, 668), 11, fill=fade(mix(BG, BLUE, .20), seam),
          outline=fade(mix(BG, BLUE, .5), seam), width=1)
    text(d, (55, 658), num, F_NUM, fade(mix(BG, BLUE, .95), seam), anchor="mm")
    text(d, (78, 657), verb, F_CAP, fade(TXT, seam * intro), anchor="lm")
    text(d, (78 + tw(verb, F_CAP) + 16, 659), cap, F_CAPS, fade(MUTED, seam * intro),
         anchor="lm")
    if a_bd > .01:
        text(d, (W - 44, 658), "outbound only  ·  no inbound port  ·  no GPU",
             F_TINY, fade((128, 110, 160), a_bd), anchor="rm")

    return img.resize((W, H), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i in range(NFRAMES):
        draw_frame(i).save(os.path.join(OUT, f"f{i:04d}.png"))
        if i % 40 == 0:
            print("frame", i, "/", NFRAMES, flush=True)
    print("done", NFRAMES)
