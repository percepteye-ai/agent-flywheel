"""Render the PerceptEye Agent Flywheel loop diagram as PNG frames.

Design constraints:
  * Every arrow that crosses the trust boundary ORIGINATES on the customer side.
    Responses return along the same lane, dimmer and thinner -- never as an
    independent arrow drawn from the platform. That is the load-bearing visual
    claim of the whole picture and nothing may contradict it.
  * The platform is ONE opaque box. No internal service names, no backend
    architecture.
  * Dark canvas at GitHub's own dark-theme background, so it blends on dark and
    reads as a deliberate figure on light.
"""
from __future__ import annotations

import math
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 660
FPS = 20
DUR = 16.0
NFRAMES = int(DUR * FPS)
SS = 2  # supersample factor for crisp edges

OUT = os.environ["FRAMES"]

# ── palette ──────────────────────────────────────────────────────────────────
BG        = (13, 17, 23)        # #0d1117 -- GitHub dark canvas
CARD      = (22, 27, 34)        # #161b22
LINE      = (48, 54, 61)        # #30363d
LINE_SOFT = (33, 38, 45)
TXT       = (230, 237, 243)     # #e6edf3
MUTED     = (125, 133, 144)
DIM       = (88, 96, 105)

BLUE      = (88, 166, 255)      # #58a6ff  request / your side
BLUE_DIM  = (37, 74, 120)
PURPLE    = (163, 113, 247)     # #a371f7  platform
PURPLE_DIM= (72, 51, 110)
GREEN     = (63, 185, 80)       # ok
RED       = (248, 81, 73)       # failed
AMBER     = (210, 153, 34)      # unknown  <- deliberately NOT green
CYAN      = (57, 197, 207)

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
MONO_PATH = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, weight: str = "Regular", mono: bool = False):
    f = ImageFont.truetype(MONO_PATH if mono else FONT_PATH, size * SS)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F_TITLE   = font(23, "Semibold")
F_H       = font(14, "Semibold")
F_ZONE    = font(11, "Bold")
F_BOX     = font(17, "Semibold")
F_SUB     = font(13, "Regular")
F_TINY    = font(11, "Medium")
F_LANE    = font(12, "Semibold")
F_CAP     = font(17, "Medium")
F_CAPNUM  = font(13, "Bold")
F_CODE    = font(13, "Regular", mono=True)
F_CHIP    = font(11, "Semibold")

# ── geometry (unscaled logical coords) ───────────────────────────────────────
LZ = (40, 84, 700, 566)          # your infrastructure zone
BOUND_X = 730
RZ = (760, 84, 1160, 566)        # percepteye zone

HARNESS  = (72, 132, 372, 214)
AGENT    = (72, 250, 372, 404)
PROD     = (72, 452, 372, 534)
PLATFORM = (792, 150, 1128, 500)

LANE_X0 = 372          # right edge of the customer boxes
LANE_X1 = 792          # left edge of the platform box

LANES = {
    "claim":  173,
    "route":  300,
    "report": 372,
    "policy": 493,
}

# ── timeline (seconds) ───────────────────────────────────────────────────────
STAGES = [
    ("claim",  0.00,  2.30, "1", "claim",   "Your harness long-polls for a rollout. Nothing listens on your side."),
    ("spawn",  2.30,  3.70, "2", "isolate", "One child process per rollout — its own env, its own single-use key."),
    ("route",  3.70,  6.60, "3", "route",   "Your agent's completions go out through a per-rollout endpoint."),
    ("record", 6.60,  9.10, "4", "record",  "Every tool call lands as ok / failed / unknown. Unknown is never rounded up."),
    ("report", 9.10, 11.10, "5", "report",  "Final text, artifacts and outcomes are reported back."),
    ("score", 11.10, 13.10, "6", "score",   "Percepteye grades the rollout against its own compiled checks."),
    ("policy",13.10, 16.00, "7", "improve", "Your production agent picks up what training produced."),
]


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


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(lerp(c1[i], c2[i], t))) for i in range(3))


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


def packet(d, x, y, color, w=17, h=7, alpha=1.0, trail=0):
    """A data chip travelling a lane, with a fading trail behind it."""
    for i in range(trail, -1, -1):
        a = alpha * (1.0 - i / (trail + 1.4))
        if a <= 0.02:
            continue
        c = mix(BG, color, a)
        ox = x - i * (w * 0.85)
        rrect(d, (ox - w / 2, y - h / 2, ox + w / 2, y + h / 2), h / 2, fill=c)


def chip(d, x, y, label, color, on: float):
    """A tri-state ledger chip. `on` in 0..1 fades it in from the dim rail."""
    wpx = tw(label, F_CHIP) + 22
    body = mix(CARD, mix(BG, color, 0.22), on)
    edge = mix(LINE, color, on)
    rrect(d, (x, y, x + wpx, y + 22), 11, fill=body, outline=edge, width=1)
    d.ellipse(S((x + 8, y + 9, x + 12, y + 13)), fill=mix(DIM, color, on))
    text(d, (x + 17, y + 11), label, F_CHIP, mix(MUTED, TXT, on), anchor="lm")
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


# ── static background ────────────────────────────────────────────────────────
def build_base() -> Image.Image:
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)

    # zones
    rrect(d, LZ, 14, fill=(16, 20, 27), outline=LINE, width=1)
    rrect(d, RZ, 14, fill=(18, 18, 28), outline=(58, 46, 84), width=1)

    text(d, (LZ[0] + 20, LZ[1] + 17), "YOUR INFRASTRUCTURE", F_ZONE, (122, 132, 145))
    text(d, (LZ[0] + 20 + tw("YOUR INFRASTRUCTURE", F_ZONE) + 12, LZ[1] + 18),
         "your process  ·  your credentials  ·  your firewall", F_TINY, DIM)
    text(d, (RZ[0] + 20, RZ[1] + 17), "PERCEPTEYE", F_ZONE, (150, 118, 214))

    # trust boundary
    dashed_v(d, BOUND_X, 62, 600, (72, 60, 96), dash=8, gap=7, width=2)
    bl = "TRUST BOUNDARY"
    bw = tw(bl, F_ZONE) + 22
    rrect(d, (BOUND_X - bw / 2, 40, BOUND_X + bw / 2, 62), 11,
          fill=(28, 24, 40), outline=(78, 64, 104), width=1)
    text(d, (BOUND_X, 52), bl, F_ZONE, (168, 142, 224), anchor="mm")
    text(d, (W - 44, 605), "every connection opens from your side", F_TINY,
         (128, 110, 160), anchor="rm")
    text(d, (W - 44, 621), "no inbound port  ·  no public DNS  ·  no TLS to terminate",
         F_TINY, (86, 76, 106), anchor="rm")

    # ── customer boxes ──
    rrect(d, HARNESS, 10, fill=CARD, outline=LINE, width=1)
    text(d, (HARNESS[0] + 18, HARNESS[1] + 24), "flywheel harness", F_BOX, TXT, anchor="lm")
    text(d, (HARNESS[0] + 18, HARNESS[1] + 50), "fw.serve(\"my.agent:build\")", F_CODE,
         (126, 176, 226), anchor="lm")
    text(d, (HARNESS[0] + 18, HARNESS[1] + 68), "one dependency  ·  MIT", F_TINY, DIM, anchor="lm")

    rrect(d, AGENT, 10, fill=CARD, outline=LINE, width=1)
    text(d, (AGENT[0] + 18, AGENT[1] + 26), "your agent", F_BOX, TXT, anchor="lm")
    text(d, (AGENT[0] + 18 + tw("your agent", F_BOX) + 10, AGENT[1] + 27),
         "unchanged", F_TINY, (94, 160, 110), anchor="lm")
    text(d, (AGENT[0] + 18, AGENT[1] + 48), "child process  ·  isolated env  ·  killable",
         F_TINY, MUTED, anchor="lm")
    text(d, (AGENT[0] + 18, AGENT[1] + 78), "TOOL OUTCOME LEDGER", F_ZONE, (102, 110, 122), anchor="lm")

    rrect(d, PROD, 10, fill=CARD, outline=LINE, width=1)
    text(d, (PROD[0] + 18, PROD[1] + 25), "your production agent", F_BOX, TXT, anchor="lm")
    text(d, (PROD[0] + 18, PROD[1] + 50), "fw.apply_current_policy(...)", F_CODE,
         (126, 176, 226), anchor="lm")

    # spawn connector, harness -> agent
    dashed_v(d, HARNESS[0] + 30, HARNESS[3], AGENT[1], LINE_SOFT, dash=5, gap=4, width=2)
    text(d, (HARNESS[0] + 42, (HARNESS[3] + AGENT[1]) / 2), "spawns", F_TINY, DIM, anchor="lm")

    # ── platform box: ONE opaque unit ──
    rrect(d, PLATFORM, 12, fill=(26, 24, 40), outline=(96, 76, 132), width=1)
    text(d, (PLATFORM[0] + 24, PLATFORM[1] + 34), "Percepteye", F_BOX, TXT, anchor="lm")
    text(d, (PLATFORM[0] + 24, PLATFORM[1] + 58), "control plane", F_BOX, (186, 156, 240), anchor="lm")
    dashed_h(d, PLATFORM[1] + 84, PLATFORM[0] + 24, PLATFORM[2] - 24, (66, 56, 88), width=1)
    return img


BASE = build_base()

PLAT_ROWS = [
    (PLATFORM[1] + 118, "generates the tasks"),
    (PLATFORM[1] + 168, "scores every rollout"),
    (PLATFORM[1] + 218, "trains the policy"),
]

LANE_LABEL = {
    "claim":  ("claim a rollout", "long-poll, leased"),
    "route":  ("completions", "per-rollout endpoint"),
    "report": ("result + outcomes", "final text, artifacts"),
    "policy": ("current policy", "an endpoint you can call"),
}


def draw_lane_rail(d, name):
    y = LANES[name]
    dashed_h(d, y, LANE_X0, LANE_X1, LINE_SOFT, dash=4, gap=5, width=1)


def draw_frame(i: int) -> Image.Image:
    t = i / FPS
    img = BASE.copy()
    d = ImageDraw.Draw(img)
    name, t0, t1, num, verb, cap = stage_at(t)
    p = ease(prog(t, t0, t1))

    # global fade-out at the seam so the loop restarts clean
    seam = 1.0 - ease(prog(t, DUR - 0.32, DUR))

    # ── header ──
    flywheel_icon(d, 56, 40, 15, t * 1.5, mix(BG, BLUE, 0.75))
    text(d, (82, 40), "the flywheel loop", F_TITLE, TXT, anchor="lm")
    text(d, (82 + tw("the flywheel loop", F_TITLE) + 14, 42),
         "your agent, your infrastructure, our rollouts", F_H, DIM, anchor="lm")

    dots_r = 4
    n_st = len(STAGES)
    dx0 = W - 40 - (n_st - 1) * 17
    for k, st in enumerate(STAGES):
        live = st[0] == name
        c = mix((46, 52, 60), BLUE, 1.0 if live else 0.0)
        cx = dx0 + k * 17
        if live:
            d.ellipse(S((cx - dots_r - 3, 41 - dots_r - 3, cx + dots_r + 3, 41 + dots_r + 3)),
                      outline=mix(BG, BLUE, 0.45), width=max(1, int(1 * SS)))
        d.ellipse(S((cx - dots_r, 41 - dots_r, cx + dots_r, 41 + dots_r)), fill=c)
    text(d, (dx0 - 18, 41), "the loop", F_TINY, DIM, anchor="rm")

    # ── lanes: rails first, then whatever is live ──
    for ln in LANES:
        draw_lane_rail(d, ln)

    def lane_labels(ln, active):
        y = LANES[ln]
        a, b = LANE_LABEL[ln]
        col = mix(DIM, TXT, 0.75 * active)
        text(d, ((LANE_X0 + LANE_X1) / 2, y - 17), a, F_LANE, col, anchor="mm")
        text(d, ((LANE_X0 + LANE_X1) / 2, y + 17), b, F_TINY,
             mix((60, 66, 75), MUTED, active), anchor="mm")

    def flow(ln, phase, color, n=1, trail=3, back=False, alpha=1.0):
        """phase 0..1 along the lane. back=True draws the RESPONSE, which
        travels the same rail in reverse -- dimmer and thinner, so it can never
        be misread as an arrow the platform opened."""
        y = LANES[ln]
        for k in range(n):
            ph = (phase - k * 0.17) % 1.0
            if ph <= 0.001:
                continue
            if back:
                x = lerp(LANE_X1, LANE_X0, ph)
                packet(d, x, y, mix(BG, color, 0.68), w=12, h=5,
                       alpha=alpha, trail=1)
            else:
                x = lerp(LANE_X0, LANE_X1, ph)
                packet(d, x, y, color, alpha=alpha, trail=trail)

    def solid_rail(ln, color, upto=1.0, width=2):
        y = LANES[ln]
        d.line([S(LANE_X0), S(y), S(lerp(LANE_X0, LANE_X1, upto)), S(y)],
               fill=color, width=max(1, int(width * SS)))
        if upto > 0.97:
            arrowhead(d, LANE_X1 - 7, y, color, size=7)

    # ── per-stage dynamics ──
    if name == "claim":
        lane_labels("claim", 1.0)
        if p < 0.55:
            q = p / 0.55
            solid_rail("claim", mix(BG, BLUE, 0.55), upto=q)
            flow("claim", q, BLUE)
        else:
            q = (p - 0.55) / 0.45
            solid_rail("claim", mix(BG, BLUE, 0.32), upto=1.0, width=1)
            flow("claim", q, CYAN, back=True)
            text(d, ((LANE_X0 + LANE_X1) / 2, LANES["claim"] + 40),
                 "← one task comes back on the connection you opened",
                 F_TINY, mix(BG, CYAN, 0.7), anchor="mm")
    else:
        lane_labels("claim", 0.15)

    if name == "spawn":
        rrect(d, (AGENT[0] - 3, AGENT[1] - 3, AGENT[2] + 3, AGENT[3] + 3), 12,
              outline=mix(BG, BLUE, 0.35 + 0.4 * math.sin(t * 9) ** 2), width=2)
        y = lerp(HARNESS[3], AGENT[1], p)
        packet(d, HARNESS[0] + 30, y, BLUE, w=7, h=7, trail=2)

    if name == "route":
        lane_labels("route", 1.0)
        solid_rail("route", mix(BG, BLUE, 0.5), upto=1.0)
        flow("route", (t - t0) / 1.15 % 1.0, BLUE, n=3)
        flow("route", ((t - t0) / 1.15 + 0.5) % 1.0, CYAN, back=True, n=2)
        rrect(d, (AGENT[0] - 3, AGENT[1] - 3, AGENT[2] + 3, AGENT[3] + 3), 12,
              outline=mix(BG, BLUE, 0.22 + 0.18 * math.sin(t * 5) ** 2), width=2)
    else:
        lane_labels("route", 0.15)

    # ── the ledger: fills during `record`, persists to the seam ──
    LEDGER = [("ok", GREEN, 0.00), ("ok", GREEN, 0.18),
              ("unknown", AMBER, 0.40), ("failed", RED, 0.62), ("ok", GREEN, 0.82)]
    lx = AGENT[0] + 18
    ly = AGENT[1] + 100
    for label, col, at in LEDGER:
        if name == "record":
            on = ease(prog(p, at, at + 0.16))
        elif t > STAGES[3][2]:
            on = seam
        else:
            on = 0.0
        lx += chip(d, lx, ly, label, col, on) + 7

    if name == "record":
        hi = None
        for label, col, at in LEDGER:
            if p >= at:
                hi = (label, col)
        if hi and hi[0] == "unknown":
            text(d, (AGENT[0] + 18, AGENT[3] + 22),
                 "unknown ≠ ok. The verifiers that need an outcome abstain.",
                 F_TINY, mix(BG, AMBER, 0.85), anchor="lm")

    if name == "report":
        lane_labels("report", 1.0)
        solid_rail("report", mix(BG, GREEN, 0.5), upto=min(1.0, p / 0.7))
        flow("report", min(1.0, p / 0.7), GREEN, n=2)
    else:
        lane_labels("report", 0.15)

    if name == "score":
        pass

    # platform rows: light up in order once the result lands
    for idx, (ry, rlabel) in enumerate(PLAT_ROWS):
        on = 0.0
        if name == "score":
            on = ease(prog(p, idx * 0.26, idx * 0.26 + 0.30))
        elif name == "policy":
            on = 0.55 * seam
        elif name in ("claim", "route") and idx == 0:
            on = 0.5 if name == "claim" else 0.0
        col = mix((84, 74, 106), (214, 196, 248), on)
        d.ellipse(S((PLATFORM[0] + 26, ry - 4, PLATFORM[0] + 34, ry + 4)),
                  fill=mix((70, 60, 92), PURPLE, on))
        text(d, (PLATFORM[0] + 48, ry), rlabel, F_SUB, col, anchor="lm")
        if on > 0.05:
            rrect(d, (PLATFORM[0] + 16, ry - 17, PLATFORM[2] - 20, ry + 17), 8,
                  outline=mix((26, 24, 40), PURPLE, on * 0.55), width=1)

    if name == "policy":
        lane_labels("policy", 1.0)
        if p < 0.30:
            q = p / 0.30
            solid_rail("policy", mix(BG, BLUE, 0.5), upto=q)
            flow("policy", q, BLUE)
            text(d, ((LANE_X0 + LANE_X1) / 2, LANES["policy"] + 40),
                 "your agent asks — outbound, like everything else",
                 F_TINY, mix(BG, BLUE, 0.7), anchor="mm")
        else:
            q = (p - 0.30) / 0.70
            solid_rail("policy", mix(BG, PURPLE, 0.30), upto=1.0, width=1)
            flow("policy", min(1.0, q / 0.8), PURPLE, back=True, n=2)
            if q > 0.26:
                a = ease(prog(q, 0.26, 0.55)) * seam
                rrect(d, (PROD[0] - 3, PROD[1] - 3, PROD[2] + 3, PROD[3] + 3), 12,
                      outline=mix(BG, PURPLE, 0.55 * a), width=2)
                text(d, ((LANE_X0 + LANE_X1) / 2, LANES["policy"] + 40),
                     "← the better policy, as an endpoint you can call",
                     F_TINY, mix(BG, PURPLE, 0.8 * a), anchor="mm")
    else:
        lane_labels("policy", 0.15)

    # ── the flywheel, made concrete: the policy the production agent is on ──
    ver_on = 0.0
    if name == "policy":
        ver_on = ease(prog(p, 0.46, 0.66)) * seam
    vlabel = "policy v2" if ver_on > 0.5 else "policy v1"
    vcol = PURPLE if ver_on > 0.5 else DIM
    vw = tw(vlabel, F_CHIP) + 20
    vx = PROD[2] - 18 - vw
    rrect(d, (vx, PROD[1] + 14, vx + vw, PROD[1] + 36), 11,
          fill=mix(CARD, mix(BG, vcol, 0.22), ver_on),
          outline=mix(LINE, vcol, ver_on), width=1)
    text(d, (vx + vw / 2, PROD[1] + 25), vlabel, F_CHIP,
         mix(MUTED, (214, 196, 248), ver_on), anchor="mm")

    # ── caption strip ──
    d.line([S(40), S(586), S(W - 40), S(586)], fill=(26, 31, 38),
           width=max(1, int(1 * SS)))
    cnx = 44
    rrect(d, (cnx, 600, cnx + 22, 622), 11, fill=mix(BG, BLUE, 0.20),
          outline=mix(BG, BLUE, 0.5), width=1)
    text(d, (cnx + 11, 612), num, F_CAPNUM, mix(BG, BLUE, 0.95), anchor="mm")
    text(d, (cnx + 34, 611), verb, F_CAP, TXT, anchor="lm")
    text(d, (cnx + 34 + tw(verb, F_CAP) + 14, 612), cap, F_SUB, MUTED, anchor="lm")

    return img.resize((W, H), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i in range(NFRAMES):
        draw_frame(i).save(os.path.join(OUT, "f%04d.png" % i))
        if i % 40 == 0:
            print("frame", i, "/", NFRAMES, flush=True)
    print("done", NFRAMES)
