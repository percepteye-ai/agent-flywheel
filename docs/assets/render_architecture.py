#!/usr/bin/env python3
"""Render the README's closed-loop architecture figure as an SVG.

    flywheel-architecture.svg  <-  this file

The two GIFs next door tell the story in time: the overview shows one agent
going round, the loop shows the components in their trust zones. Neither shows
the thing the README's "How the loop closes" section is actually claiming --
that the four lanes form a *ring*, and that half of that ring is yours.

So this figure is a ring. Four stations, clockwise, with a dashed trust
boundary straight down the middle: the two stations on the left are processes
you run, the two on the right are the part you do not build. The four lanes
that cross the boundary are the four lanes in the prose, in the same order,
with the same numbers.

Above the ring sits the beat that is NOT part of it: onboarding. Registration
happens once, at startup, before the first rollout -- and it is the step that
makes everything below it possible, because the tasks are derived from the
agent description rather than from a list the customer enumerates. Drawing it
as a fifth lane in the ring would say it recurs, which it does not; drawing it
as a band above says "once, then loop forever", which is what happens.

SVG rather than a GIF because this one is static, and a static figure that is
text is one a reviewer can diff. Same palette as ``render_flywheel_loop.py`` --
GitHub's dark canvas, blue for your side, purple for the platform -- because
two figures on one page that disagree about what blue means are worse than one.

No Pillow, no ffmpeg, no network. Run it and it prints where it wrote::

    ./render_architecture.py
"""
from __future__ import annotations

import os
from xml.sax.saxutils import escape

# ── canvas ───────────────────────────────────────────────────────────────────
W, H = 1200, 900

# ── palette ── identical to render_flywheel_loop.py, deliberately ────────────
BG = "#0d1117"          # GitHub dark canvas
CARD = "#161b22"
LINE = "#30363d"
TXT = "#e6edf3"
MUTED = "#8b949e"
DIM = "#6e7681"

BLUE = "#58a6ff"        # your side
PURPLE = "#a371f7"      # platform
GREEN = "#3fb950"       # ok
RED = "#f85149"         # failed
AMBER = "#d29922"       # unknown -- deliberately NOT green

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

# ── geometry ─────────────────────────────────────────────────────────────────
# Cards are 380 wide, leaving a 340px channel in the middle. The boundary sits
# at its centre, so every crossing lane is the same length on both sides of it
# -- the figure should not imply that one direction travels further.
CARD_W, CARD_H = 380, 132
ONB_H = 104                     # the onboarding band's cards are shorter
L_X, R_X = 52, 768              # card left edges
ONB_Y = 200                     # the band: once, at startup
TOP_Y, BOT_Y = 380, 672         # the ring: then, every lap
MID_X = 600                     # the trust boundary
L_MID = L_X + CARD_W // 2       # 242 -- centre of the left column
R_MID = R_X + CARD_W // 2       # 958 -- centre of the right column

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def tw(text: str, size: float, mono: bool = False) -> float:
    """Approximate rendered width. Only chips need this, and they are padded."""
    return len(text) * size * (0.60 if mono else 0.53)


def text(
    x: float, y: float, s: str, *, size: float = 13, fill: str = TXT,
    weight: str = "400", anchor: str = "start", mono: bool = False,
    spacing: float | None = None, opacity: float | None = None,
) -> None:
    attrs = [
        f'x="{x}"', f'y="{y}"', f'font-family="{MONO if mono else SANS}"',
        f'font-size="{size}"', f'font-weight="{weight}"', f'fill="{fill}"',
    ]
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if spacing is not None:
        attrs.append(f'letter-spacing="{spacing}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    add(f'<text {" ".join(attrs)}>{escape(s)}</text>')


def rect(
    x: float, y: float, w: float, h: float, *, r: float = 14,
    fill: str = "none", stroke: str | None = None, sw: float = 1,
    dash: str | None = None, opacity: float | None = None,
) -> None:
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
             f'rx="{r}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{sw}"')
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    add(f'<rect {" ".join(attrs)}/>')


def chip(x: float, y: float, label: str, colour: str, *, size: float = 11.5,
         mono: bool = False, dot: bool = False) -> float:
    """A small pill. Returns its width, so callers can lay a row out."""
    pad = 9
    lead = 13 if dot else 0
    w = tw(label, size, mono) + pad * 2 + lead
    h = 22
    rect(x, y, w, h, r=11, fill=colour, stroke=colour, sw=1, opacity=0.14)
    rect(x, y, w, h, r=11, stroke=colour, sw=1, opacity=0.55)
    if dot:
        add(f'<circle cx="{x + pad + 3}" cy="{y + h / 2}" r="3" fill="{colour}"/>')
    text(x + pad + lead, y + h / 2 + 4, label, size=size, fill=colour,
         weight="600", mono=mono)
    return w


def arrow(x1: float, y1: float, x2: float, y2: float, colour: str, *,
          sw: float = 1.6, dash: str | None = None, opacity: float = 1.0) -> None:
    marker = f"a{colour.lstrip('#')}"
    attrs = [f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
             f'stroke="{colour}"', f'stroke-width="{sw}"',
             f'marker-end="url(#{marker})"', f'opacity="{opacity}"']
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    add(f'<line {" ".join(attrs)}/>')


def lane(y: float, direction: str, colour: str, num: str, label: str,
         sub: str, *, opt_in: bool = False) -> None:
    """One numbered lane crossing the trust boundary.

    ``direction`` is "out" (your side -> platform) or "in". Both are drawn the
    same weight: the arrowhead is the only thing that differs, because which
    side *opened* the connection is a fact about the protocol, not about the
    picture, and the README says it in words directly underneath.

    ``opt_in`` dashes the line, for a lane that carries nothing until someone
    turns it on. A solid line would draw a flow that is off by default as if
    it were always running -- the same false picture as drawing no lane at all
    for one that runs once enabled.
    """
    x_from, x_to = (L_X + CARD_W + 18, R_X - 18)
    if direction == "in":
        x_from, x_to = x_to, x_from
    arrow(x_from, y, x_to, y, colour, dash="6 5" if opt_in else None)

    # The number rides on the line, on a plug of background so the rule does
    # not strike through it.
    add(f'<circle cx="{MID_X}" cy="{y}" r="13" fill="{BG}"/>')
    add(f'<circle cx="{MID_X}" cy="{y}" r="12" fill="{colour}" opacity="0.16"/>')
    add(f'<circle cx="{MID_X}" cy="{y}" r="12" fill="none" stroke="{colour}"'
        f' stroke-width="1.2" opacity="0.7"/>')
    text(MID_X, y + 4.5, num, size=12, fill=colour, weight="700", anchor="middle")

    text(MID_X, y - 26, label, size=13.5, fill=TXT, weight="600", anchor="middle")
    text(MID_X, y + 26, sub, size=11.5, fill=DIM, anchor="middle")


# ── document ─────────────────────────────────────────────────────────────────
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
    f' width="{W}" height="{H}" role="img"'
    f' aria-label="The agent flywheel as a closed ring: your agent runs a'
    f' rollout, PerceptEye scores the observed outcomes and trains a policy,'
    f' your production agent picks it up, and, once an operator enables'
    f' capture, its live evidence feeds the next lap. A dashed trust boundary'
    f' runs down the middle; the two stations on the left are processes you'
    f' run, the two on the right are the control plane. Four numbered lanes'
    f' cross the boundary; the fourth is dashed because it is opt-in.">')

add('<defs>')
for c in (BLUE, PURPLE, GREEN):
    add(f'<marker id="a{c.lstrip("#")}" viewBox="0 0 10 10" refX="9" refY="5"'
        f' markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
add('</defs>')

rect(0, 0, W, H, r=0, fill=BG)

# ── header ───────────────────────────────────────────────────────────────────
add(f'<circle cx="52" cy="52" r="17" fill="none" stroke="{BLUE}" stroke-width="1.6"/>')
add(f'<circle cx="52" cy="52" r="5.5" fill="none" stroke="{BLUE}" stroke-width="1.6"/>')
for a in (0, 90, 180, 270):
    add(f'<line x1="52" y1="52" x2="52" y2="35" stroke="{BLUE}" stroke-width="1.4"'
        f' transform="rotate({a} 52 52)" opacity="0.75"/>')
text(84, 58, "the loop closes", size=21, weight="700")
text(258, 58, "one import  ·  one call  ·  one runtime dependency",
     size=13.5, fill=MUTED)
text(W - 52, 58, "0 lines changed inside your agent", size=13, fill=GREEN,
     weight="600", anchor="end")

# ── trust boundary ───────────────────────────────────────────────────────────
add(f'<line x1="{MID_X}" y1="112" x2="{MID_X}" y2="{H - 74}" stroke="{LINE}"'
    f' stroke-width="1.4" stroke-dasharray="5 9"/>')
bw = tw("TRUST BOUNDARY", 10.5) + 26
rect(MID_X - bw / 2, 92, bw, 24, r=12, fill=BG, stroke=LINE, sw=1)
text(MID_X, 108, "TRUST BOUNDARY", size=10.5, fill=MUTED, weight="700",
     anchor="middle", spacing=1.1)

# ── zone labels ──────────────────────────────────────────────────────────────
text(L_X, 134, "YOUR INFRASTRUCTURE", size=11.5, fill=BLUE, weight="700",
     spacing=1.2)
text(L_X, 152, "your process · your credentials · your firewall",
     size=11.5, fill=DIM)
text(W - L_X, 134, "PERCEPTEYE CONTROL PLANE", size=11.5, fill=PURPLE,
     weight="700", spacing=1.2, anchor="end")
text(W - L_X, 152, "you don't build it — you can watch every lap of it",
     size=11.5, fill=DIM, anchor="end")

# ── beat zero: onboarding, which happens once and then never again ──────────
text(L_X, ONB_Y - 16, "ONCE, AT STARTUP", size=10.5, fill=MUTED, weight="700",
     spacing=1.2)
text(L_X + 150, ONB_Y - 16, "before the first rollout, never during one",
     size=10.5, fill=DIM)

rect(L_X, ONB_Y, CARD_W, ONB_H, fill=CARD, stroke=LINE, sw=1)
add(f'<rect x="{L_X}" y="{ONB_Y + 14}" width="3" height="{ONB_H - 28}" rx="1.5"'
    f' fill="{BLUE}"/>')
text(L_X + 24, ONB_Y + 32, "describes your agent", size=17, weight="700")
cx = L_X + 24
for name in ("system prompt", "tool schemas", "sub-agent tree"):
    cx += chip(cx, ONB_Y + 46, name, BLUE, size=10.5) + 6
text(L_X + 24, ONB_Y + 92, "read from inside the process, after templating",
     size=11.5, fill=MUTED)

rect(R_X, ONB_Y, CARD_W, ONB_H, fill=CARD, stroke=LINE, sw=1)
add(f'<rect x="{R_X}" y="{ONB_Y + 14}" width="3" height="{ONB_H - 28}" rx="1.5"'
    f' fill="{PURPLE}"/>')
text(R_X + 24, ONB_Y + 32, "the tasks worth training on", size=17, weight="700")
text(R_X + 24, ONB_Y + 60, "derived from the tool surface you actually have",
     size=11.5, fill=MUTED)
text(R_X + 24, ONB_Y + 84, "not a list you enumerate, not a rubric you write",
     size=11.5, fill=DIM)

lane(ONB_Y + 52, "out", BLUE, "0", "registers, on its own",
     "fw.serve() and fw.attach() both do it")

text(L_X + 24, ONB_Y + ONB_H + 26,
     "a CLI agent has no object to read, so it is warmed up once — well under a"
     " second — and described from that", size=11, fill=DIM)
text(W - L_X, ONB_Y + ONB_H + 26, "opt out with introspect=False", size=11,
     fill=DIM, anchor="end")

# ── then the ring, which runs for as long as you serve ──────────────────────
text(L_X, TOP_Y - 22, "THEN, EVERY LAP", size=10.5, fill=MUTED, weight="700",
     spacing=1.2)
text(L_X + 138, TOP_Y - 22, "four lanes, and every one of them opened from your"
     " side", size=10.5, fill=DIM)

# ── the four stations ────────────────────────────────────────────────────────
def station(x: float, y: float, accent: str) -> None:
    rect(x, y, CARD_W, CARD_H, fill=CARD, stroke=LINE, sw=1)
    add(f'<rect x="{x}" y="{y + 16}" width="3" height="{CARD_H - 32}" rx="1.5"'
        f' fill="{accent}"/>')


# top-left -- your agent
station(L_X, TOP_Y, BLUE)
text(L_X + 24, TOP_Y + 36, "your agent", size=18, weight="700")
chip(L_X + 132, TOP_Y + 21, "unchanged", GREEN)
cx = L_X + 24
for name in ("Python", "Node", "any CLI"):
    cx += chip(cx, TOP_Y + 54, name, BLUE, size=11) + 7
text(L_X + 24, TOP_Y + 104, 'fw.serve("my_package.agent:build")', size=12.5,
     fill=MUTED, mono=True)

# top-right -- scoring
station(R_X, TOP_Y, PURPLE)
text(R_X + 24, TOP_Y + 36, "scores what was observed", size=18, weight="700")
cx = R_X + 24
for name, col in (("ok", GREEN), ("failed", RED), ("unknown", AMBER)):
    cx += chip(cx, TOP_Y + 54, name, col, dot=True) + 8
text(R_X + 24, TOP_Y + 104, "unknown is never rounded up to ok", size=12.5,
     fill=MUTED)

# bottom-right -- training
station(R_X, BOT_Y, PURPLE)
text(R_X + 24, BOT_Y + 36, "trains + promotes", size=18, weight="700")
cx = R_X + 24
cx += chip(cx, BOT_Y + 54, "the model", PURPLE, size=11) + 7
chip(cx, BOT_Y + 54, "the system prompt", PURPLE, size=11)
text(R_X + 24, BOT_Y + 104, "co-optimized, and gated by the same approval",
     size=12.5, fill=MUTED)

# bottom-left -- production
station(L_X, BOT_Y, BLUE)
text(L_X + 24, BOT_Y + 36, "your production agent", size=18, weight="700")
text(L_X + 24, BOT_Y + 62, "fw.apply_current_policy()", size=12.5, fill=BLUE,
     mono=True)
text(L_X + 24, BOT_Y + 104, "promotion and rollback reach it with no redeploy",
     size=12.5, fill=MUTED)

# ── the four crossing lanes ──────────────────────────────────────────────────
lane(TOP_Y + 28, "in", PURPLE, "1", "claims a task",
     "generated from your real tool surface")
lane(TOP_Y + 110, "out", BLUE, "2", "rollout + per-call outcomes",
     "ok · failed · unknown, per tool call")
lane(BOT_Y + 28, "in", PURPLE, "3", "the promoted policy",
     "an endpoint your agent already calls")
# Opt-in per agent and off by default: `Attachment.flush()` uploads served
# turns only once an operator enables capture for that agent. Before that
# existed this lane was simply false, and it was deleted from the SVG by hand
# -- which the renderer never learned, so a re-render quietly redrew it as an
# unconditional flow. It is true again now, conditionally, and drawn that way.
lane(BOT_Y + 110, "out", BLUE, "4", "live evidence, via fw.attach()",
     "sent only once an operator enables capture", opt_in=True)

# ── the two edges that close the ring ────────────────────────────────────────
# Right: scoring feeds training, inside the control plane.
arrow(R_MID, TOP_Y + CARD_H + 10, R_MID, BOT_Y - 12, PURPLE, sw=1.4)
lw = tw("scored, then trained on", 11.5) + 20
rect(R_MID - lw / 2, TOP_Y + CARD_H + 38, lw, 22, r=11, fill=BG)
text(R_MID, TOP_Y + CARD_H + 53, "scored, then trained on", size=11.5,
     fill=DIM, anchor="middle")

# Left: the better agent runs the next lap. This is the edge that makes it a
# flywheel rather than a pipeline, so it is the one drawn in green.
arrow(L_MID, BOT_Y - 12, L_MID, TOP_Y + CARD_H + 10, GREEN, sw=1.4)
lw = tw("the better agent runs the next lap", 11.5) + 20
rect(L_MID - lw / 2, TOP_Y + CARD_H + 38, lw, 22, r=11, fill=BG)
text(L_MID, TOP_Y + CARD_H + 53, "the better agent runs the next lap",
     size=11.5, fill=GREEN, anchor="middle", opacity=0.9)

# ── footer ───────────────────────────────────────────────────────────────────
add(f'<line x1="{L_X}" y1="{H - 58}" x2="{W - L_X}" y2="{H - 58}"'
    f' stroke="{LINE}" stroke-width="1"/>')
text(L_X, H - 32, "every lane is opened from your side", size=12.5, fill=TXT,
     weight="600")
text(L_X + 232, H - 32,
     "no inbound port  ·  no public DNS  ·  no container  ·  no GPU",
     size=12.5, fill=DIM)
text(W - L_X, H - 32, "outbound only", size=12.5, fill=BLUE, weight="600",
     anchor="end")

add("</svg>")

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "flywheel-architecture.svg")
with open(dest, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
print(dest)
