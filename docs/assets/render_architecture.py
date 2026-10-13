#!/usr/bin/env python3
"""Render the README's closed-loop figure as an SVG.

    flywheel-architecture.svg  <-  this file

The loop, as a ring, with a dashed trust boundary straight down the middle:
the stations on the left are yours, the ones on the right are the part you do
not build. One round of it, clockwise:

1. your agent's real conversations are recorded where it runs -- by our SDK,
   by the OpenTelemetry traces it already exports, or read from your
   observability provider -- and sent once you allow it;
2. we learn from them;
3. each new version is tested against your agent's current best, on
   conversations it has never seen;
4. a version that does better comes to you in Decisions, and you approve it
   or not;
5. your agent runs the winner, and the next round starts from it as new
   conversations arrive.

Above the ring sits what happens once: connecting the agent and allowing it to
be learned from. Drawing that as a lane in the ring would say it recurs, which
it does not. Practice tasks are an optional extra and get a footnote, not a
lane.

Customer words only: no internal system, and the loop is described rather than
summarised as "we train a model". Both SDK READMEs and the docs site use this
same file.

SVG rather than a GIF because this one is static, and a static figure that is
text is one a reviewer can diff. Same palette as the two animations.

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
GREEN = "#3fb950"       # the winner, and what runs
RED = "#f85149"
AMBER = "#d29922"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

# ── geometry ─────────────────────────────────────────────────────────────────
# Cards are 380 wide, leaving a 340px channel in the middle. The boundary sits
# at its centre, so every crossing lane is the same length on both sides of it
# -- the figure should not imply that one direction travels further.
CARD_W, CARD_H = 380, 132
ONB_H = 104                     # the onboarding band's cards are shorter
L_X, R_X = 52, 768              # card left edges
ONB_Y = 200                     # the band: once, to connect
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
    f' aria-label="The improvement loop as a ring. Once: you connect your'
    f' agent, with our SDK, the OpenTelemetry traces it already exports, or'
    f' your observability provider, and allow us to learn from its'
    f' conversations. Then every round: its real conversations are sent from'
    f' your side; we learn from them; each new version is tested against your'
    f' agent&apos;s current best on conversations it has never seen; a version'
    f' that does better comes to you in Decisions; and your agent runs the'
    f' version you approved, and the next round starts from it. A dashed trust'
    f' boundary runs down the middle; every connection is opened from your'
    f' side.">')

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
text(258, 58, "your agent gets better from its own conversations",
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
text(W - L_X, 134, "PERCEPTEYE", size=11.5, fill=PURPLE,
     weight="700", spacing=1.2, anchor="end")
text(W - L_X, 152, "you don't build it — you can watch every round of it",
     size=11.5, fill=DIM, anchor="end")

# ── once: connect it, and allow it ───────────────────────────────────────────
text(L_X, ONB_Y - 16, "ONCE", size=10.5, fill=MUTED, weight="700",
     spacing=1.2)
text(L_X + 46, ONB_Y - 16, "connect it, then allow it to be learned from",
     size=10.5, fill=DIM)

rect(L_X, ONB_Y, CARD_W, ONB_H, fill=CARD, stroke=LINE, sw=1)
add(f'<rect x="{L_X}" y="{ONB_Y + 14}" width="3" height="{ONB_H - 28}" rx="1.5"'
    f' fill="{BLUE}"/>')
text(L_X + 24, ONB_Y + 32, "connect your agent", size=17, weight="700")
cx = L_X + 24
for name in ("our SDK", "OpenTelemetry", "your observability provider"):
    cx += chip(cx, ONB_Y + 46, name, BLUE, size=10.5) + 6
text(L_X + 24, ONB_Y + 92, "nothing inside your agent changes",
     size=11.5, fill=MUTED)

rect(R_X, ONB_Y, CARD_W, ONB_H, fill=CARD, stroke=LINE, sw=1)
add(f'<rect x="{R_X}" y="{ONB_Y + 14}" width="3" height="{ONB_H - 28}" rx="1.5"'
    f' fill="{PURPLE}"/>')
text(R_X + 24, ONB_Y + 32, "your permission", size=17, weight="700")
text(R_X + 24, ONB_Y + 60, "none of its conversations are kept until you allow it",
     size=11.5, fill=MUTED)
text(R_X + 24, ONB_Y + 84, "switch it off at any time, without a restart",
     size=11.5, fill=DIM)

lane(ONB_Y + 52, "out", BLUE, "0", "connects, outbound",
     "with a key that belongs to this one agent")

# ── then the ring, every round ───────────────────────────────────────────────
text(L_X, TOP_Y - 22, "THEN, EVERY ROUND", size=10.5, fill=MUTED, weight="700",
     spacing=1.2)
text(L_X + 150, TOP_Y - 22, "each round starts from the best so far",
     size=10.5, fill=DIM)


def station(x: float, y: float, accent: str) -> None:
    rect(x, y, CARD_W, CARD_H, fill=CARD, stroke=LINE, sw=1)
    add(f'<rect x="{x}" y="{y + 16}" width="3" height="{CARD_H - 32}" rx="1.5"'
        f' fill="{accent}"/>')


# top-left -- the agent, serving its users
station(L_X, TOP_Y, BLUE)
text(L_X + 24, TOP_Y + 36, "your agent", size=18, weight="700")
chip(L_X + 132, TOP_Y + 21, "unchanged", GREEN)
cx = L_X + 24
for name in ("what users asked", "what it did", "what it answered"):
    cx += chip(cx, TOP_Y + 54, name, BLUE, size=10.5) + 6
text(L_X + 24, TOP_Y + 104, "its real conversations, recorded where it runs",
     size=12.5, fill=MUTED)

# top-right -- learning
station(R_X, TOP_Y, PURPLE)
text(R_X + 24, TOP_Y + 36, "2 · learns from them", size=18, weight="700")
cx = R_X + 24
for name in ("what the model saw", "how it went on"):
    cx += chip(cx, TOP_Y + 54, name, PURPLE, size=10.5) + 6
text(R_X + 24, TOP_Y + 104, "personal data is found, and never learned from",
     size=12.5, fill=MUTED)

# bottom-right -- testing
station(R_X, BOT_Y, PURPLE)
text(R_X + 24, BOT_Y + 36, "3 · tests the new version", size=18, weight="700")
cx = R_X + 24
cx += chip(cx, BOT_Y + 54, "against your current best", PURPLE, size=10.5) + 6
text(R_X + 24, BOT_Y + 104, "on conversations it has never seen",
     size=12.5, fill=MUTED)

# bottom-left -- the person who decides
station(L_X, BOT_Y, GREEN)
text(L_X + 24, BOT_Y + 36, "you approve the winner", size=18, weight="700")
cx = L_X + 24
for name in ("Decisions", "approve", "reject"):
    cx += chip(cx, BOT_Y + 54, name, GREEN, size=10.5) + 6
text(L_X + 24, BOT_Y + 104, "until you do, your agent keeps what it has",
     size=12.5, fill=MUTED)

# ── the two lanes that cross the boundary ────────────────────────────────────
lane(TOP_Y + 66, "out", BLUE, "1", "its conversations",
     "sent only once you allow it", opt_in=True)
lane(BOT_Y + 66, "in", PURPLE, "4", "a version that did better",
     "waits for your approval")

# ── the two edges that close the ring ────────────────────────────────────────
# Right: what was learned becomes a new version, and is tested.
arrow(R_MID, TOP_Y + CARD_H + 10, R_MID, BOT_Y - 12, PURPLE, sw=1.4)
lw = tw("a new version", 11.5) + 20
rect(R_MID - lw / 2, TOP_Y + CARD_H + 38, lw, 22, r=11, fill=BG)
text(R_MID, TOP_Y + CARD_H + 53, "a new version", size=11.5,
     fill=DIM, anchor="middle")

# Left: the approved version runs, and the next round starts from it. This is
# the edge that makes it a loop rather than a pipeline, so it is the green one.
arrow(L_MID, BOT_Y - 12, L_MID, TOP_Y + CARD_H + 10, GREEN, sw=1.4)
lw = tw("5 · runs the winner, and repeats", 11.5) + 20
rect(L_MID - lw / 2, TOP_Y + CARD_H + 38, lw, 22, r=11, fill=BG)
text(L_MID, TOP_Y + CARD_H + 53, "5 · runs the winner, and repeats",
     size=11.5, fill=GREEN, anchor="middle", opacity=0.9)

text(W - L_X, BOT_Y + CARD_H + 30,
     "optional: practice tasks we write from your agent's own tools",
     size=11, fill=DIM, anchor="end")

# ── footer ───────────────────────────────────────────────────────────────────
add(f'<line x1="{L_X}" y1="{H - 58}" x2="{W - L_X}" y2="{H - 58}"'
    f' stroke="{LINE}" stroke-width="1"/>')
text(L_X, H - 32, "every connection is opened from your side", size=12.5,
     fill=TXT, weight="600")
text(L_X + 292, H - 32,
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
