#!/usr/bin/env bash
# Rebuild the README's animations from their renderers.
#
#   flywheel-overview.{gif,png}  <- render_overview.py         (the hero)
#   flywheel-loop.{gif,png}      <- render_flywheel_loop.py    (the architecture)
#   flywheel-architecture.svg    <- render_architecture.py     (the closed loop)
#   flywheel-spinner.gif         <- render_spinner.py          (the title mark)
#
# The overview and loop figures come in TWO FLAVOURS. They show what a customer
# would type, and that differs per package -- so `./render.sh node` writes
# *-node.gif variants carrying `openclaw plugins install` instead of
# `pip install`. Shipping the Python frames in the Node README advertised the
# wrong SDK to everyone who opened it.
#
# Both are checked in, because a README cannot fetch a build artifact -- and
# both are generated, so a checked-in asset with no build script drifts from
# its source the first time someone edits one without the other. Run this after
# touching either renderer.
#
#   ./render.sh            # all three
#   ./render.sh overview   # just one
#
# Needs: python with Pillow, and ffmpeg -- EXCEPT for the architecture figure,
# which is an SVG and needs neither. It is static, so it does not need a frame
# pipeline, and being text it is the one figure a reviewer can read in a diff.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

# palettegen/paletteuse rather than ffmpeg's default quantiser: both figures are
# flat colour on a flat ground, so an adaptive palette with NO dithering both
# looks cleaner and compresses far better -- dither noise defeats the GIF
# encoder's inter-frame diffing, which is what keeps these files small.
#
# The colour budgets are not interchangeable, and both were measured rather than
# guessed: the overview is 1.70 MB at 128 and 1.84 MB at 192 with no visible
# difference, so it takes 128; the loop has more simultaneous fades and keeps 224.
render() {   # name renderer colours poster   (honours $FLAVOUR)
  local name=$1 renderer=$2 colours=$3 poster=$4
  local frames; frames=$(mktemp -d)
  trap 'rm -rf "$frames"' RETURN

  FRAMES="$frames" "$PY" "$renderer"

  local fps; fps=$("$PY" -c "import re,sys;print(re.search(r'^FPS = (\d+)',open('$renderer').read(),re.M).group(1))")
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames/f%04d.png" \
    -vf "palettegen=max_colors=$colours:stats_mode=diff" "$frames/palette.png"
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames/f%04d.png" -i "$frames/palette.png" \
    -lavfi "paletteuse=dither=none:diff_mode=rectangle" -loop 0 "$name.gif"

  # The poster frame is a late one, where the whole picture is on screen -- it
  # is what renders anywhere a GIF does not animate.
  cp "$frames/$poster" "$name.png"
  ls -l "$name.gif" "$name.png"
}

architecture() { "$PY" render_architecture.py; }

# The title mark. Its own function rather than a `render` call, because it is
# the one figure with TRANSPARENCY: it sits inline beside a heading on whatever
# ground GitHub is painting, so the palette has to reserve a transparent index
# and paletteuse has to be told where to cut it. Passing those flags to the two
# opaque figures would cost them a palette slot for nothing.
spinner() {
  local frames; frames=$(mktemp -d)
  trap 'rm -rf "$frames"' RETURN
  FRAMES="$frames" "$PY" render_spinner.py
  local fps; fps=$("$PY" -c "import re;print(re.search(r'^FPS = (\d+)',open('render_spinner.py').read(),re.M).group(1))")
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames/f%04d.png" \
    -vf "palettegen=max_colors=64:reserve_transparent=1:stats_mode=diff" "$frames/palette.png"
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames/f%04d.png" -i "$frames/palette.png" \
    -lavfi "paletteuse=dither=none:alpha_threshold=128:diff_mode=rectangle" -loop 0 flywheel-spinner.gif
  ls -l flywheel-spinner.gif
}

case "${1:-all}" in
  overview)     render flywheel-overview render_overview.py 128 f0455.png ;;
  spinner)      spinner ;;
  loop)         render flywheel-loop render_flywheel_loop.py 224 f0466.png ;;
  architecture) architecture ;;
  node)
    # The Node README's copies. Same diagrams, the customer-facing strings
    # swapped; copied into the sibling repo, which has no renderer of its own.
    FLAVOUR=node render flywheel-overview-node render_overview.py 128 f0455.png
    FLAVOUR=node render flywheel-loop-node render_flywheel_loop.py 224 f0466.png
    ;;
  all)
    render flywheel-overview render_overview.py 128 f0455.png
    render flywheel-loop render_flywheel_loop.py 224 f0466.png
    spinner
    architecture
    ;;
  *) echo "usage: $0 [all|overview|loop|spinner|architecture|node]" >&2; exit 2 ;;
esac
