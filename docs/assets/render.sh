#!/usr/bin/env bash
# Rebuild the README's animations from their renderers.
#
#   flywheel-overview.{gif,png}  <- render_overview.py         (the hero)
#   flywheel-loop.{gif,png}      <- render_flywheel_loop.py    (the architecture)
#   flywheel-architecture.svg    <- render_architecture.py     (the closed loop)
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
render() {   # name renderer colours poster
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

case "${1:-all}" in
  overview)     render flywheel-overview render_overview.py 128 f0455.png ;;
  loop)         render flywheel-loop render_flywheel_loop.py 224 f0466.png ;;
  architecture) architecture ;;
  all)
    render flywheel-overview render_overview.py 128 f0455.png
    render flywheel-loop render_flywheel_loop.py 224 f0466.png
    architecture
    ;;
  *) echo "usage: $0 [all|overview|loop|architecture]" >&2; exit 2 ;;
esac
