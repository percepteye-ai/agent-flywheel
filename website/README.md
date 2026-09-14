# The documentation site

Source for **https://docs.percepteye.ai/flywheel/**.

```bash
pip install -e ".[docs]"
cd website
mkdocs serve          # http://127.0.0.1:8000, live-reloading
mkdocs build --strict # -> website/site/, warnings are errors
```

`--strict` is the one that matters: it turns a broken internal link into a failed
build rather than a 404 someone finds later.

## Layout

| | |
|:--|:--|
| `mkdocs.yml` | theme, nav, markdown extensions |
| `docs/` | the pages, one markdown file each |
| `docs/stylesheets/flywheel.css` | the theme — palette, landing cards, typography |
| `docs/assets/` | figures, copied from `../docs/assets/` |
| `site/` | build output, gitignored |

## Why the site has its own content root

The repo's top-level `docs/` holds internal planning notes (`plan-*.md`) and the
figure renderers. Pointing mkdocs at it would publish the planning notes. This
directory contains only what is meant to be public.

## The figures

`docs/assets/flywheel-architecture.svg` is a copy, not an original. It is
generated — edit `../docs/assets/render_architecture.py`, run
`../docs/assets/render.sh architecture`, then re-copy:

```bash
cp ../docs/assets/flywheel-architecture.svg docs/assets/
```

`tests/test_architecture_svg_matches_its_renderer.py` re-renders the figure and
fails if either copy differs from it byte for byte, so a hand-edited SVG — or a
forgotten re-copy — turns the suite red instead of the site stale.

Only what a page actually references lives here. The two animated GIFs in
`../docs/assets/` are ~4 MB together and no page uses them, so they are
deliberately not duplicated into the site. Copy one in if you reference it.

They are drawn on GitHub's dark canvas and are not relit for the light theme —
the stylesheet frames them as deliberate dark panels instead.

## Deploying

`mkdocs build` emits a static directory. Any static host serves it. Two notes
for whoever wires up DNS:

- `docs.percepteye.ai` does not resolve yet. It is NXDOMAIN on both public
  resolvers, so the record still has to be created.
- `site_url` in `mkdocs.yml` is `https://docs.percepteye.ai/flywheel/`. If the
  site is served at the domain root instead, change it — it is what canonical
  URLs and the sitemap are built from.
