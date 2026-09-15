# Releasing the Agent Flywheel SDKs

Two packages, two registries, one release train. The Python repository leads:
it owns the wire schema and renders both READMEs' figures, and the Node
repository takes copies of both.

| | Python | Node |
|:--|:--|:--|
| repository | `percepteye-ai/agent-flywheel` | `percepteye-ai/agent-flywheel-node` |
| registry | PyPI [`agent-flywheel`](https://pypi.org/project/agent-flywheel/) — published | npm [`@percepteye/agent-flywheel`](https://www.npmjs.com/package/@percepteye/agent-flywheel), owned by the `percepteye` npm organization |
| version lives in | `src/agent_flywheel/_version.py`; `pyproject.toml` and the goose manifest repeat it, and `tests/test_one_version.py` holds all three together | `package.json`; the User-Agent derives from it, and `test/version.test.js` holds `openclaw.plugin.json` (version *and* description) equal to it |
| publishes through | `.github/workflows/workflow.yml` on a `v*` tag, via PyPI trusted publishing | `.github/workflows/publish.yml` on a `v*` tag, via npm trusted publishing, after a manual first publish ([§3](#3-node-release-npm)) |

Neither repository is public yet. That is why the PyPI page shows the figures'
alt text; [§4](#4-making-the-repositories-public--not-done-yet) is the checklist
for fixing it.

Run every command below from the repository root, not from `docs/`.

## 1. Before any release

**Python**, from a checkout with every extra installed and the Node repository
checked out beside it (the parity tests find it there, or through
`PERCEPTEYE_NODE_SDK`):

```bash
.venv/bin/pytest -q                        # N passed, 0 skipped
.venv/bin/ruff check src tests
./scripts/cross_language_abi_check.sh
.venv/bin/python -m pip install build twine      # once; the release job uses the same two
rm -rf dist && .venv/bin/python -m build && .venv/bin/python -m twine check --strict dist/*
.venv/bin/mkdocs build --strict -f website/mkdocs.yml -d /tmp/site
```

`dist/` is cleared first on purpose: a local `dist/` can hold artifacts from an
older package name, and nothing should ever upload those.

**Node:**

```bash
npm test                                   # N passed, 0 skipped
npm pack --dry-run                         # percepteye-agent-flywheel-<version>.tgz, no test/ or docs/
```

**If a figure renderer changed**, re-render on macOS (the GIF/PNG renderers use
the system SF fonts; on this setup the output reproduces the committed bytes
exactly, which is what makes a re-render reviewable), then copy the four Node
flavours across under their plain names:

```bash
./docs/assets/render.sh                    # every figure, both flavours
for f in overview loop; do for e in gif png; do
  cp docs/assets/flywheel-$f-node.$e ../agent-flywheel-node/docs/assets/flywheel-$f.$e
done; done
```

`tests/test_node_figures_match.py` fails until the copies land, and
`tests/test_architecture_svg_matches_its_renderer.py` fails if the architecture
SVG was edited by hand instead of through its renderer.

**If `schema/flywheel-1.json` changed**, copy it into the Node repository too.

**Merge order.** When figures or the schema changed, merge the Node PR first:
this repository's `test-full` job checks out the Node repository's `main`.

## 2. Python release (PyPI)

1. **Bump the version** in `src/agent_flywheel/_version.py`, `pyproject.toml`
   and `src/agent_flywheel/hosts/goose/plugin.json`, then run `uv lock`.
   `tests/test_one_version.py` fails if one is missed.
2. **If tests were added or removed**, update the published suite size: three
   places in the README, plus `website/docs/python.md` and
   `website/docs/index.md`. `tests/test_documented_counts_are_true.py` names
   every stale copy.
3. **Merge to `main` through a PR.**
4. **Tag `main` and push the tag.** It must equal the `pyproject.toml` version,
   because the build job asserts it:

   ```bash
   git switch main && git pull --ff-only
   git tag -a vX.Y.Z -m "agent-flywheel X.Y.Z" && git push origin vX.Y.Z
   ```

5. **Watch it.** The `release` workflow builds, runs `twine check --strict`, and
   publishes:

   ```bash
   gh run watch "$(gh run list --workflow workflow.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
   ```

6. **Verify on PyPI itself**, not just the green check: the version's JSON
   (`https://pypi.org/pypi/agent-flywheel/X.Y.Z/json`), then a fresh
   environment, pinned first and then unpinned:

   ```bash
   uv venv /tmp/af && uv pip install --python /tmp/af/bin/python --no-cache "agent-flywheel==X.Y.Z"
   /tmp/af/bin/python -c "import agent_flywheel as fw; print(fw.__version__)"
   /tmp/af/bin/agent-flywheel --help
   ```

### Python: what bit us, and what to know

- **The workflow's file name is part of the PyPI publisher.** The publisher is:
  owner `percepteye-ai`, repository `agent-flywheel`, workflow `workflow.yml`,
  environment `pypi`. PyPI matches the *file* name, not `name:`. The first
  0.1.0 run came from `release.yml`, and the token exchange failed with
  `invalid-publisher` before anything uploaded. Rename the file only together
  with the publisher.
- **A tag publishes immediately.** The `pypi` environment has no required
  reviewer, and cannot have one yet: on the GitHub Team plan, required
  reviewers only work on public repositories. Add one on the day the
  repository goes public ([§4d](#4d-also-on-the-day)).
- **PyPI is append-only.** A version or filename can never be reused, even
  after deletion, and a release's description can never be edited. A README
  fix reaches PyPI only as a new version: 0.1.1 was published for exactly that.
- **A failed publish spends nothing.** If the build or the token exchange
  fails, nothing uploads. Fix it, then run `gh run rerun <run-id>`. A re-run
  replays the tagged commit's own workflow, so if the fix is *in* the workflow,
  move the tag instead. That is safe only while the version's JSON above still
  answers 404:

  ```bash
  git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z
  git tag -a vX.Y.Z -m "agent-flywheel X.Y.Z" && git push origin vX.Y.Z
  ```

- **"Latest" lags for a minute or two.** Right after a publish, PyPI's CDN can
  still answer the previous version. The JSON simple index that `pip` and `uv`
  read is cached separately from the HTML page, so an unpinned install can
  briefly resolve the old release. Verify pinned first.
- **The Actions billing block looks like a crash.** When the organisation's
  payments fail or its spending limit is reached, jobs are "not started" and
  fail in seconds with no logs. The reason is only in the run's annotation.
  macOS jobs bill at 10× on private repositories.
- **CI needs `NODE_SDK_READ_TOKEN`.** The `test-full` and `node-plugins` jobs
  check out the private Node repository and stop at checkout without it.
- **A version bump is a new execution identity.** The `sdk` string is part of
  the hashed descriptor, so every agent's `execution_sha256` changes on every
  release.
- **TestPyPI dry runs** (`gh workflow run workflow.yml -f target=testpypi`) need
  their own pending publisher on test.pypi.org, which is not configured.
- **The dashboard prints the install lines.** The connect page at
  launch.percepteye.ai shows `pip install` and the import. Change it whenever
  the package or import name changes.

## 3. Node release (npm)

Published as
[`@percepteye/agent-flywheel`](https://www.npmjs.com/package/@percepteye/agent-flywheel),
under the `percepteye` npm organization. npm lets only personal accounts own
an unscoped name, which is why the package is scoped. It first went out on
2026-09-14 as the unscoped `agent-flywheel`, and that 0.1.0 is still on the
registry, deprecated with a pointer to the scoped name, so nobody else can
take the name.

Releases go through the Node repository's `.github/workflows/publish.yml`, the
same shape as `workflow.yml` here. A `v*` tag runs the whole CI workflow on the
tagged commit, checks the tag against `package.json`, and publishes over OIDC.
Only the very first publish is manual.

1. **Bump `package.json`,** and set `openclaw.plugin.json` to the same version
   and description. `test/version.test.js` fails on either mismatch.
2. **Run `npm test` and `npm pack --dry-run`,** as in §1. To rehearse the whole
   pipeline on GitHub's runners, use Actions → publish → Run workflow: it runs
   every check and ends in `npm publish --dry-run`.
3. **The first publish is manual,** from a clean checkout of `main`. npm
   configures trusted publishers only on a package that already exists, so the
   workflow cannot create it. The npm account must be a member of the
   `percepteye` organization, and it must have two-factor turned on, because
   npm refuses to publish from an account without it:

   ```bash
   npm login
   npm publish --access public
   ```

   Two things to get right before this runs:
   - **The README ships inside the tarball, and npm renders it.** It must
     already read as published: no "not on npm yet" notice.
   - **Don't tag this commit.** A `v*` tag starts `publish.yml`, which fails
     on a version that already exists.

4. **Then, once, on npmjs.com → the package → Settings:**
   - **Trusted Publisher → GitHub Actions:** organization `percepteye-ai`,
     repository `agent-flywheel-node`, workflow `publish.yml`, environment
     `npm`. npm matches every field exactly, the file name included.
   - **"Require two-factor authentication and disallow tokens",** so no
     long-lived token can publish.
   - **No required reviewer on the GitHub `npm` environment yet.** On the
     Team plan, required reviewers only work on public repositories, so until
     the repository goes public ([§4d](#4d-also-on-the-day)), a `v*` tag
     publishes as soon as CI passes.
5. **Every release after that** is a merge, then a tag, in the Node repository:

   ```bash
   git switch main && git pull --ff-only
   git tag -a vX.Y.Z -m "agent-flywheel X.Y.Z" && git push origin vX.Y.Z
   ```

6. **Verify every release** with `npm view @percepteye/agent-flywheel version`.
   Then run `openclaw plugins install @percepteye/agent-flywheel` on a test
   host, and check the version `openclaw plugins list` reports.

### Node: what to know

- **Two names, on purpose.** The npm package is `@percepteye/agent-flywheel`.
  The OpenClaw plugin id and the DeepSeek Harness plugin name stay
  `agent-flywheel`, so the OpenClaw config key stays too. `cordis.patch.yml`
  loads `@percepteye/agent-flywheel/dsh`: the published package name plus the
  subpath. The two must move together, and `test/dsh.test.js` fails if they
  differ.
- **Renaming the plugin id strands existing installs.** OpenClaw keys config by
  it (`plugins.entries.agent-flywheel`), so a rename needs every install
  reinstalled and its config entry moved.
- **Some ids are internal and stay put:** the model-provider id
  `percepteye-flywheel`, the `percepteye-rollout-*` session prefix and
  `JOB_PREFIX`.
- **The tarball ships no `docs/`.** The `files` allowlist excludes it, and
  `test/oss-boundary.test.js` checks what `npm pack` produces. So the README's
  figures can render on npm only from a public URL; see §4.
- **The workflow's file name is part of the npm publisher,** exactly as on
  PyPI. Rename `publish.yml` only together with the publisher settings.
- **Provenance follows the repository's visibility.** npm cannot attest a
  build from a private repository, so `publish.yml` turns provenance off while
  the repository is private. The day it goes public, provenance switches on,
  and a publish that cannot attest where it was built fails.
- **npm is append-only too.** A version can never be published twice, not
  even after an unpublish.

## 4. Making the repositories public — not done yet

**Why.** Both READMEs point at their figures by relative path
(`docs/assets/*.gif`) and link to files in the repository the same way. GitHub
resolves those for signed-in members. PyPI cannot resolve a relative path at
all. npm rewrites relative image paths to `raw.githubusercontent.com` URLs on
the repository's default branch. But a `raw.githubusercontent.com` URL answers
404 to everyone while a repository is private, because images are fetched
without a GitHub session. So today neither registry page shows the figures,
and PyPI's sidebar links (Homepage, Documentation, Source, Issues) 404 for
anyone outside the organisation.

**Order:** audit, then flip visibility, then flip the README URLs and their
tests, then release. Do both repositories in the same window. The Node README
says of its two halves that they "flip together, never separately", and this
repository's CI checks the Node one out.

### 4a. Audit first — everything becomes public at once

- **All history and every branch,** not just `main`'s current tree. That
  includes deleted files: removing something today does not remove it from
  history.
- **Issues, PRs and Actions run logs:** descriptions, comments, review threads,
  and whatever a workflow printed.
- **The current tree:** `tests/test_oss_boundary.py` checks every tracked file
  for internal system names, internal details and local developer paths. It
  does not look at history.
- **History:** scan for secrets with a scanner such as `gitleaks detect` or
  `trufflehog git file://.`. Then search `git log -p --all` for the name lists
  in `tests/test_oss_boundary.py`.
- **Planning notes:** decide on `docs/plan-*.md` and
  `docs/nemoclaw-sandbox.md`. They are tracked, so they go public with
  everything else, and they stay in history either way.
- **Old branches:** review `byoh-fixes` and `research-tiers-2026-08`, or delete
  them first.

### 4b. Flip visibility

```bash
gh repo edit percepteye-ai/agent-flywheel      --visibility public --accept-visibility-change-consequences
gh repo edit percepteye-ai/agent-flywheel-node --visibility public --accept-visibility-change-consequences
```

(Or use Settings → General → Danger Zone → Change visibility.)

### 4c. Then one PR per repository

**Python:**
- **Make every README image and in-repo link absolute.**
  - Images: `https://raw.githubusercontent.com/percepteye-ai/agent-flywheel/main/docs/assets/…`
  - Links: `https://github.com/percepteye-ai/agent-flywheel/blob/main/…`
  - Alternative: keep the README relative and rewrite only the PyPI copy at
    build time, with a Hatch metadata hook such as `hatch-fancy-pypi-readme`
    and its regex substitutions. That is a build-time requirement, not a
    runtime dependency, so the one-dependency rule holds.
- **Update `tests/test_readme_renders_on_github.py` in the same commit.** Today
  it fails any `raw.githubusercontent.com` figure and any absolute link into
  this repository. That is deliberate: its docstring calls it "what tells you
  it is now safe to flip".
- **Restore the `ci` badge,** which is kept verbatim in the comment above the
  badges.
- **Drop `NODE_SDK_READ_TOKEN` if the Node repository is public too.** The
  cross-repository checkouts in `ci.yml` no longer need `token:`, so remove
  them and delete the secret. Pull requests from forks never receive secrets,
  so this is also what lets outside contributors' PRs run `test-full`.
- **Release a new patch version.** The corrected images and links reach PyPI
  only with a new version, because descriptions are frozen per release.

**Node:**
- **Leave the README's image paths relative.** npm already rewrites them to
  `raw.githubusercontent.com` URLs on the default branch (seen on the live
  package page on 2026-09-14). Those resolve the moment the repository is
  public, with no new release. `test/oss-boundary.test.js` holds them relative,
  and should keep doing so.
- **Nothing to change in `publish.yml`.** It reads the repository's visibility
  on every run, so the next release after the flip carries npm provenance.
  From then on, a release that cannot attest where it was built fails.

### 4d. Also on the day

- **Protect `main`** in both repositories: require a PR and the checks that
  pass today. Neither has branch protection now.
- **Add a required reviewer** to the `pypi` environment here and the `npm`
  environment in the Node repository. On the Team plan that only works once a
  repository is public. Until then, a `v*` tag publishes as soon as CI passes.
- **Actions costs change.** Standard GitHub-hosted runners are free for public
  repositories, so the minutes and spending-limit block no longer applies to
  them. Under Settings → Actions, require approval for workflows from
  first-time contributors.
- **Verify signed out,** in a private window:
  - both GitHub pages render their figures;
  - the PyPI page renders its figures, and its sidebar links resolve;
  - the npm page renders its figures;
  - `pip install agent-flywheel` still works.
