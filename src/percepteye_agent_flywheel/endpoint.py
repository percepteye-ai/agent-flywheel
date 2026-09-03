"""Which LLM endpoint the agent talks to — owned by the MODE, not by us.

THE PROBLEM THIS SOLVES. Capturing a completion means the agent's traffic has to
reach our gateway, and the only place a CLI agent's endpoint lives is its own
config. Writing that config once, at install, would leave a customer permanently
routed through us with no way back short of hand-editing YAML — which is a thing
we would be doing TO them rather than FOR them.

So the endpoint is a FUNCTION OF THE MODE, and the mode is one environment
variable the customer already owns:

    PERCEPTEYE_AGENT_MODE=training     -> our gateway (completions captured)
    PERCEPTEYE_AGENT_MODE=production   -> whatever they had before we touched it

``training`` is the default, and it is the default that makes the flywheel work
out of the box. ``production`` hands the endpoint back. Nothing else in the
install changes, and no third command exists.

WHY A LINE EDITOR AND NOT A YAML PARSER. This package declares exactly one
runtime dependency (httpx). Adding PyYAML to rewrite one scalar would put a
second one in every customer's environment, and a full parse-and-dump would also
REFORMAT a file the customer wrote — dropping their comments and re-ordering
their keys as a side effect of us changing an endpoint. ``set_scalar`` rewrites
one line and leaves every other byte alone.

WHY ``OPENAI_BASE_URL`` AND NOT ``OPENAI_HOST``. Read from goose
``providers/openai_def.rs::resolve_base_url``, highest precedence first:

    1. OPENAI_HOST      env
    2. OPENAI_BASE_URL  env
    3. OPENAI_BASE_URL  config      <- what we write
    4. OPENAI_HOST      config
    5. https://api.openai.com

Writing ``OPENAI_HOST`` in the config would sit at rank 4 and be silently
overridden for any customer who has ``OPENAI_BASE_URL`` set — routing that
appears installed and never takes effect. Rank 3 is the highest rank we can
occupy without reaching into the customer's environment.

THAT THE TWO ENV RANKS OUTRANK US IS THE FEATURE. A customer who exports
``OPENAI_BASE_URL`` wins over this file, immediately, with no command and no
uninstall. That is the escape hatch, and it costs us nothing to honour.

THE ONE TIMING PROPERTY, said out loud because it is a real constraint and not
an implementation detail: goose resolves its endpoint ONCE, at process start.
MEASURED against goose 1.48.0 — a SessionStart hook rewrote the config and the
completion still went to the endpoint present at startup (1 hit against 0). So a
mode flip takes effect on the NEXT session, not the running one. The hook
reconciles on every SessionStart, which is what makes it self-correcting rather
than a state someone has to remember to repair.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

#: The config key we occupy. See the module docstring for why this one.
GOOSE_ENDPOINT_KEY = "OPENAI_BASE_URL"

#: An explicit gateway URL. OPTIONAL, and deliberately not part of the
#: three-step install: the control plane is meant to name the agent-keyed
#: capture endpoint, and until it does, an operator can point this at one by
#: hand. Unset means "leave routing alone" — see :func:`plan`.
TRAINING_URL_ENV = "PERCEPTEYE_LLM_BASE_URL"

#: The env ranks that outrank the config file. Read to WARN, never to fight:
#: if one of these is set, whatever we write is inert, and saying so is the
#: difference between a customer debugging for an hour and reading one line.
_OVERRIDING_ENV = ("OPENAI_HOST", "OPENAI_BASE_URL")

#: A top-level ``key: value`` line. Anchored at column 0 on purpose — an
#: indented ``OPENAI_BASE_URL:`` is a key inside somebody else's mapping and is
#: none of our business.
_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*:(.*)$")


def goose_config_path() -> Path:
    """Where goose keeps its config, honouring ``GOOSE_PATH_ROOT``.

    Same override as :func:`install_hooks.goose_plugins_dir`, for the same
    reason: it is what an isolated test and a relocated install both rely on.
    Measured — setting ``GOOSE_PATH_ROOT`` relocates the CONFIG too, not just
    the plugins directory, which is how a demo run lost its provider.
    """
    root = os.environ.get("GOOSE_PATH_ROOT")
    base = Path(root) if root and Path(root).is_absolute() else Path.home()
    return base / ".config" / "goose" / "config.yaml"


def _unquote(raw: str) -> str:
    """The scalar value of one YAML line's right-hand side.

    Handles the three shapes a URL is ever written in — bare, single-quoted,
    double-quoted — and strips a trailing ``# comment`` only from the BARE
    form, because a ``#`` inside quotes is part of the value.
    """
    s = raw.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s.split(" #", 1)[0].strip()


def read_scalar(text: str, key: str) -> str | None:
    """The value of top-level ``key``, or ``None`` if it is absent or empty."""
    for line in text.splitlines():
        m = _KEY_LINE.match(line)
        if m and m.group(1) == key:
            value = _unquote(m.group(2))
            return value or None
    return None


def set_scalar(text: str, key: str, value: str | None) -> str:
    """``text`` with top-level ``key`` set to ``value``, or removed for ``None``.

    Every other byte is preserved: comments, ordering, indentation, and the
    customer's own keys. Only the matching line is rewritten, and a key that is
    absent is APPENDED rather than causing a rewrite of the document.

    Removal is not the same as setting an empty string. goose filters an empty
    ``OPENAI_BASE_URL`` back out (``.filter(|s| !s.is_empty())``), so both would
    happen to work today — but an empty key left in the file reads to a human as
    "PerceptEye set this to nothing", and to the next reader of the config as
    ours to keep updating. Absence is the honest record of "we are not here".
    """
    out, found = [], False
    for line in text.splitlines():
        m = _KEY_LINE.match(line)
        if m and m.group(1) == key:
            if found:
                continue          # a duplicate key: YAML takes the first
            found = True
            if value is None:
                continue          # drop the line entirely
            out.append(f'{key}: "{value}"')
            continue
        out.append(line)
    if not found and value is not None:
        out.append(f'{key}: "{value}"')
    body = "\n".join(out)
    return body + "\n" if body else ""


def write_config(path: Path, text: str) -> None:
    """Replace ``path`` atomically, creating its directory if needed.

    A hook process and the customer's own editor can both be holding this file.
    A truncate-then-write would give a concurrent reader an empty or half-written
    config — which for THIS file means goose starting with no provider. The
    rename is atomic, so a reader sees the old bytes or the new ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".percepteye-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        # Leave no debris on a failure path; the config itself is untouched
        # because the rename is what would have changed it.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def training_base_url() -> str | None:
    """The gateway URL to route at in training mode, or ``None`` if unknown.

    ``None`` means LEAVE ROUTING ALONE. It is not an error and it is not a
    fallback to a guess: an endpoint we invent would send the customer's traffic
    — and their API key — somewhere they never named. The honest behaviour when
    we do not know where our own gateway is, is to change nothing and say so.
    """
    raw = os.environ.get(TRAINING_URL_ENV)
    return raw.strip() or None if raw else None


def overriding_env() -> str | None:
    """The name of an env var that outranks the config, if one is set."""
    for name in _OVERRIDING_ENV:
        raw = os.environ.get(name)
        if raw and raw.strip():
            return name
    return None


def plan(mode: str, *, original: str | None,
         training_url: str | None = None,
         has_key: bool | None = None) -> dict[str, Any]:
    """What the config should say, and why. Pure — reads no files.

    ``original`` is the value the config carried BEFORE we first wrote to it,
    captured at install and carried in the stamp. It is the whole restore path:
    production does not write a default or clear the key blindly, it puts back
    the thing that was there, including putting back "nothing".

    TRAINING IS NOT ENOUGH ON ITS OWN; there must also be a key. Routing at our
    gateway without one sends every completion to an endpoint that will refuse
    it, so the customer's agent stops working and the visible cause is goose.
    That is the exact install the README calls the ordinary one -- "step 3 is
    optional, and with no key nothing leaves the machine" -- so the keyless case
    is not an edge, it is the default first run. ``training`` is the mode; the
    key is the CONSENT, and routing needs both.
    """
    if mode == "production":
        return {"mode": mode, "value": original, "routed": False,
                "reason": "production: the customer's own endpoint"}
    keyed = has_key if has_key is not None else bool(
        (os.environ.get("PERCEPTEYE_API_KEY") or "").strip())
    if not keyed:
        return {"mode": mode, "value": original, "routed": False,
                "reason": "training, but PERCEPTEYE_API_KEY is unset — "
                          "recording locally; routing left unchanged"}
    url = training_url if training_url is not None else training_base_url()
    if not url:
        return {"mode": mode, "value": original, "routed": False,
                "reason": f"training, but no gateway URL is known "
                          f"(set {TRAINING_URL_ENV}); routing left unchanged"}
    return {"mode": mode, "value": url, "routed": True,
            "reason": "training: completions captured for the flywheel"}
