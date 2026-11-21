"""No body this SDK sends carries a key, in any shape it is printed in (PY-M1).

``redact_keys`` matched a ``pefw_`` key of EXACTLY 43 characters followed by a
non-key character, so a key printed with ``_``, a letter or a digit glued to
its end left whole; it did not know Mission Control's own ``mck_`` API keys at
all; and it rebuilt a named tuple by passing one list to its constructor,
which raised.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import NamedTuple

import pytest

from agent_flywheel._http import REDACTED_KEY, redact_keys

BODY = "A" * 43
PEFW = "pefw_" + BODY
MCK = "mck_" + "b-_9" * 10 + "xyz"


@pytest.mark.parametrize("text", [
    PEFW, PEFW + "_", PEFW + "x", PEFW + "9", PEFW + "-tail",
    f"key={PEFW}_suffix;", f"Bearer {PEFW}",
])
def test_a_flywheel_key_is_redacted_whatever_is_glued_to_it(text):
    out = redact_keys(text)
    assert BODY not in out and REDACTED_KEY in out


def test_a_mission_control_api_key_is_redacted():
    out = redact_keys(f"export MC_KEY={MCK}\n")
    assert MCK not in out and "mck_[REDACTED]" in out


def test_a_short_prefix_lookalike_is_not_a_key():
    assert redact_keys("pefw_short and mck_tiny") == "pefw_short and mck_tiny"


class _Step(NamedTuple):
    stdout: str
    code: int


def test_a_named_tuple_comes_back_the_same_named_tuple():
    out = redact_keys(_Step(f"PERCEPTEYE_API_KEY={PEFW}", 0))
    assert isinstance(out, _Step)
    assert out == _Step(f"PERCEPTEYE_API_KEY={REDACTED_KEY}", 0)


def test_a_read_only_mapping_is_redacted():
    out = redact_keys(MappingProxyType({"k": PEFW, "nested": (PEFW,)}))
    assert out == {"k": REDACTED_KEY, "nested": (REDACTED_KEY,)}
