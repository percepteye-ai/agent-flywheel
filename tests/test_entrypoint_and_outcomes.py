"""Entrypoint resolution (at registration) and the outcome ledger."""
from __future__ import annotations

import os

import pytest

from percepteye_flywheel import record_tool_call, resolve_entrypoint
from percepteye_flywheel.errors import EntrypointError
from percepteye_flywheel.outcomes import LEDGER_DIR_ENV, ROLLOUT_ID_ENV, OutcomeLedger


class Invoking:
    def invoke(self, x):
        return f"invoke:{x}"

    def run(self, x):
        return f"run:{x}"


class OnlyChat:
    def chat(self, x):
        return f"chat:{x}"


class Opaque:
    value = 3


# ── resolution ──────────────────────────────────────────────────────────────

def test_resolution_order_prefers_invoke():
    r = resolve_entrypoint(Invoking(), input_shape="text")
    assert r.name == "invoke"
    assert r.fn("a") == "invoke:a"


def test_falls_through_to_chat():
    assert resolve_entrypoint(OnlyChat(), input_shape="text").name == "chat"


def test_explicit_method_overrides_the_probe():
    assert resolve_entrypoint(Invoking(), input_shape="text", method="run").name == "run"


def test_missing_entrypoint_raises_at_registration_and_says_what_it_tried():
    with pytest.raises(EntrypointError) as exc:
        resolve_entrypoint(Opaque(), input_shape="text")
    assert "invoke" in str(exc.value) and "chat" in str(exc.value)


def test_a_bad_explicit_method_raises_rather_than_falling_back():
    """Silently falling back would run a DIFFERENT function than the caller
    named, which is worse than failing."""
    with pytest.raises(EntrypointError):
        resolve_entrypoint(Invoking(), input_shape="text", method="nope")


def test_plain_callable_is_its_own_entrypoint():
    assert resolve_entrypoint(lambda x: x, input_shape="text").fn("z") == "z"


def test_input_shape_is_declared_not_inferred():
    assert resolve_entrypoint(Invoking(), input_shape="text").build_arg("hi") == "hi"
    assert resolve_entrypoint(Invoking(), input_shape="messages").build_arg("hi") == [
        {"role": "user", "content": "hi"}
    ]
    assert resolve_entrypoint(Invoking(), input_shape="dict").build_arg("hi") == {"input": "hi"}


def test_bad_input_shape_is_rejected():
    with pytest.raises(EntrypointError):
        resolve_entrypoint(Invoking(), input_shape="freeform")  # type: ignore[arg-type]


def test_async_entrypoints_are_detected():
    class A:
        async def invoke(self, x):
            return x

    assert resolve_entrypoint(A(), input_shape="text").is_async is True


# ── the ledger ──────────────────────────────────────────────────────────────

def test_record_outside_a_rollout_is_a_noop(monkeypatch, tmp_path):
    """Instrumentation can live in production code paths unconditionally."""
    monkeypatch.delenv(ROLLOUT_ID_ENV, raising=False)
    monkeypatch.setenv(LEDGER_DIR_ENV, str(tmp_path))
    record_tool_call("x", {}, outcome="ok", status_code=200)
    assert not list(tmp_path.iterdir())


def test_ledger_round_trips_through_a_file(monkeypatch, tmp_path):
    """A file, not a list: the customer's agent runs in a CHILD process, so
    in-memory state would record nothing while appearing to work."""
    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_1")
    monkeypatch.setenv(LEDGER_DIR_ENV, str(tmp_path))
    record_tool_call("create", {"a": 1}, outcome="ok", status_code=201)
    record_tool_call("charge", {}, outcome="failed", status_code=200, error="app_auth_error")
    record_tool_call("opaque", {}, outcome="unknown", status_code=200)

    calls = OutcomeLedger(tmp_path, "rl_1").read()
    assert calls is not None and len(calls) == 3
    assert (calls[0].outcome, calls[0].status_code) == ("ok", 201)
    assert (calls[1].outcome, calls[1].status_code) == ("failed", 200)
    # The unknown call was scrubbed on the way in and stays scrubbed.
    assert (calls[2].outcome, calls[2].status_code) == ("unknown", None)


def test_nothing_recorded_reads_as_none_not_empty(tmp_path):
    """None means 'not reported'; [] would assert 'zero calls occurred'."""
    assert OutcomeLedger(tmp_path, "rl_missing").read() is None


def test_ledger_is_per_rollout(monkeypatch, tmp_path):
    monkeypatch.setenv(LEDGER_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_a")
    record_tool_call("a", {}, outcome="ok", status_code=200)
    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_b")
    record_tool_call("b", {}, outcome="ok", status_code=200)

    a = OutcomeLedger(tmp_path, "rl_a").read() or []
    b = OutcomeLedger(tmp_path, "rl_b").read() or []
    assert [c.name for c in a] == ["a"]
    assert [c.name for c in b] == ["b"]


def test_unknown_count_is_surfaceable(monkeypatch, tmp_path):
    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_u")
    monkeypatch.setenv(LEDGER_DIR_ENV, str(tmp_path))
    record_tool_call("a", {}, outcome="ok", status_code=200)
    record_tool_call("b", {}, outcome="unknown", status_code=None)
    assert OutcomeLedger(tmp_path, "rl_u").unknown_count() == 1


def test_env_rollout_id_is_what_a_child_inherits(monkeypatch, tmp_path):
    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_from_env")
    monkeypatch.setenv(LEDGER_DIR_ENV, str(tmp_path))
    record_tool_call("x", {}, outcome="ok", status_code=200)
    assert os.path.exists(tmp_path / "rl_from_env" / "tool_calls.jsonl")
