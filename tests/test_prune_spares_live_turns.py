"""Retention never deletes a turn that is still running, never raises into
it, counts a turn lost only once it is gone, and never erases an upload
marker it did not mean to.

``turn()`` prunes BEFORE the caller's body runs. With ``retention_turns=0`` --
documented as valid -- the turn about to run was deleted under the code
writing to it; under ``-W error`` the unsent-turn warning raised out of the
customer's ``with px.turn()``; a failed ``rmtree`` was still counted as lost
data; and ``.uploaded`` was rewritten in place, without the flush lock, from a
snapshot, so an id a flush appended meanwhile was erased.
"""
from __future__ import annotations

import shutil
import warnings

import agent_flywheel as fw
from agent_flywheel import production
from agent_flywheel.outcomes import record_tool_call

from .test_production_turn_upload import _attached, _Transport


def _call():
    record_tool_call(name="lookup", arguments={}, outcome="ok",
                     status_code=200)


def test_retention_zero_never_deletes_the_turn_that_is_running(
        monkeypatch, tmp_path):
    monkeypatch.setattr(production, "_PRUNE_EVERY", 1)
    monkeypatch.setattr(fw.Attachment, "_flush_soon", lambda self: None)
    with _attached(monkeypatch, tmp_path, _Transport(), retention_turns=0) as px:
        with warnings.catch_warnings():
            warnings.simplefilter("error")       # a customer's -W error
            for n in range(3):
                with px.turn(turn_id=f"t{n}", input_text="q") as tid:
                    _call()
                    assert (tmp_path / tid).is_dir(), (
                        "the running turn's directory was deleted under it")
                    assert px.read(tid), "its call was not recorded"


def test_a_turn_retention_could_not_remove_is_not_counted_lost(
        monkeypatch, tmp_path):
    monkeypatch.setattr(fw.Attachment, "_flush_soon", lambda self: None)
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        for n in range(3):
            with px.turn(turn_id=f"t{n}", input_text="q"):
                _call()
        monkeypatch.setattr(shutil, "rmtree",
                            lambda p: (_ for _ in ()).throw(OSError("busy")))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert px.prune(keep_last=0) == 0
        assert not [w for w in caught if "retention deleted" in str(w.message)]


def test_pruning_keeps_every_marker_but_the_removed_turns(
        monkeypatch, tmp_path):
    monkeypatch.setattr(fw.Attachment, "_flush_soon", lambda self: None)
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        for n in range(3):
            with px.turn(turn_id=f"t{n}", input_text="q"):
                _call()
            px.record_answer(f"t{n}", "a")
        assert px.flush()["sent"] == 3
        real = shutil.rmtree

        def rmtree_while_a_flush_lands(path, *a, **kw):
            # A turn uploaded after prune listed the tree.
            px._mark_uploaded(["t-late"])
            if path.name == "t1":
                raise OSError("busy")             # this one stays on disk
            return real(path, *a, **kw)

        monkeypatch.setattr(shutil, "rmtree", rmtree_while_a_flush_lands)
        px.prune(keep_last=0)
        ids = px._uploaded_ids()
        assert "t-late" in ids, "an id appended meanwhile was erased"
        assert "t1" in ids, "a turn still on disk lost its marker"
        assert not {"t0", "t2"} & ids
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
