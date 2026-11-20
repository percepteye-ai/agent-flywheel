"""A turn's identity sidecars are published whole, so parallel writers of the
SAME identity never read one half-written and call it a conflict.

Both used ``open("x")`` then ``write``: a concurrent tool call of the same turn
got ``FileExistsError``, read the still-empty file, compared ``""`` with the
identity it meant to write and wrote the permanent ``.conflict`` marker --
destroying the turn's execution fingerprint or task id. Measured before the
fix: 8 threads, one identity, a fresh directory, a false conflict ~55% of the
time.
"""
from __future__ import annotations

import threading

import pytest

from agent_flywheel.outcomes import (
    _AGENT_FINGERPRINT_CONFLICT_FILENAME,
    _TASK_ID_CONFLICT_FILENAME,
    AGENT_FINGERPRINT_FILENAME,
    TASK_ID_FILENAME,
    _read_execution_fingerprint,
    _write_execution_fingerprint,
    _write_task_id,
    read_task_id,
)

DIGEST = "a" * 64
OTHER = "b" * 64

WRITERS = [
    pytest.param(_write_execution_fingerprint, DIGEST, OTHER,
                 AGENT_FINGERPRINT_FILENAME, _AGENT_FINGERPRINT_CONFLICT_FILENAME,
                 lambda d: _read_execution_fingerprint(d),
                 {"execution_sha256": DIGEST}, id="execution-fingerprint"),
    pytest.param(_write_task_id, "ticket-4417", "ticket-9",
                 TASK_ID_FILENAME, _TASK_ID_CONFLICT_FILENAME,
                 read_task_id, "ticket-4417", id="task-id"),
]


def _race(write, directory, value, threads=8):
    barrier = threading.Barrier(threads)

    def run():
        barrier.wait()
        write(directory, value)

    workers = [threading.Thread(target=run) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


@pytest.mark.parametrize(
    "write,value,other,name,conflict,read,expected", WRITERS)
def test_parallel_writers_of_one_identity_never_conflict(
        tmp_path, write, value, other, name, conflict, read, expected):
    for round_ in range(60):
        turn = tmp_path / f"t{round_}"
        turn.mkdir()
        _race(write, turn, value)
        assert not (turn / conflict).exists(), (
            f"round {round_}: a false conflict destroyed the identity")
        assert read(turn) == expected
        assert sorted(p.name for p in turn.iterdir()) == [name], (
            "a writer's own temporary file was left behind")


@pytest.mark.parametrize(
    "write,value,other,name,conflict,read,expected", WRITERS)
def test_a_different_identity_is_still_a_conflict(
        tmp_path, write, value, other, name, conflict, read, expected):
    write(tmp_path, value)
    write(tmp_path, other)
    assert (tmp_path / conflict).exists()
    assert read(tmp_path) is None
