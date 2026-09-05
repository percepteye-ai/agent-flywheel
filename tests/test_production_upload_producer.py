"""Something actually CALLS flush, without the customer doing it.

THE GAP THESE PIN. ``flush()`` existed and worked, ``close()`` called it, and
``__exit__`` called ``close()`` -- so every test of the upload path passed while
the path had no automatic producer at all. The documented usage is::

    px = fw.attach(...)                       # held for the life of the process
    with px.turn(conversation_id=c) as tid:   # a scope per REQUEST
        ...

which never exits a scope on the ``Attachment`` itself. A long-running server
following this package's own README captured turns forever and uploaded none of
them, and nothing went red, because every existing test either called
``flush()`` by hand or used ``close()`` in a fixture.

So these do neither. They drive turns the way a server does and assert that the
data leaves anyway -- which is the property the customer actually has.
"""
from __future__ import annotations

import atexit
import gc
import threading
import time
import weakref
from contextlib import contextmanager

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.outcomes import record_tool_call
from percepteye_agent_flywheel.production import _FLUSH_EVERY


class _Transport:
    def __init__(self, *, capture=True, slow=None):
        self.batches: list[list[dict]] = []
        self._capture = capture
        self._slow = slow
        self._lock = threading.Lock()
        #: How many report_turns calls were in flight at once. Two would mean
        #: the same turns could be sent twice before either wrote its marker.
        self.max_concurrent = 0
        self._in_flight = 0

    def register(self, spec):
        return {"capture": {"turns": self._capture}}

    def policy_current(self):
        return {}

    def report_turns(self, agent_id, turns):
        with self._lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            if self._slow:
                time.sleep(self._slow)
            with self._lock:
                self.batches.append(list(turns))
            return {"agent_id": agent_id, "turns": []}
        finally:
            with self._lock:
                self._in_flight -= 1

    @property
    def sent(self):
        with self._lock:
            return [t for b in self.batches for t in b]


@contextmanager
def _attached(monkeypatch, tmp_path, transport, **kw):
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    monkeypatch.setattr(
        "percepteye_agent_flywheel.transport.AttachTransport",
        lambda client, agent_id: transport,
    )
    monkeypatch.setattr(
        "percepteye_agent_flywheel._http.ControlPlaneClient.close",
        lambda self: None,
    )
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path), **kw)
    try:
        yield px
    finally:
        px.close()


def _serve(px, n, *, answer=True, start=0):
    """Drive `n` turns the way a server does: a scope per request, nothing else."""
    for i in range(start, start + n):
        tid = f"t{i}"
        with px.turn(turn_id=tid, input_text=f"ask {i}", conversation_id="c1"):
            record_tool_call(name="lookup", arguments={"i": i},
                             outcome="ok", status_code=200)
        if answer:
            px.record_answer(tid, f"answer {i}")


def _until(predicate, timeout=10.0):
    """Wait for the background flush. Returns whether it happened."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ── the headline ──────────────────────────────────────────────────────────


def test_a_server_that_never_closes_still_uploads(monkeypatch, tmp_path):
    """THE regression. No flush(), no close(), no context manager on `px`."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY)
        assert _until(lambda: tr.sent), (
            "a long-running attachment uploaded nothing; the periodic producer "
            "is gone and flush() is once again reachable only from close()"
        )
        ids = {t["turn_id"] for t in tr.sent}
        assert "t0" in ids


def test_it_does_not_upload_before_the_cadence_is_due(monkeypatch, tmp_path):
    """Not every turn. The network is not on the request path that often."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY - 1)
        # Give a wrongly-eager producer time to prove itself.
        assert not _until(lambda: tr.sent, timeout=0.5)


def test_the_upload_is_off_the_request_path(monkeypatch, tmp_path):
    """A slow control plane must not slow a customer's user down.

    The turn that trips the cadence returns while the upload is still running;
    if the flush were inline, this would take at least the transport's delay.
    """
    tr = _Transport(slow=1.0)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY - 1)
        start = time.monotonic()
        _serve(px, 1, start=_FLUSH_EVERY)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, (
            f"the turn that triggered the flush took {elapsed:.2f}s; the "
            f"upload is blocking the request path"
        )


def test_unanswered_turns_are_held_back_by_the_periodic_flush(
        monkeypatch, tmp_path):
    """An answer may still be coming; only shutdown knows it is not."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY, answer=False)
        assert not _until(lambda: tr.sent, timeout=0.5)
        # ...and close() sends them, because nothing more is coming.
        px.close()
    assert {t["turn_id"] for t in tr.sent}
    assert all(t["final_text"] is None for t in tr.sent)


# ── two flushes must not send one turn twice ──────────────────────────────


def test_a_manual_flush_cannot_race_the_automatic_one(monkeypatch, tmp_path):
    tr = _Transport(slow=0.3)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY)
        px.flush()                      # straight into the background one
        assert _until(lambda: tr.sent)
        assert tr.max_concurrent <= 1, (
            "two uploads ran at once; both read the same unsent turns before "
            "either wrote its marker"
        )
        ids = [t["turn_id"] for t in tr.sent]
        assert len(ids) == len(set(ids)), f"a turn was uploaded twice: {ids}"


def test_a_second_background_flush_does_not_pile_up(monkeypatch, tmp_path):
    tr = _Transport(slow=0.3)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY)
        px._flush_soon()
        px._flush_soon()
        assert _until(lambda: tr.sent)
        assert tr.max_concurrent <= 1


# ── the trigger refuses when it must ──────────────────────────────────────


def test_no_upload_without_consent(monkeypatch, tmp_path):
    tr = _Transport(capture=False)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, _FLUSH_EVERY)
        assert not _until(lambda: tr.sent, timeout=0.5)
    assert tr.sent == []


def test_a_closed_attachment_starts_no_thread(monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.close()
        before = threading.active_count()
        px._flush_soon()
        assert threading.active_count() <= before


# ── shutdown ──────────────────────────────────────────────────────────────


def test_shutdown_flush_is_registered_only_with_consent(monkeypatch, tmp_path):
    """An agent with no consent must not add an exit hook that does nothing.

    Counted rather than inspected: `atexit` exposes no public registry, and
    reading the private one would pin an implementation detail of CPython.
    """
    seen: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: seen.append(fn) or fn)

    with _attached(monkeypatch, tmp_path, _Transport(capture=False)):
        pass
    # ABSOLUTE, NOT RELATIVE. This asserted `len(seen) == without + 1` across
    # the two cases, which is true whether or not the guard exists -- both
    # register, and the difference is still one. A mutation removing the guard
    # entirely left it green: a proxy that survives the bug it names.
    assert seen == [], (
        "an exit hook was registered for an agent with NO capture consent; it "
        "can only ever be a no-op, and it makes every process that merely "
        "imported the SDK look like one that talks to us at shutdown"
    )

    with _attached(monkeypatch, tmp_path, _Transport(capture=True)):
        pass
    assert len(seen) == 1, (
        "the shutdown flush was not registered for an agent that MAY upload"
    )


def test_the_shutdown_hook_does_not_keep_the_attachment_alive(
        monkeypatch, tmp_path):
    """Registering it must not be what pins the object for the process's life."""
    hooks: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: hooks.append(fn) or fn)

    px = None
    with _attached(monkeypatch, tmp_path, _Transport()) as attached:
        px = attached
    ref = weakref.ref(px)
    del px, attached
    gc.collect()
    assert ref() is None, (
        "the atexit hook holds a strong reference; every attach() would leak "
        "its Attachment, its policy payload and its adapter unbinds"
    )
    # And the orphaned hook is still safe to run.
    for fn in hooks:
        fn()


def test_close_does_not_hang_on_an_unresponsive_control_plane(
        monkeypatch, tmp_path):
    """Shutdown is bounded, even mid-upload.

    `close()` also runs from the atexit hook. A periodic flush can be holding
    the upload lock with a request that takes the HTTP client's full timeout,
    and waiting on it unbounded would let an unresponsive control plane keep a
    customer's process from exiting. A turn not sent now is still on disk.
    """
    from percepteye_agent_flywheel.production import _CLOSE_FLUSH_GRACE_S

    tr = _Transport(slow=30.0)
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    monkeypatch.setattr(
        "percepteye_agent_flywheel.transport.AttachTransport",
        lambda client, agent_id: tr,
    )
    monkeypatch.setattr(
        "percepteye_agent_flywheel._http.ControlPlaneClient.close",
        lambda self: None,
    )
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    _serve(px, _FLUSH_EVERY)
    assert _until(lambda: tr.max_concurrent >= 1, timeout=5.0), (
        "the background flush never started, so this proves nothing"
    )

    start = time.monotonic()
    px.close()
    elapsed = time.monotonic() - start
    assert elapsed < _CLOSE_FLUSH_GRACE_S + 3.0, (
        f"close() took {elapsed:.1f}s waiting on an in-flight upload; a hung "
        f"control plane would hold the customer's process open at exit"
    )


# ── retention must not silently delete what was going to be uploaded ──────


def _dirs(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def test_retention_deletes_UPLOADED_turns_before_pending_ones(
        monkeypatch, tmp_path):
    """The defect: retention ordered by mtime alone, upload state ignored.

    The periodic flush deliberately holds back a turn with no answer, because
    one may still be coming. Retention deleted by age regardless -- so on an
    agent with `retention_turns` set, a turn whose answer was still coming
    could be removed before it was ever sent. Silently and permanently.

    The same NUMBER is still deleted; only which ones.
    """
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        # THE PENDING TURNS ARE THE OLDEST ONES, which is what forces the
        # choice -- and is the real shape: a turn still waiting on its answer
        # while newer traffic flows through and uploads cleanly. An earlier
        # version of this test made the pending turns the NEWEST, so mtime
        # alone already kept them and the assertion held with the fix deleted.
        _serve(px, 2, answer=False, start=100)          # t100, t101: oldest
        _serve(px, 4, answer=True)                       # t0..t3: newer
        px.flush()
        assert len(tr.sent) == 4, "only the answered turns should have gone"
        assert len(_dirs(tmp_path)) == 6

        # Room for two. By mtime that is t3 and t2 -- both already uploaded --
        # and the two pending turns would be destroyed.
        px.prune(keep_last=2)
        kept = set(_dirs(tmp_path))
        assert {"t100", "t101"} == kept, (
            f"retention deleted turns that had never been uploaded while "
            f"already-uploaded ones remained: kept {sorted(kept)}"
        )
        assert len(kept) == 2, "retention must still bound the directory"


def test_it_warns_when_retention_really_does_lose_an_unsent_turn(
        monkeypatch, tmp_path):
    """Legitimate -- the customer asked for a bounded directory -- but not silent.

    An operator who is not told will go looking for those turns upstream.
    """
    import warnings as _w

    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, 3, answer=False)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            px.prune(keep_last=1)
        assert any("had not been uploaded" in str(c.message) for c in caught), (
            "retention destroyed unsent turns and said nothing"
        )


def test_no_warning_when_everything_deleted_was_already_sent(
        monkeypatch, tmp_path):
    """The ordinary case must stay quiet, or the warning means nothing."""
    import warnings as _w

    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _serve(px, 3, answer=True)
        px.flush()
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            px.prune(keep_last=1)
        assert not [c for c in caught
                    if "had not been uploaded" in str(c.message)]
