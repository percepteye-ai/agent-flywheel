"""``attach()`` -- capture in the process that serves your users.

:func:`~percepteye_agent_flywheel.serve` contributes CAPACITY: it claims tasks from
the control plane and runs your agent against them in a child process. This
module contributes EVIDENCE instead, from the turns your agent is already serving.
Same adapters, same tri-state, same trajectory file, same absence contract --
scoped around a live user turn rather than a claimed rollout.

**It captures. It never reroutes.** Those are different jobs and mixing them
would be a correctness bug, not an inconvenience: pointing a production agent
at a new endpoint is a decision with user-visible consequences, and it belongs
to :func:`~percepteye_agent_flywheel.apply_current_policy`, where the customer
makes it explicitly. ``attach`` binds the adapters with no endpoint at all,
which the adapters handle by design -- ``LangChainAdapter._route`` returns
immediately when it was handed no chat model, and the ADK adapter ignores the
URL entirely.

Three rules govern every line here, and they are the same three the rest of the
package keeps:

* a capture bug costs the observation and nothing else -- never the turn;
* an unreachable control plane never stops the agent serving its users;
* absence is never collapsed into zero.

**Uploading is OPT-IN, per agent, and off by default.** A production turn is
your END USERS' own message and the answer you gave them, so nothing here
decides to send it. :meth:`Attachment.flush` uploads only when the control
plane said this agent is enabled for capture -- a per-agent switch an operator
sets, which the registration response reports back. Until then turns
accumulate as trajectory directories under the root you name and go nowhere,
which is exactly what they did before the route existed.

**When it uploads.** ``flush()`` sends turns that have an answer; ``close()``
flushes once more and sends the rest, including turns whose answer never
arrived -- those carry real tool calls and the control plane stores them,
flagged and excluded from anything needing a completion. Nothing is sent at
turn-scope exit, because :meth:`Attachment.record_answer` is callable after
the scope closes and an upload there would ship every turn before its answer
existed.

**What it never does.** It never raises into your process, and it never blocks
a turn on the network: a control plane that is unreachable leaves the turns on
disk for the next flush.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

# NOT imported at module scope. `_http` pulls in httpx, which is 83-87% of
# this package's import time and ~88% of its resident memory -- and a process
# that only reads the promoted policy, or only captures locally, never opens a
# socket at all. Deferring it means those customers stop paying for a transport
# they do not use. `policy.py` has always done this; these were the paths that
# forced it anyway.
if TYPE_CHECKING:
    from ._http import ControlPlaneClient
from .errors import ConfigurationError
from .outcomes import (
    Trajectory,
    _release,
    _RolloutScope,
    read_conversation_id,
    read_turn_text,
    record_turn_text,
)

__all__ = ["Attachment", "attach"]

ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"
ENV_API_KEY = "PERCEPTEYE_API_KEY"


#: How many turns pass between automatic prunes when `retention_turns` is set.
#: Amortises one directory scan over many turns; the ceiling is therefore
#: `retention_turns + _PRUNE_EVERY`, not `retention_turns` exactly.
_PRUNE_EVERY = 256


class Attachment:
    """A bound capture session for one process. Returned by :func:`attach`.

    Construct it through :func:`attach`; the adapters are already bound by the
    time you hold one.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        trajectory_root: str,
        policy: dict[str, Any],
        unbinds: list[Any],
        adapters_bound: tuple[str, ...],
        retention_turns: int | None = None,
        control_plane_url: str | None = None,
        api_key: str | None = None,
        may_upload: bool = False,
        upload_refusal: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.trajectory_root = trajectory_root
        #: What the control plane said this agent should be running, or ``{}``
        #: when nothing is deployed OR the control plane could not be reached.
        #: Both read the same on purpose: keep your current configuration.
        #: INFORMATIONAL here -- ``attach`` does not act on it. Applying it is
        #: ``apply_current_policy``'s job and the customer's decision.
        self.policy = policy
        #: Which adapters actually bound. ``()`` is a real answer -- an agent
        #: using no supported framework captures nothing automatically, and
        #: saying so is better than implying coverage that is not there.
        self.adapters_bound = adapters_bound
        self._unbinds = unbinds
        self._closed = False
        #: When set, `prune(keep_last=...)` runs itself. Checked every
        #: _PRUNE_EVERY turns rather than every turn, because pruning is a
        #: directory scan and doing one per user request would cost more than
        #: the capture it is cleaning up after.
        self._retention_turns = retention_turns
        self._turns_since_prune = 0
        self._lock = threading.Lock()
        self._cp_url = control_plane_url
        self._api_key = api_key
        #: What the control plane SAID at registration, not what we assume.
        #: Capture is opt-in per agent because a turn carries this agent's end
        #: users' own messages -- so the default is False, and finding out by
        #: uploading would mean sending the data whose permission is in
        #: question.
        self.may_upload = may_upload
        #: Why not, in a sentence, when it is False and the control plane said.
        self.upload_refusal = upload_refusal

    @contextmanager
    def turn(self, *, turn_id: str | None = None,
             conversation_id: str | None = None,
             input_text: str | None = None) -> Iterator[str]:
        """Scope one user turn. Tool calls inside it land in its own trajectory.

        Yields the turn id, which the caller should keep: it is the join key
        for any outcome that arrives later than the turn does -- the draft was
        kept an hour afterwards, the pull request merged the next morning.

        A turn boundary is DECLARED rather than inferred. Guessing where one
        request ends and the next begins is the same class of guess this SDK
        refuses everywhere else, and getting it wrong silently merges two
        users' tool calls into one record.

        Nothing in here can fail the turn. A trajectory that cannot be written
        disables capture for itself and says so once; the body still runs.

        **A turn that records nothing leaves nothing behind.** The directory is
        created by the first write, not by entering the turn. It used to be
        created here, unconditionally, so an agent that answered from context
        without calling a tool still cost one directory -- and at a million
        turns that was a third of a gigabyte and a million inodes of empty
        directories, for turns that had nothing to say.

        ``input_text`` records what the user ASKED, and :meth:`record_answer`
        records what the agent finally said. Both are OPTIONAL and both must be
        passed by the caller, because nothing here can see them: the adapters
        declare ``reports_outcomes=True`` and observe tool calls, not turn
        content, so a prompt and a completion are facts only the customer's own
        code holds. Without them a turn records its tool calls and nothing
        else, which is exactly what it did before -- and is why tool outcomes
        stay zero-instrumentation while turn text is opt-in.
        """
        tid = turn_id or f"t_{uuid.uuid4().hex[:16]}"
        path = Path(self.trajectory_root) / tid
        if conversation_id:
            # The one thing that still needs the directory up front, because it
            # writes a file whether or not a tool is ever called.
            try:
                path.mkdir(parents=True, exist_ok=True)
                (path / "conversation_id").write_text(
                    str(conversation_id), encoding="utf-8")
            except OSError:
                pass
        if input_text:
            record_turn_text(path, "task", input_text,
                             turn_id=tid, conversation_id=conversation_id or "")
        if self._retention_turns is not None:
            with self._lock:
                self._turns_since_prune += 1
                due = self._turns_since_prune >= _PRUNE_EVERY
                if due:
                    self._turns_since_prune = 0
            if due:
                self.prune(keep_last=self._retention_turns)

        with _RolloutScope(tid, str(path), kind="turn",
                           conversation_id=conversation_id):
            yield tid

    def record_answer(self, turn_id: str, text: str) -> bool:
        """Record what the agent finally answered for one turn.

        SEPARATE FROM ``turn(input_text=...)`` because the answer does not
        exist yet when the turn opens. Callable after the scope has closed,
        for the same reason :meth:`turn` yields its id: an outcome can arrive
        later than the turn does.

        Returns whether anything was written, and never raises -- a capture bug
        costs the observation and never the turn.
        """
        return record_turn_text(
            Path(self.trajectory_root) / turn_id, "answer", text,
            turn_id=turn_id,
        )

    def read_turn(self, turn_id: str) -> dict[str, str] | None:
        """The task and answer recorded for one turn, or ``None``.

        ``None`` means nothing was recorded, which is not the same as a turn
        that recorded neither half -- the distinction absence is never
        collapsed into.
        """
        return read_turn_text(Path(self.trajectory_root) / turn_id)

    def read(self, turn_id: str) -> Any:
        """The tool calls captured for one turn, or ``None`` if none were.

        ``None`` and ``[]`` mean different things here for the same reason they
        do everywhere else: no file means the turn reported nothing, and must
        not be rendered as an assertion that zero tool calls occurred.
        """
        return Trajectory(Path(self.trajectory_root) / turn_id).read()

    def prune(
        self,
        *,
        keep_last: int | None = None,
        older_than_s: float | None = None,
    ) -> int:
        """Delete recorded turns from ``trajectory_root``. Returns how many.

        The trajectory is append-only and nothing else removes it, so without
        this a long-running production process accumulates one directory per
        RECORDING turn forever -- roughly 2.4 GB and a million inodes per
        million turns at typical record sizes, and far more if tool outputs are
        large, because ``output`` is stored verbatim.

        This is a method rather than a default because the trajectory is the
        customer's data, sitting in a directory the customer named, and a
        library that silently deletes it would be worse than one that grows.
        Pass ``retention_turns=`` to :func:`attach` to have it called for you.

        Args:
            keep_last: keep this many most-recent turns, delete the rest.
            older_than_s: delete turns last modified longer ago than this.

        Both may be given; a turn is deleted if EITHER says so. Passing neither
        deletes nothing and returns 0 -- an unqualified "prune" would be a
        request to delete everything, and that is not a thing to infer.

        Never raises. A turn that cannot be read or removed is left alone and
        does not stop the others; retention failing must not take down the
        process it was protecting.
        """
        if keep_last is None and older_than_s is None:
            return 0

        root = Path(self.trajectory_root)
        try:
            entries = [e for e in root.iterdir() if e.is_dir()]
        except OSError:
            return 0

        def mtime(e: Path) -> float:
            try:
                return e.stat().st_mtime
            except OSError:
                return 0.0

        # Newest first, so "keep_last" is a slice rather than a sort per turn.
        entries.sort(key=mtime, reverse=True)
        doomed: list[Path] = []
        if keep_last is not None:
            doomed.extend(entries[max(0, keep_last):])
        if older_than_s is not None:
            cutoff = time.time() - older_than_s
            doomed.extend(e for e in entries[:max(0, keep_last or 0)]
                          if mtime(e) < cutoff)

        removed = 0
        for e in dict.fromkeys(doomed):
            # Release any handle we still hold on it first, or on a platform
            # that refuses to unlink an open file this would silently no-op.
            _release(str(e))
            try:
                shutil.rmtree(e)
                removed += 1
            except OSError:
                continue
        if removed:
            # The upload index describes this tree, so it is bounded by this
            # tree. Left alone it would grow for the life of the process while
            # the directory it describes stays capped -- which is the opposite
            # of what a retention setting is for.
            self._forget_uploaded({
                e.name for e in entries if e not in set(doomed)
            })
        return removed

    #: Where upload state is kept: ONE FILE at the trajectory root, holding the
    #: ids already accepted by the control plane, newline-delimited.
    #:
    #: NOT A MARKER INSIDE THE TURN DIRECTORY, which is where this started and
    #: which was a real defect. Creating a file inside a directory bumps that
    #: DIRECTORY's mtime, and mtime is the ordering key for both readers of
    #: this tree: ``export.iter_records`` sorts by it to keep a conversation's
    #: turns in the order they happened, and ``prune(keep_last=N)`` uses it to
    #: decide which turns are newest. Marking a turn uploaded would therefore
    #: have floated it to the end of its own conversation and made retention
    #: delete the wrong turns -- a capture bug caused entirely by the
    #: bookkeeping for the upload.
    #:
    #: A FILE and not a directory, because neither reader filters dotfiles:
    #: they take everything under the root that ``is_dir()``, so a
    #: ``.uploaded/`` directory would have been scanned as a turn.
    #:
    #: Per TURN, which is safe here and would not have been for a
    #: conversation-shaped upload: each turn is its own row keyed on its own
    #: id, so sending turn 4 later does not re-send or truncate turns 1-3.
    _UPLOADED_INDEX = ".uploaded"

    def _uploaded_ids(self) -> set[str]:
        """Turn ids the control plane has already accepted."""
        try:
            raw = (Path(self.trajectory_root) / self._UPLOADED_INDEX).read_text(
                encoding="utf-8")
        except OSError:
            return set()
        return {line.strip() for line in raw.splitlines() if line.strip()}

    def _mark_uploaded(self, turn_ids: list[str]) -> None:
        """Append accepted ids. Append-only, so a crash mid-write can lose the
        record of an upload but never invent one -- and a lost record costs a
        re-send the server dedupes, never a duplicate row."""
        if not turn_ids:
            return
        try:
            with (Path(self.trajectory_root) / self._UPLOADED_INDEX).open(
                    "a", encoding="utf-8") as fh:
                fh.write("".join(t + "\n" for t in turn_ids))
        except OSError:
            # The turns landed; we just cannot remember that. A re-send is
            # deduped server-side, so this costs bandwidth, never correctness.
            pass

    def _forget_uploaded(self, keep: set[str]) -> None:
        """Drop ids whose turn directory no longer exists.

        Called from ``prune``. Without it the index grows for the life of the
        process while the tree it describes stays bounded -- an unbounded file
        is not an acceptable price for retention that exists to bound things.
        """
        try:
            path = Path(self.trajectory_root) / self._UPLOADED_INDEX
            current = self._uploaded_ids() & keep
            path.write_text("".join(t + "\n" for t in sorted(current)),
                            encoding="utf-8")
        except OSError:
            pass

    def _turn_wire(self, d: Path) -> dict[str, Any] | None:
        """One captured turn in the intake's wire shape, or None if empty.

        TURN-NATIVE, not corpus-shaped. What crosses the boundary is what the
        agent observed -- the user's message, the answer, the calls -- and the
        projection into any training-row format happens on the far side, where
        it can change without a customer redeploying.
        """
        text = read_turn_text(d) or {}
        traj = Trajectory(d)
        calls = traj.read()
        answer = text.get("answer")
        if not text.get("task") and not answer and not calls:
            return None
        # THE TRI-STATE, asked of the trajectory FILE rather than inferred.
        # ``[] if self.adapters_bound else None`` answered a different
        # question: whether an adapter bound at ATTACH time, which says
        # nothing about whether THIS turn recorded anything. A turn whose
        # trajectory was never written would have been uploaded as a positive
        # assertion that the agent made zero calls -- an observation nobody
        # made, and the one transformation this SDK exists to refuse.
        #
        # File present, no rows  -> ``[]``  (it ran and called nothing)
        # File absent            -> ``None`` (nothing was recorded for it)
        if calls:
            tool_calls: list[dict[str, Any]] | None = [c.to_wire() for c in calls]
        elif traj.path.exists():
            tool_calls = []
        else:
            tool_calls = None
        wire: dict[str, Any] = {
            "turn_id": d.name,
            "input_text": text.get("task"),
            "final_text": answer,
            "tool_calls": tool_calls,
            "tool_calls_omitted_count": traj.malformed,
        }
        cid = read_conversation_id(d)
        if cid:
            wire["conversation_id"] = cid
            wire["turn_index"] = self._turn_index(d, cid)
        return wire

    def _turn_index(self, turn_dir: Path, conversation_id: str) -> int:
        """This turn's position within its conversation.

        DERIVED FROM THE SAME ORDERING ``export.iter_records`` USES -- the
        directory mtime -- rather than from a counter. A counter held in the
        Attachment would restart at zero on every process restart and on every
        flush, and the control plane indexes on
        ``(org, agent, conversation_id, turn_index)``: every turn landing at 0
        makes a conversation unorderable, which is the one thing the index
        exists to prevent.

        Computed over every turn of the conversation ON DISK, not just the
        ones being uploaded now, so a turn sent in a later flush keeps the
        position it actually had. It shifts only if an earlier turn is pruned
        away, which is the honest answer: the earlier turn is gone.
        """
        root = Path(self.trajectory_root)
        siblings: list[tuple[float, str]] = []
        try:
            entries = list(root.iterdir())
        except OSError:
            return 0
        for d in entries:
            if not d.is_dir():
                continue
            if read_conversation_id(d) != conversation_id:
                continue
            try:
                siblings.append((d.stat().st_mtime, d.name))
            except OSError:
                continue
        siblings.sort()
        for i, (_, name) in enumerate(siblings):
            if name == turn_dir.name:
                return i
        return 0

    def flush(self, *, include_unanswered: bool = False,
              batch_size: int = 64) -> dict[str, int]:
        """Upload captured turns to the control plane. Returns counts.

        WHY NOT AT TURN-SCOPE EXIT, which is where an earlier design put it:
        ``record_answer`` is callable AFTER the scope closes, and the package's
        own example records the answer after the ``with`` block. Uploading on
        exit therefore shipped a turn whose ``final_text`` did not exist yet --
        every turn, for the documented usage.

        So the default holds back a turn that has no answer, because the answer
        may still be coming. ``close()`` passes ``include_unanswered=True``:
        at shutdown nothing more is coming, and a turn that made real tool
        calls and never recorded an answer is honest data rather than a
        validation error -- the intake stores it, flags it, and excludes it
        from objectives that need a completion.

        Never raises. A control plane that is unreachable must not take down a
        process that is serving users; the turns stay on disk and the next
        flush retries them.
        """
        sent = skipped = failed = 0
        if not self.may_upload or not self._cp_url or not self._api_key:
            return {"sent": 0, "skipped": 0, "failed": 0, "not_permitted": 1}

        root = Path(self.trajectory_root)
        if not root.is_dir():
            return {"sent": 0, "skipped": 0, "failed": 0}

        already = self._uploaded_ids()
        pending: list[tuple[Path, dict[str, Any]]] = []
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name)
        except OSError:
            return {"sent": 0, "skipped": 0, "failed": 0}
        for d in entries:
            if not d.is_dir() or d.name in already:
                continue
            # PER DIRECTORY. The scan used to sit outside the try below, so a
            # truncated trajectory -- which this package documents as the
            # EXPECTED case, not an exotic one, because a deadline kill cuts
            # the final line -- raised straight out of a method whose docstring
            # says it never raises, inside a customer's shutdown path. One
            # unreadable turn now costs that turn.
            try:
                wire = self._turn_wire(d)
            except Exception:
                failed += 1
                continue
            if wire is None:
                continue
            if wire.get("final_text") is None and not include_unanswered:
                skipped += 1
                continue
            pending.append((d, wire))

        if not pending:
            return {"sent": 0, "skipped": skipped, "failed": 0}

        client = None
        try:
            from ._http import ControlPlaneClient
            from .transport import AttachTransport

            client = ControlPlaneClient(self._cp_url, self._api_key)
            tr = AttachTransport(client, self.agent_id)
            for i in range(0, len(pending), max(1, batch_size)):
                chunk = pending[i:i + max(1, batch_size)]
                try:
                    tr.report_turns(self.agent_id, [w for _, w in chunk])
                except Exception:
                    # Whole chunk unacknowledged. Leave every marker unwritten
                    # so the next flush retries it: the intake dedupes on
                    # turn_id, so a retry of a batch that DID land is answered
                    # as a duplicate rather than stored twice.
                    failed += len(chunk)
                    continue
                self._mark_uploaded([d.name for d, _ in chunk])
                sent += len(chunk)
        except Exception:
            failed += len(pending) - sent
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        return {"sent": sent, "skipped": skipped, "failed": failed}

    def close(self) -> None:
        """Unbind the adapters. Idempotent, and never raises.

        THE UPLOAD HAPPENS OUTSIDE THE LOCK, deliberately. ``turn()`` takes the
        same lock to advance its prune counter, so flushing while holding it
        would park a live user request behind a network call for as long as the
        control plane took to answer -- in a threaded server, on the shutdown
        path, which is exactly when latency is least affordable. The lock is
        held only long enough to claim the close and take the unbind list.

        Claiming ``_closed`` FIRST is what keeps this idempotent across
        threads: a second caller returns immediately rather than flushing
        again. And a double flush would be harmless anyway -- the uploaded
        markers make it find nothing -- so the ordering costs nothing to get
        right.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            unbinds, self._unbinds = self._unbinds, []

        # LAST CHANCE, and the only place unanswered turns are sent: nothing
        # more is coming after this. Before unbinding, so a turn completed
        # during shutdown is still on disk to find.
        try:
            self.flush(include_unanswered=True)
        except Exception:
            # Shutdown must not raise. The turns remain on disk.
            pass
        for u in unbinds:
            try:
                u()
            except Exception:
                # Unbinding is cleanup. A framework that changed shape under
                # us must not take the customer's shutdown with it.
                pass

    def __enter__(self) -> Attachment:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def attach(
    *,
    agent_id: str,
    trajectory_root: str,
    control_plane_url: str | None = None,
    api_key: str | None = None,
    adapters: list[str] | str | None = "auto",
    register: bool = True,
    retention_turns: int | None = None,
) -> Attachment:
    """Bind capture in THIS process, for an agent already serving users.

    One call at startup::

        import percepteye_agent_flywheel as fw

        px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

        with px.turn() as turn_id:
            reply = agent.invoke(user_text)      # your agent, unmodified

    ``trajectory_root`` is REQUIRED and that is deliberate. Turns accumulate
    there as directories and nothing removes them, because until the contract
    carries a production turn there is nowhere to ship them and deleting a
    customer's only copy would be the worse failure. Naming the directory is
    how retention becomes a decision somebody made rather than a surprise.

    ``retention_turns=N`` keeps only the N most recently written turns, pruning
    on an interval so it costs one directory scan per 256 turns rather than one
    per turn. Unset, NOTHING is deleted and the trajectory grows without bound:
    roughly 2.4 GB and a million inodes per million recording turns, and more
    if tool outputs are large. That default is deliberate -- the trajectory is
    your data in a directory you named, and a library that silently deleted it
    would be worse than one that grows -- but it is a decision, so make it.

    ``register=False`` skips the control-plane call entirely, for a process
    that should make no outbound connection at all.

    Nothing here raises on a control-plane failure. ``agent_id`` missing or
    ``trajectory_root`` missing raise -- those are configuration mistakes in
    front of the person making them -- but an unreachable control plane yields
    an attachment with ``policy == {}``, because an agent must keep serving its
    users whatever we happen to think of our own availability.
    """
    if not agent_id or not isinstance(agent_id, str):
        raise ConfigurationError("attach() requires a non-empty agent_id")
    if not trajectory_root or not isinstance(trajectory_root, str):
        raise ConfigurationError(
            "attach() requires trajectory_root -- the directory captured turns "
            "accumulate in. It is not defaulted because nothing deletes them "
            "yet, and where a customer's data lands is not ours to assume."
        )
    try:
        os.makedirs(trajectory_root, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            f"attach() cannot use trajectory_root={trajectory_root!r}: {exc}"
        ) from exc

    # ── adapters, bound with NO endpoint ────────────────────────────────────
    # The empty strings are load-bearing, not placeholders. bind_selected
    # forwards them to each adapter's bind(), and both built-ins treat "no
    # endpoint" as "do not touch routing": LangChainAdapter._route returns at
    # once when self.llm is None, and the ADK adapter never reads the URL. So
    # this installs observation and cannot re-point a single production call.
    unbinds: list[Any] = []
    bound: tuple[str, ...] = ()
    try:
        from .binding import bind_selected

        report, unbinds = bind_selected(adapters, llm_url="", api_key="")
        bound = tuple(report.bound or ())
    except Exception:
        # A framework that cannot be bound is a smaller loss than a process
        # that will not start. serve() raises here instead, and correctly: a
        # rollout with no capture is a wasted rollout, while a production turn
        # with no capture is still a served user.
        unbinds, bound = [], ()

    # ── the control plane, entirely optional ────────────────────────────────
    policy: dict[str, Any] = {}
    register_error: Exception | None = None
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL)
    key = api_key or os.environ.get(ENV_API_KEY)
    may_upload = False
    upload_refusal: str | None = None
    if register and url and key:
        client: ControlPlaneClient | None = None
        try:
            from . import __version__
            from ._http import ControlPlaneClient
            from .contract import CONTRACT_VERSION
            from .transport import AttachTransport

            client = ControlPlaneClient(url, key)
            tr = AttachTransport(client, agent_id)
            try:
                reg = tr.register({
                    "agent_id": agent_id,
                    # This process serves users; it does not contribute capacity.
                    # `serve()` sends "rollout". Sending the same value from
                    # here made the field carry zero bits -- intake could not
                    # tell a training donor from a production reader, which is
                    # the one question a `mode` on a registration exists to
                    # answer.
                    "mode": "production",
                    "sdk": f"percepteye-agent-flywheel-python/{__version__}",
                    "contract_version": CONTRACT_VERSION,
                    "reports_tool_calls": bool(bound),
                    "isolation": "inprocess",
                    "concurrency": 1,
                    "input_shape": "text",
                    "entrypoint": "attach",
                })
                # WHAT THE SERVER SAID, not what we assume. Capture is opt-in
                # per agent, so the default is "no" and the only way to learn
                # otherwise is to be told -- finding out by uploading would
                # mean sending the data whose permission is in question.
                cap = (reg or {}).get("capture") or {}
                may_upload = bool(cap.get("turns"))
                upload_refusal = cap.get("reason")
            except Exception as exc:
                # A REFUSED registration must not cost the last mile.
                #
                # `policy_current` is a bare GET scoped by API key -- it does
                # not put agent_id on the wire and does not require a
                # registration to have landed -- so there is no reason a
                # rejected register should also cost this process its trained
                # model. It used to: one `except` covered both calls, so
                # anything intake disliked about the payload silently demoted
                # production to the untrained policy.
                register_error = exc
            policy = tr.policy_current() or {}
        except Exception:
            policy = {}
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    # Warned LAST, outside the try that owns the network calls, so raising
    # here cannot cost the policy read -- and guarded, because under
    # `-W error` a warning IS an exception and this function must never fail a
    # customer's startup. Under that setting the warning is lost; the policy,
    # which is the part that matters, is not.
    #
    # Only when the policy read SUCCEEDED, which is what distinguishes the two
    # causes without parsing an error string: the network demonstrably works,
    # so intake refused this payload specifically. A register that failed
    # because the control plane was unreachable is left silent, exactly as it
    # was before -- an offline laptop should not print a warning about intake.
    if register_error is not None and policy:
        try:
            warnings.warn(
                f"[percepteye] the control plane refused this registration for "
                f"agent_id={agent_id!r}: {type(register_error).__name__}: "
                f"{register_error}. The policy read succeeded, so capture and "
                f"the trained policy are both unaffected. The likely cause is "
                f"an intake that has not been updated for this SDK version. "
                f"Capture continues locally either way; turns are uploaded "
                f"only when the control plane says this agent is enabled for "
                f"it.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:  # -W error turns the line above into a raise
            pass

    return Attachment(
        agent_id,
        trajectory_root=trajectory_root,
        policy=policy,
        unbinds=unbinds,
        adapters_bound=bound,
        retention_turns=retention_turns,
        control_plane_url=url,
        api_key=key,
        may_upload=may_upload,
        upload_refusal=upload_refusal,
    )
