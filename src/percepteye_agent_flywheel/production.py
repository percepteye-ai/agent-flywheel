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

**What this does NOT do yet.** It does not ship anything. The ``flywheel/1``
contract has no route for a production turn -- ``/health``, ``/agents``,
``/policy/current`` and ``/rollouts/*`` are the whole surface -- so turns
accumulate as trajectory directories under a root you name, and you own them
until that route exists. Writing a client for a route the server does not serve
is how an earlier SDK ended up with five modes of which four were strings
nobody read; this module will not repeat it.
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

        with _RolloutScope(tid, str(path), kind="turn"):
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
        return removed

    def close(self) -> None:
        """Unbind the adapters. Idempotent, and never raises."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for u in self._unbinds:
                try:
                    u()
                except Exception:
                    # Unbinding is cleanup. A framework that changed shape
                    # under us must not take the customer's shutdown with it.
                    pass
            self._unbinds = []

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
                tr.register({
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
                f"a control plane that does not yet accept mode='production' "
                f"from attach(); it is safe to ignore until intake is updated.",
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
    )
