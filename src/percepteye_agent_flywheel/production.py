"""``attach()`` -- capture in the process that serves your users.

:func:`~percepteye_agent_flywheel.serve` donates CAPACITY: it claims tasks from
the control plane and runs your agent against them in a child process. This
module donates EVIDENCE instead, from the turns your agent is already serving.
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
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._http import ControlPlaneClient
from .errors import ConfigurationError
from .outcomes import Trajectory, _RolloutScope

__all__ = ["Attachment", "attach"]

ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"
ENV_API_KEY = "PERCEPTEYE_API_KEY"


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
        self._lock = threading.Lock()

    @contextmanager
    def turn(self, *, turn_id: str | None = None,
             conversation_id: str | None = None) -> Iterator[str]:
        """Scope one user turn. Tool calls inside it land in its own trajectory.

        Yields the turn id, which the caller should keep: it is the join key
        for any outcome that arrives later than the turn does -- the draft was
        kept an hour afterwards, the pull request merged the next morning.

        A turn boundary is DECLARED rather than inferred. Guessing where one
        request ends and the next begins is the same class of guess this SDK
        refuses everywhere else, and getting it wrong silently merges two
        users' tool calls into one record.

        Nothing in here can fail the turn. If the directory cannot be created,
        capture is skipped for that turn and the body still runs.
        """
        tid = turn_id or f"t_{uuid.uuid4().hex[:16]}"
        path = Path(self.trajectory_root) / tid
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # No destination, so nothing to scope. The turn is the customer's
            # user request; it runs regardless of whether we can watch it.
            yield tid
            return
        if conversation_id:
            try:
                (path / "conversation_id").write_text(
                    str(conversation_id), encoding="utf-8")
            except OSError:
                pass
        with _RolloutScope(tid, str(path)):
            yield tid

    def read(self, turn_id: str) -> Any:
        """The tool calls captured for one turn, or ``None`` if none were.

        ``None`` and ``[]`` mean different things here for the same reason they
        do everywhere else: no file means the turn reported nothing, and must
        not be rendered as an assertion that zero tool calls occurred.
        """
        return Trajectory(Path(self.trajectory_root) / turn_id).read()

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
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL)
    key = api_key or os.environ.get(ENV_API_KEY)
    if register and url and key:
        client: ControlPlaneClient | None = None
        try:
            from . import __version__
            from .contract import CONTRACT_VERSION
            from .transport import AttachTransport

            client = ControlPlaneClient(url, key)
            tr = AttachTransport(client, agent_id)
            tr.register({
                "agent_id": agent_id,
                "mode": "rollout",
                "sdk": f"percepteye-agent-flywheel-python/{__version__}",
                "contract_version": CONTRACT_VERSION,
                "reports_tool_calls": bool(bound),
                "isolation": "inprocess",
                "concurrency": 1,
                "input_shape": "text",
                "entrypoint": "attach",
            })
            policy = tr.policy_current() or {}
        except Exception:
            policy = {}
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    return Attachment(
        agent_id,
        trajectory_root=trajectory_root,
        policy=policy,
        unbinds=unbinds,
        adapters_bound=bound,
    )
