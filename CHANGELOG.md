# Changelog

What changed in `agent-flywheel`, by release. The Node SDK,
[`@percepteye/agent-flywheel`](https://www.npmjs.com/package/@percepteye/agent-flywheel),
ships the same version number in the same release. Commit hashes are this
repository's.

## 0.1.3

### Added

- **The control-plane URL has a default**, as it already had in the Node SDK:
  the `control_plane_url=` given in code, then `PERCEPTEYE_CONTROL_PLANE_URL`,
  then `https://launch.percepteye.ai/api/flywheel/v1`, each trimmed and a
  blank one falling through, through one resolver
  (`agent_flywheel.control_plane.resolve_control_plane_url()`, beside
  `DEFAULT_CONTROL_PLANE_URL`) that every reader uses, the goose hook and the flywheel key's home
  included. `serve()` no longer refuses a missing URL, and the key is the one
  thing that can be missing. `agent-flywheel install-hooks` and `status` name
  the URL they resolved to. (from 5e91987)
- **What the model saw, under a consent tier of its own.** With an operator's
  model-content grant for the agent, the LangChain and Google ADK adapters
  observe each model call inside an `attach()` turn — system instructions, the
  whole input, the tool definitions offered, the answer — and every tool's
  result, and the turn uploads it as one OpenTelemetry GenAI trace
  (`genai_trace`). Off by default and observe-only; without the grant nothing
  of it is recorded, not even to disk, and recording stops when the control
  plane refuses a trace for a turn it can attribute to no Product. Both SDKs
  read a refused turn batch the same way. (ba0df5a, c5d6fae, fa8318b; merged
  in aded8af)
- **`rules=` on `serve()` and `attach()`** declares the Product's workflow
  rules once, after the registration. `None` sends nothing and `[]` retracts. A
  refusal costs only the rules, a slow rulebook route never holds back startup,
  and a failure says "not registered" only when that is known. (ffeb5f6,
  1fa1dd3, f1e97e3)
- **A withdrawn model is given back.** When the control plane moves an agent
  back to a base arm with nothing to serve — no champion, no endpoint of the
  customer's own — it answers `withdrawn: true`, and `policy_source()` reverts
  to the snapshot it was constructed with, the agent's own configuration,
  reporting it once as `changed`. It used to read that answer's null endpoint
  as "keep what you run" and stay on the candidate a person had rolled back. A
  null endpoint without the flag still keeps the last configuration.
  `is_withdrawn()` is exported, and `endpoint_kwargs()` refuses a withdrawn
  answer before any endpoint beside it, as the Node SDK does. The snapshot
  says so in its own field, `PolicySnapshot.withdrawn`, and a withdrawal is
  logged once as a warning naming the control plane's reason, by both
  `policy_source()` and `apply_current_policy()`. **Upgrading:** `on_change`
  now also fires on a withdrawal, with that empty snapshot, so a callback (or
  a turn loop rebuilding on `changed`) that builds a client from the snapshot
  must check `snap.withdrawn` and go back to its own configuration instead of
  building one from empty fields.

### Changed

- **The goose hook claims practice tasks only in training mode.** It claims
  a task on `SessionStart`, and reports the session as its result on `Stop`,
  only when `PERCEPTEYE_AGENT_MODE` resolves to training and the key is set;
  a production install with a key used to lease a rollout into the
  customer's session and POST that session as its result. An unreadable mode
  claims nothing. `install-hooks` and `status` say which applies, through the
  same predicate (`install_hooks.claim_refusal`). (8c9b151)
- **`attach(retention_turns=...)` refuses a value that is not a whole number
  0 or more**, before capture starts, as the Node SDK drops one. (from
  52bec17)
- **A Prime harness snapshot requires its kernel venv** and is refused without
  one, naming why: without it Prime bootstrapped a runtime into the rollout's
  private home at every session start, and the release of that home failed.
  (6379c52)
- **Your flywheel key opens only the control plane's own gateway.** A
  `pefw_` key now goes to an endpoint only when the control plane names
  `PERCEPTEYE_API_KEY` for it and it sits at the origin of your control-plane
  URL (resolved as everywhere else, the default included), as the Node SDK
  already required. It used to be the
  last fallback of the credential order, so it was sent to a provider endpoint
  the control plane named no variable for, to one whose named variable was
  unset, and to a self-hosted box's raw address; and the `api_key=` passed to
  `apply_current_policy()` or `policy_source()` — the key to the control
  plane — beat a provider's own key as an endpoint override. Each of those is
  now not applied, with a warning naming the cause. `endpoint_kwargs()` takes
  `flywheel_key=` and `control_plane_url=` (defaulting to `PERCEPTEYE_API_KEY`
  and the resolved control-plane URL);
  its `api_key=` stays your own credential for the endpoint. An endpoint with
  no `model` is no longer applied, as in Node. **Upgrading:** a production
  agent pointed at a self-hosted endpoint by a control plane without the
  gateway projection keeps its own configuration until that control plane
  serves the gateway URL, rather than sending its key to that box.
- **A `policy_source()` refresh that cannot reach the control plane is a
  failed refresh**: `last_refresh_ok` goes `False` and `stale_for_s` grows
  while the last good snapshot stands. It used to be absorbed as a successful
  answer of nothing deployed, so an outage reported a fresh, healthy policy.
- **A bounded tool or error body keeps both ends** behind a marker that counts
  what it replaced, in every producer — the host recorder, both adapters and
  the parser — and an error body gets the same 4000 characters as output
  instead of 1000. (7e5499a, 202691b, fb4c6d3)
- **Artifacts travel only in the rollout report.** The SDK no longer POSTs each
  artifact to a route the control plane never served. (ef4e1cb, 7b80fec)
- **A rollout's outcome is never lost**: every terminal path reports, and the
  model call that could eat a whole episode is bounded by a ceiling derived
  from the episode budget. (dcd0a6f)
- **A per-process socket no longer changes the agent's execution identity**, so
  restarting an agent no longer invalidates a prepared run. (5ea44cd)
- **The Prime adapter seals its kernel against the serving shell**: it hands
  the sealed kernel to prime-agent as `PRIME_AGENT_KERNEL_PYTHON`; removes
  `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `IPYTHONDIR`,
  `PYTHONEXECUTABLE`, `__PYVENV_LAUNCHER__` and `PYTHONPYCACHEPREFIX` from
  every environment it configures; refuses a kernel venv that cannot import
  `ipykernel` and `rlm` when the snapshot is built; keeps its daemon socket
  within the length a Unix socket can bind; and excludes a rollout whose tool
  call came back as Prime's refusal to start the kernel as
  `substrate_unavailable`. (69bce71, 40a5d6c, f411916, 5689386, f4d98c2,
  7eff63e, f63842d, ad568dc)

## 0.1.2, 0.1.1, 0.1.0

Released before this file existed; see the `v0.1.2`, `v0.1.1` and `v0.1.0`
tags.
