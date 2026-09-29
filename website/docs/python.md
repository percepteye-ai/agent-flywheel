# Python agents

A Python agent is served by handing `serve()` an import path. Everything on this page is about what happens on the far side of that string.

```python
import agent_flywheel as fw

fw.serve("my_package.agent:build", agent_id="research-agent", input_shape="text")
```

Adapters are on by default, so a LangChain, LangGraph or Google ADK agent reports per-call outcomes with **no Agent Flywheel code inside it**. Not Python? Pass an argv list instead — see [CLI and other languages](cli.md).

## The entrypoint is an import path, not an object

The default isolation spawns one child process per rollout, and a child cannot receive a live Python object. So the entrypoint is a string the child can import. The parent resolves it at startup and the child re-resolves the same string per rollout, through one shared function — when those two arrived at the agent by different code, the child could run something the parent never validated.

What `attr` may name:

| `"module:attr"` resolves to | what happens |
|:--|:--|
| a **class** | instantiated with no arguments. One that needs arguments raises `ConfigurationError` naming the missing one |
| a **zero-argument factory** | called; its return value is the agent |
| an **already-built object** with `invoke` / `run` / `execute` / `chat` | used as-is, never called |
| a **function taking the task** | called as a factory, so it raises `TypeError` — wrap it in a class or a factory |

That last row is the sharp edge. `"my_package.agent:answer"`, where `answer(task)` is the agent itself, is indistinguishable from a factory at import time and is treated as one.

A constructed object is accepted directly, but only under `isolation="inprocess"`, which is capped at `concurrency=1` — concurrent in-process rollouts share one set of environment variables and would silently cross-attribute each other's completions. Both constraints are checked at startup, not at rollout time.

## Which method gets called

Probed in order, first match wins:

| order | attribute |
|:--|:--|
| 1 | `invoke` |
| 2 | `run` |
| 3 | `execute` |
| 4 | `chat` |
| 5 | `__call__` |

Pass `method="handle"` to name one instead of probing. `async def` is detected and awaited; you write nothing for it.

**Resolution happens once, at startup, and raises there.** That is not tidiness. A rollout is one member of a comparison group, so a rollout that never emits because a method name was wrong does not merely lose itself — it biases the score of every sibling it was supposed to be compared against. A typo must stop the process in front of the person who made it. The error names what was actually available:

```text
no entrypoint found on Nothing. Tried, in order: invoke, run, execute, chat,
__call__. Available callables: helper.
Pass method='...' to name one explicitly, or pass a plain callable.
```

## `input_shape`

Declared, never inferred. `run(str)` and `run(dict)` are indistinguishable from the outside until something fails at depth, so the caller says which one it is.

| `input_shape` | your entrypoint receives |
|:--|:--|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "the task description"}]` |
| `"dict"` | `{"input": "the task description"}` |

Any other value raises `EntrypointError` at startup.

## Adapters

`adapters="auto"` is the default. It walks the built-in registry, probes each one, and binds those whose framework is actually installed. An absent framework is skipped and recorded, never an error — because asking every customer to name their framework is instrumentation, and the point of an adapter is that there is none.

| adapter | distribution | versions | covers |
|:--|:--|:--|:--|
| `langchain` | `langchain-core` | `>=0.3,<2` | LangChain **and** LangGraph |
| `adk` | `google-adk` | `>=1,<3` | Google ADK |

There is no separate LangGraph adapter. LangGraph tool calls run through langchain-core's own callback system, so binding `langchain` captures both; what LangGraph adds is the node name.

### What each one can honestly report

| framework | `ok` | `failed` | `unknown` | call attributed to |
|:--|:--|:--|:--|:--|
| **LangChain** | a `ToolMessage` with `status="success"` | the tool raised, or a `ToolMessage` with `status="error"` | every other return value | an `agent:`-prefixed run tag |
| **LangGraph** | same | same | same | the executing node, from `langgraph_node` run metadata |
| **Google ADK** | **never reported** | the tool raised, or it returned a dict with a truthy `"error"` key | every other return value | `ToolContext.agent_name`, natively |

`status_code` is always `null` from an adapter. Neither framework surfaces an HTTP status at the tool boundary, and recovering one from the payload is exactly how an unobserved call becomes a landed one.

**ADK has no `ok` path, and that is the design.** A returned value proves the tool did not raise; it does not prove the tool worked. ADK's compensation is the better half of the trade: `on_tool_error_callback` separates *raised* from *returned*, so a genuine `failed` is available where LangChain's `on_tool_end` fires for success and swallowed failure alike.

Adapters are best-effort acceleration over [`record_tool_call`](outcomes.md), never a substitute for it. Where you know the answer, say it.

### Adapters add no dependencies

No adapter imports its framework at module scope. Importing the SDK, and even constructing both built-in adapters, pulls in neither `langchain_core` nor `google.adk` — the framework import happens inside `bind()`, and the probe only reaches an import after package metadata has already confirmed the distribution is installed. The runtime dependency list stays at one (`httpx`), asserted by a clean-install CI job.

### `[]`, `"auto"`, and a name are three different instructions

| `adapters=` | behavior |
|:--|:--|
| `"auto"` (default) | bind every built-in whose framework is installed; skip and record the rest |
| `[]` | bind nothing. A real choice, and it must be silent rather than half-instrumented |
| `["langchain"]` | **require** it. A named adapter that cannot bind raises `RoutingNotInstallable` |
| `["nosuchname"]` | `KeyError`, listing the built-ins that do exist |

Naming an adapter is an assertion that it is how your calls get captured, so it fails closed. `"auto"` asserts nothing, so it degrades quietly:

```text
[AdapterFrameworkAbsent] adk: distribution 'google-adk' is not installed

  remedy: install google-adk, or remove the adapter from serve(adapters=[...])
```

`RoutingNotInstallable` carries a machine-readable `reason` — `AdapterFrameworkAbsent`, `VersionOutOfRange` or `SymbolMissing` — plus the evidence and a printed remedy. The version gate is soft: `packaging` is not a runtime dependency, so when it is unavailable the range is skipped and the symbol probe decides. That probe resolves exactly the tuple of `module:attr` paths the adapter will later touch, which is what makes detection and binding inseparable — detection cannot pass while binding is broken.

!!! warning "Adapters bind in the process that runs your agent, not at startup"

    `serve()` validates the entrypoint, credential and contract version before the first rollout, but it does **not** probe adapters there. Binding happens inside the rollout child, because a callback registered in the parent would never see a child's tool calls. So a named adapter that cannot bind surfaces per rollout: the child exits non-zero and the rollout is reported as the typed exclusion `flywheel_crash`, carrying the `RoutingNotInstallable` message in its detail. It is never scored as a bad decision by your agent. Under `attach()` the opposite rule applies — a binding failure is swallowed, because a production turn with no capture is still a served user.

## Zero instrumentation, worked end to end

This agent contains no reference to Agent Flywheel of any kind:

```python
# lc_agent.py — your existing agent, untouched
from langchain_core.tools import tool


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"found: {q}"


class Agent:
    tools = [lookup]

    def invoke(self, task: str) -> str:
        return str(lookup.invoke({"q": task}))


def build():
    return Agent()
```

Served with the three lines from the [Quickstart](quickstart.md):

```python
import agent_flywheel as fw

fw.serve("lc_agent:build", agent_id="research-agent", input_shape="text")
```

On a rollout whose task is `"count to three"`, the reported trajectory is:

```json
"tool_calls": [
  {
    "name": "lookup",
    "arguments": {"q": "count to three"},
    "outcome": "unknown",
    "status_code": null,
    "output": "found: count to three",
    "latency_ms": 0.226
  }
]
```

**`unknown`, not `ok`.** The tool returned a string and nothing observed whether the call succeeded. `Tool.invoke({"q": ...})` with a plain dict has no tool-call id to attach, so LangChain returns raw content rather than a `ToolMessage` — and raw content from a tool that worked is byte-identical to raw content from a tool that caught its own exception and returned prose. Calling that `ok` is the single most consequential lie a capture can tell, so the adapter does not.

That is the honest floor zero instrumentation buys you, and it is enforced by [`tests/test_zero_instrumentation.py`](https://github.com/percepteye-ai/agent-flywheel/blob/main/tests/test_zero_instrumentation.py) inside the 1234-test suite. Drive the same tool through a model-issued tool call and the same adapter reports `"outcome": "ok"` with the model's `tool_call_id` attached, because then there is a `ToolMessage` with a real `status`.

One call where you *do* know the answer gets you the rest:

```python
from agent_flywheel import record_tool_call

record_tool_call("charge_card", {"amount": 4200}, outcome="ok", status_code=201)
```

`outcome` and `status_code` are keyword-only and required, so half an outcome is unconstructible. Outside a rollout the call is a no-op, which is why the line can stay in a shared code path. See [Outcomes](outcomes.md).

??? note "Routing a chat model you constructed yourself"

    `LangChainAdapter(llm=...)` re-points a chat model at the per-rollout gateway and reads the value back to check it took. It exists because `ChatOpenAI(base_url=...)` in your own code beats every environment variable: the rollout completes, the completions go to a live third party, and nothing distinguishes it from a clean run. A client whose endpoint is fixed at construction raises `RoutingNotInstallable(reason="HardcodedBaseUrl")`.

    This is reachable only by passing an adapter **instance**, which means `isolation="inprocess"` — the subprocess payload is JSON, and an adapter object cannot cross it. `adapters=["langchain"]` constructs the adapter with `llm=None`, which touches routing not at all. The ordinary arrangement is the default one: the child's environment routes, because the agent is constructed inside a child that already has those variables set. Build your model inside the factory and it is handled for you.

## Capturing from production traffic

`attach()` takes the same `adapters=` argument with the same `"auto"` default, and binds them with **no endpoint at all** — both built-ins treat an empty URL as "do not touch routing". It captures; it cannot reroute a single production call. See [Production](production.md).

```python
px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

with px.turn(task_id=case_id, conversation_id=conv_id) as turn_id:
    reply = agent.invoke(user_text)       # your agent, unmodified
```

## Shipping your own adapter

The bar is the one the built-ins meet: no framework import at module scope, report `unknown` rather than guessing, and fail loudly at bind time rather than degrade silently. An adapter is a small object — `name`, `distribution`, `supported`, `required_symbols`, `capabilities()`, and a `bind()` that returns an unbind callable.

Be aware of where the wiring currently stops. `serve()` and `attach()` resolve adapter **names** against the built-in registry only; the `agent_flywheel.adapters` entry-point group is read by `binding.discover_adapters()`, and no public entry point calls it. So today a third-party adapter distribution is not picked up by `adapters="auto"`, and its name is not accepted in `adapters=[...]`. Reach it by passing the instance under `isolation="inprocess"`, or open an issue — a framework worth an adapter is worth wiring properly.

## Related

| page | what it covers |
|:--|:--|
| [Outcomes](outcomes.md) | `ok` · `failed` · `unknown`, and why `unknown` is never rounded up |
| [Configuration](configuration.md) | `PERCEPTEYE_AGENT_MODE`, isolation, concurrency, every environment variable |
| [Production](production.md) | `attach()`, `apply_current_policy()`, `policy_source()` |
| [The contract](contract.md) | the wire format the adapters write into |
