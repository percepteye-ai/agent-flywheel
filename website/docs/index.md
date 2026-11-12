---
hide:
  - navigation
  - toc
---

<div class="fw-hero" markdown>

<span class="fw-hero__eyebrow">continual learning for agents already in production</span>

# Your agent, made better by its own work

<p class="fw-hero__sub">
<code>agent-flywheel</code> records your agent's real conversations. We learn from them,
test each new version against your agent's current best on conversations it has never
seen, and you approve the winner. Your agent runs in your process, behind your firewall,
on your credentials — unchanged.
</p>

<div class="fw-cta" markdown>
[Quickstart](quickstart.md){ .fw-btn .fw-btn--primary }
[Production](production.md){ .fw-btn .fw-btn--ghost }
</div>

<p class="fw-note">pip install agent-flywheel · one runtime dependency · MIT</p>

</div>

<div class="fw-stats">
<div class="fw-stat"><div class="fw-stat__n">0</div><div class="fw-stat__l">lines changed inside your agent</div></div>
<div class="fw-stat"><div class="fw-stat__n">1</div><div class="fw-stat__l">runtime dependency (<code>httpx</code>)</div></div>
<div class="fw-stat"><div class="fw-stat__n">0</div><div class="fw-stat__l">inbound ports, containers, GPUs</div></div>
<div class="fw-stat"><div class="fw-stat__n">1530</div><div class="fw-stat__l">tests, zero skips permitted</div></div>
</div>

## Start here

<div class="fw-grid">

<a class="fw-card" href="quickstart/">
  <div class="fw-card__art fw-art--rose"><div class="fw-card__tile">🚀</div></div>
  <div class="fw-card__body">
    <h3>Quickstart</h3>
    <p>Record your agent's conversations, see your first trace arrive, and start improving.</p>
    <div class="fw-card__more">Get started</div>
  </div>
</a>

<a class="fw-card" href="how-it-works/">
  <div class="fw-card__art fw-art--sky"><div class="fw-card__tile">🎡</div></div>
  <div class="fw-card__body">
    <h3>Practice tasks</h3>
    <p>Optional: tasks written from your agent's own tools, worked through and graded, every lane opened from your side.</p>
    <div class="fw-card__more">See how they run</div>
  </div>
</a>

<a class="fw-card" href="outcomes/">
  <div class="fw-card__art fw-art--amber"><div class="fw-card__tile">🎯</div></div>
  <div class="fw-card__body">
    <h3>Outcomes</h3>
    <p>Every tool call is <code>ok</code>, <code>failed</code> or <code>unknown</code> — and <code>unknown</code> is never rounded up.</p>
    <div class="fw-card__more">Why this matters</div>
  </div>
</a>

<a class="fw-card" href="python/">
  <div class="fw-card__art fw-art--mint"><div class="fw-card__tile">🐍</div></div>
  <div class="fw-card__body">
    <h3>Python agents</h3>
    <p>LangChain, LangGraph and Google ADK report per-call outcomes with no Agent Flywheel code in your agent.</p>
    <div class="fw-card__more">Bind an adapter</div>
  </div>
</a>

<a class="fw-card" href="node/">
  <div class="fw-card__art fw-art--lilac"><div class="fw-card__tile">📦</div></div>
  <div class="fw-card__body">
    <h3>Node harnesses</h3>
    <p>One npm package covering OpenClaw and DeepSeek Harness. Zero dependencies, and inert until you switch it on.</p>
    <div class="fw-card__more">Install the plugin</div>
  </div>
</a>

<a class="fw-card" href="cli/">
  <div class="fw-card__art fw-art--sage"><div class="fw-card__tile">⌨️</div></div>
  <div class="fw-card__body">
    <h3>CLI and other languages</h3>
    <p>Not Python? Pass an argv list. There is no protocol to implement, so an existing CLI needs no changes.</p>
    <div class="fw-card__more">Serve a binary</div>
  </div>
</a>

</div>

## The loop, in one figure

![The improvement loop as a ring across a trust boundary: your agent's conversations, recorded where it runs, go out once you allow it; a new version is learned from them and tested against your agent's current best on conversations it has never seen; you approve the winner in Decisions; and your agent runs it, and the next round starts from it.](assets/flywheel-architecture.svg)

**Register once, then loop forever.** The SDK describes your agent from inside its own
process and registers it — once, before the first rollout. After that: it claims a task,
your agent runs it, the outcomes go back, a policy is trained and promoted, and your
production agent picks it up. Once an operator enables capture for it, its live
turns feed the next lap too.

That last step is what makes it a flywheel rather than a pipeline: **each round starts
from the best so far.**

## What it costs to integrate

| | |
|:--|:--|
| Lines changed **inside** your agent | **0** — adapters bind from the outside |
| Lines of integration code | **1 import + 1 call** |
| Runtime dependencies | **1** (`httpx`), enforced by a test and a clean-install CI job |
| Inbound ports · public DNS · TLS | **0 · 0 · 0** — there is no listening socket in this package |
| Containers to build · GPUs to provision | **0 · 0** — your agent runs as your process |
| Languages | Python by import path, anything else by argv |
| License | [MIT](https://github.com/percepteye-ai/agent-flywheel/blob/main/LICENSE) |

Every row is checkable from [the repository](https://github.com/percepteye-ai/agent-flywheel).

!!! tip "Not ready to contribute capacity?"

    Set `PERCEPTEYE_AGENT_MODE=production` and `serve()` returns immediately without
    opening a connection — so the same deployed image either contributes or does not,
    with no code change. See [Configuration](configuration.md).
