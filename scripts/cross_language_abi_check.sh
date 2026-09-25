#!/usr/bin/env bash
# The THREE contracts the Node SDK and the Python SDK share, checked across
# the language boundary by running both implementations.
#
# 1. THE CAPTURE CONTRACT. Whatever a plugin appends to
#    $PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl, `Trajectory` has to read back
#    with the same meaning. A JS suite that only checks itself would let the
#    halves drift -- a plugin staying green while writing a line the reader
#    counts as malformed, surfacing later as a silently short tool-call list.
#
# 2. THE EXECUTION DIGEST. Both SDKs hash the same descriptor under the same
#    two domain separators, and Mission Control binds the result into an
#    ed25519-signed training workload and refuses every episode whose
#    attestation does not match. Until this section existed the two agreed only
#    by convention: nothing executed both. A one-sided change to the canonical
#    encoding, or to the command-argv projection that drops declared transport
#    values, would make the SAME agent attest a DIFFERENT identity depending on
#    which SDK registered it -- worse than the per-restart instability the
#    projection was added to fix, because it is silent on both sides.
#
# 3. THE MODEL BOUNDARY. A production turn may carry what its model calls saw
#    and said as ONE OTel GenAI trace, and Mission Control stores it by the
#    digest of its canonical bytes and treats a DIFFERENT trace for the same
#    turn as a conflict. Both SDKs render one shared golden
#    (tests/fixtures/genai_capture_golden.json here, a byte-identical copy at
#    test/fixtures/genai-capture-golden.json there) and must write the same
#    bytes; and a turn the Node plugin records on disk must render, through
#    THIS package's reader, to exactly the trace the plugin uploads.
#
# THIS FILE EXISTS SO CI AND A DEVELOPER RUN THE SAME CHECK. It used to live
# inline in the workflow, and a local approximation of it passed while the real
# one failed: the JS half had grown two records and the assertion had not. One
# definition, invoked from both places -- and, since the execution digest was
# added, from tests/test_cross_language_abi_check.py as well, so the Python
# suite goes red on a drift in either language.
#
#   ./scripts/cross_language_abi_check.sh
#
# Requires: a WORKING node, and the package importable (pip install -e .).
#
# NODE IS RESOLVED, NOT ASSUMED. A bare `node` is not a working node: this
# machine's Homebrew node 25.4.0 aborts before running a line of script on a
# broken simdjson link, so the check died with a dyld error that says nothing
# about the ABI it exists to guard. tests/node_sdk.py already resolves the same
# way; two answers to "which node" is one too many.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# `--print-node-sdk` resolves the SDK, prints its root directory on stdout and
# stops without running the check.
#
# THE RESOLVER BELOW IS THE ONLY ONE. tests/test_report_turns_wire.py asks for
# the path through this flag instead of keeping a second copy of the rule. The
# copy it used to have looked two directories up for a fixed sibling name,
# which is nowhere near where CI puts the checkout, so the cross-SDK parity
# test skipped on every CI run while looking green on a laptop.
PRINT_ONLY=""
if [ "${1:-}" = "--print-node-sdk" ]; then
  PRINT_ONLY=1
  shift
fi

# WHERE THE NODE SDK IS, now that it is a separate repo
# (percepteye-ai/agent-flywheel-node). It used to be a subdirectory here, and
# this line was a relative path into it.
#
# Resolved rather than assumed, and the two cases are NOT the same case:
#
#   - AUTO-DISCOVERY skips. A developer who checked out only this repo has no
#     Node SDK to check against, and turning that into a red suite would teach
#     everyone to ignore the failure -- which costs more than the check.
#   - AN EXPLICIT PERCEPTEYE_NODE_SDK IS A PROMISE. Somebody said the SDK is
#     there. If it is not, that is a broken environment and this fails. CI
#     always sets the variable, so CI can no longer turn this check into a
#     green no-op by moving the checkout or by the Node repo renaming src/.
EXPLICIT=""
if [ -n "${PERCEPTEYE_NODE_SDK:-}" ]; then
  SDK="${PERCEPTEYE_NODE_SDK%/}"
  EXPLICIT=1
else
  for cand in \
    "$ROOT/../percepteye-agent-flywheel-node" \
    "$ROOT/../agent-flywheel-node"; do
    if [ -d "$cand/src" ]; then SDK="$cand"; break; fi
  done
fi

if [ -n "$EXPLICIT" ] && [ ! -d "$SDK/src" ]; then
  {
    echo "cross-language ABI check FAILED: PERCEPTEYE_NODE_SDK points at"
    echo "  $SDK"
    echo "  but there is no src/ directory there. The variable was set"
    echo "  deliberately, so this is a broken checkout -- not a reason to stop"
    echo "  checking the contract."
  } >&2
  exit 1
fi

if [ -z "${SDK:-}" ]; then
  # In --print-node-sdk mode stdout carries the resolved path and nothing
  # else, so the caller sees it empty and skips; the explanation goes to
  # stderr there rather than being mistaken for a path.
  if [ -n "$PRINT_ONLY" ]; then exec 1>&2; fi
  echo "cross-language ABI check SKIPPED: the Node SDK is not checked out."
  echo "  It lives at github.com/percepteye-ai/agent-flywheel-node."
  echo "  Clone it beside this repo, or set PERCEPTEYE_NODE_SDK=/path/to/it."
  exit 0
fi

PKG="$SDK/src"
if [ -n "$PRINT_ONLY" ]; then
  echo "$SDK"
  exit 0
fi
echo "cross-language ABI check: reading the Node SDK at $PKG"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# SAME RULE AS $NODE ABOVE. A bare `python` is not a Python: it does not exist
# on a modern macOS, and where it does it is not necessarily the interpreter
# with this package installed. Prefer the repo venv, which by definition has it.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for cand in "$ROOT/.venv/bin/python" python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c "import agent_flywheel" >/dev/null 2>&1; then
      PYTHON="$cand"; break
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "✗ no python with agent_flywheel importable" >&2
  echo "  (tried \$PYTHON, $ROOT/.venv/bin/python, python3, python)." >&2
  echo "  Run: pip install -e ." >&2
  exit 1
fi

NODE=""
for cand in ${PERCEPTEYE_NODE_BIN:-} /opt/homebrew/opt/node@24/bin/node node; do
  if [ -n "$cand" ] && command -v "$cand" >/dev/null 2>&1 \
     && "$cand" --version >/dev/null 2>&1; then
    NODE="$cand"; break
  fi
done
if [ -z "$NODE" ]; then
  echo "✗ no working node found (tried \$PERCEPTEYE_NODE_BIN, node@24, node)." >&2
  echo "  A node that cannot run --version cannot run the ABI check." >&2
  exit 1
fi

"$NODE" --input-type=module -e "
import { createRecorder } from '$PKG/record.js';
import { createWriter } from '$PKG/trajectory.js';
import { recordFor } from '$PKG/dsh.js';

// OpenClaw: its plugin correlates nothing -- one event, one record.
const rec = createRecorder({ writer: createWriter({ dir: '$OUT' }) });
rec.handle({ toolName:'read', params:{path:'/etc/hosts'}, toolCallId:'c1', result:'ok', durationMs:3 }, {agentId:'coder'});
rec.handle({ toolName:'exec', params:{command:'exit 3'}, toolCallId:'c2', result:{details:{status:'completed',exitCode:3}}, durationMs:91 });
rec.handle({ toolName:'exec', params:{command:'echo hi'}, toolCallId:'c3', result:{details:{status:'completed',exitCode:0}}, durationMs:93 });
rec.handle({ toolName:'mcp__jira__transition', params:{}, toolCallId:'c4', result:{id:'10001',key:'ENG-7'} });

// DeepSeek Harness: same format, its own classifier.
const w = createWriter({ dir: '$OUT' });
w.write(recordFor({ name:'bash', arguments:{command:'exit 3'}, callId:'c5' },
                  { isError:false, value:{kind:'foreground', exitCode:3} }));
w.write(recordFor({ name:'bash', arguments:{command:'sleep 1 &'}, callId:'c6' },
                  { isError:false, value:{kind:'background', jobId:'j1'} }));
"

"$PYTHON" - "$OUT" <<'PY'
import sys
from agent_flywheel.outcomes import Trajectory

t = Trajectory(sys.argv[1])
calls = t.read()
assert t.malformed == 0, f"the reader rejected {t.malformed} line(s)"

got = [(c.name, c.outcome) for c in calls]
assert got == [
    # OpenClaw
    ("read", "ok"),
    ("exec", "failed"),          # exit 3, though status said "completed"
    ("exec", "ok"),
    ("mcp__jira__transition", "unknown"),
    # DeepSeek Harness
    ("bash", "failed"),          # exit 3, though it settled as a non-error
    ("bash", "unknown"),         # backgrounded: launched, not finished
], got

assert calls[0].tool_call_id == "c1"   # the join key survives

# Identity crosses the boundary, and survives `unknown` -- an unknown call
# WITH an identity is the one a later read-back can actually settle, so losing
# it here would cost exactly the records that need it most.
assert calls[3].outcome == "unknown"
assert calls[3].entity_ids == {"id": "10001", "key": "ENG-7"}, calls[3].entity_ids

# dsh: a non-zero exit is a failure even though the harness settled it as a
# non-error, and a BACKGROUNDED command has not finished, so it is unknown
# rather than a free success.
assert (calls[4].name, calls[4].outcome, calls[4].error_class) == \
    ("bash", "failed", "ExitStatus:3"), calls[4]
assert (calls[5].outcome, calls[5].error_class) == ("unknown", "StillRunning"), calls[5]

print("cross-language capture contract holds:", got)
print("identity crossed the boundary:", calls[3].entity_ids)
print("dsh classifier agrees with the reader")
PY

# ─────────────────────────────────────────────────────────────────────────────
# THE EXECUTION DIGEST, computed by BOTH implementations over the SAME bytes.
#
# The descriptor and the command vectors are written ONCE, as JSON, and each
# language parses that one file. Writing them twice would have let a typo in
# either copy masquerade as agreement.
#
# Each projection case is hashed into the same descriptor with only
# `entrypoint.argv` replaced, so a digest that moves means the PROJECTION
# moved. The entries themselves are compared too, not just the digests: a
# digest match tells you the bytes agreed, and the entries tell you which byte
# differed when they do not.
#
# FOUR of the five named refusals live in the projection and are checked here.
# The fifth -- declaring volatile command values on an importable, argv-less
# entrypoint -- is raised by Python's RolloutRunner, which has no Node
# counterpart (this package's descriptor carries `entrypoint.kind:
# "host_plugin"` and no argv at all), so it is checked by the Python suite
# alone. A fifth projection-level refusal, an empty command, is checked instead.
#
# ONE refusal is deliberately asymmetric and is checked separately, at
# "node_only_refusals" below: an argv token that is not already a string.
# Python stringifies at the entrypoint, before the projection; Node has no
# entrypoint to stringify at and must not guess, because the two languages
# spell the same value differently. Asserting it in the symmetric table would
# have been a lie about what Python does.
echo
echo "cross-language ABI check: the execution digest"

cat > "$OUT/vectors.json" <<'JSON'
{
  "descriptor": {
    "schema": "percepteye.agent_execution/v1",
    "sdk": "agent-flywheel-abi-check/0",
    "contract_version": "flywheel/1",
    "entrypoint": {"kind": "command", "argv": [{"literal": "agent"}]},
    "configuration": {
      "input_shape": "text",
      "isolation": "subprocess",
      "concurrency": 1,
      "method": null,
      "adapters": "auto",
      "gate_flag": null,
      "endpoint_env": null,
      "reports_tool_calls": false
    },
    "discovered_agent": {
      "agent_name": "customer-agent",
      "unicode": "😀",
      "ordered": {"": 1, "😀": 2},
      "numbers": [1, 1.0, -0.0, 0.1, 9007199254740991],
      "values": [null, true, false, "line\n"]
    },
    "discovery_observation_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "attested_components": {
      "customer.release": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
  },
  "projections": {
    "undeclared_a": {
      "command": ["agent", "--daemon-socket", "/tmp/agent-111.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": []
    },
    "undeclared_b": {
      "command": ["agent", "--daemon-socket", "/tmp/agent-222.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": []
    },
    "declared_a": {
      "command": ["agent", "--daemon-socket", "/tmp/agent-111.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": ["--daemon-socket"]
    },
    "declared_b": {
      "command": ["agent", "--daemon-socket", "/tmp/agent-222.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": ["--daemon-socket"]
    },
    "joined_a": {
      "command": ["agent", "--daemon-socket=/tmp/agent-111.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": ["--daemon-socket"]
    },
    "joined_b": {
      "command": ["agent", "--daemon-socket=/tmp/agent-222.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": ["--daemon-socket"]
    },
    "no_option": {
      "command": ["agent", "--mode", "json", {"task_sentinel": true}],
      "declare": []
    }
  },
  "refusals": {
    "option_absent_from_command": {
      "command": ["agent", "--daemon-socket", "/tmp/agent-111.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": ["--daemon-sock"]
    },
    "option_is_argv0": {
      "command": ["agent", "--daemon-socket", "/tmp/agent-111.sock", "--mode", "json", {"task_sentinel": true}],
      "declare": ["agent"]
    },
    "option_is_last_argument": {
      "command": ["agent", "--mode", "json", "--daemon-socket"],
      "declare": ["--daemon-socket"]
    },
    "value_slot_is_the_task_sentinel": {
      "command": ["agent", "--daemon-socket", {"task_sentinel": true}],
      "declare": ["--daemon-socket"]
    },
    "command_is_empty": {
      "command": [],
      "declare": ["--daemon-socket"]
    }
  },
  "node_only_refusals": {
    "non_string_token": {
      "command": ["agent", "--mode", 1.0, true, null, {"task_sentinel": true}],
      "declare": [],
      "python_entries": [
        {"literal": "agent"}, {"literal": "--mode"}, {"literal": "1.0"},
        {"literal": "True"}, {"literal": "None"}, {"task_input": true}
      ]
    },
    "non_string_value_for_a_declared_option": {
      "command": ["agent", "--daemon-socket", 8080, "--mode", "json"],
      "declare": ["--daemon-socket"],
      "python_entries": [
        {"literal": "agent"}, {"literal": "--daemon-socket"},
        {"volatile_value": "--daemon-socket"}, {"literal": "--mode"},
        {"literal": "json"}
      ]
    }
  }
}
JSON

"$NODE" --input-type=module -e "
import { readFileSync, writeFileSync } from 'node:fs';
import {
  canonicalCommandArgv, executionDescriptorSha256,
  normalizeVolatileCommandValues,
} from '$PKG/execution-identity.js';

const vectors = JSON.parse(readFileSync('$OUT/vectors.json', 'utf8'));
// The argv slot each rollout's task is substituted into. Python spells it
// \`fw.TASK\`; both sides compare it by identity, never by its text, which is
// why the shared file carries a marker each language swaps for its own.
const TASK = Symbol('task');
const materialize = (command) => command.map((item) => (
  item !== null && typeof item === 'object' && item.task_sentinel === true
    ? TASK : item));
const project = (entry) => canonicalCommandArgv(materialize(entry.command), {
  volatileValueOptions: normalizeVolatileCommandValues(entry.declare),
  taskSentinel: TASK,
});

const out = { fixed: executionDescriptorSha256(vectors.descriptor),
              projections: {}, refusals: {} };
for (const [name, entry] of Object.entries(vectors.projections)) {
  const argv = project(entry);
  out.projections[name] = { argv, digest: executionDescriptorSha256({
    ...vectors.descriptor,
    entrypoint: { ...vectors.descriptor.entrypoint, argv },
  }) };
}
for (const [name, entry] of Object.entries(vectors.refusals)) {
  try {
    project(entry);
    out.refusals[name] = null;   // accepted -- the Python half fails on this
  } catch (err) {
    out.refusals[name] = err.name;
  }
}
out.node_only_refusals = {};
for (const [name, entry] of Object.entries(vectors.node_only_refusals)) {
  try {
    project(entry);
    out.node_only_refusals[name] = null;
  } catch (err) {
    out.node_only_refusals[name] = err.name;
  }
}
writeFileSync('$OUT/node-execution.json', JSON.stringify(out));
"

"$PYTHON" - "$OUT/vectors.json" "$OUT/node-execution.json" <<'PY'
import json
import sys

from agent_flywheel import TASK
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.execution_identity import (
    canonical_command_argv,
    execution_descriptor_sha256,
    normalize_volatile_command_values,
)

with open(sys.argv[1], encoding="utf-8") as fh:
    vectors = json.load(fh)
with open(sys.argv[2], encoding="utf-8") as fh:
    node = json.load(fh)

descriptor = vectors["descriptor"]


def materialize(command):
    return [
        TASK if isinstance(item, dict) and item.get("task_sentinel") is True
        else item
        for item in command
    ]


def project(entry):
    return canonical_command_argv(
        materialize(entry["command"]),
        volatile_value_options=normalize_volatile_command_values(entry["declare"]),
        task_sentinel=TASK,
    )


# (a) ONE IDENTICAL DESCRIPTOR. Both languages parsed the same file and hashed
# what they parsed. The pin catches the case the comparison cannot: both halves
# changing together, which would silently re-mint every agent already
# registered against a signed workload.
PINNED_FIXED = "ef0f4220250f17ab96e790a4a436ffd119ce99df9d94415c5a7f1e493d588034"
fixed = execution_descriptor_sha256(descriptor)
assert fixed == node["fixed"], (
    "the SAME descriptor hashes differently in Python and Node:\n"
    f"  python {fixed}\n  node   {node['fixed']}")
assert fixed == PINNED_FIXED, (
    f"the execution descriptor digest MOVED in both languages: {fixed} "
    f"(was {PINNED_FIXED}). Every agent already registered attests the old "
    f"one, and Mission Control refuses an episode whose attestation does not "
    f"match. If this change is deliberate, it is a contract version bump.")

# (b)+(c) THE PROJECTION. Entries first: a digest mismatch says only that some
# byte differed, and the entries say which.
digests = {}
for name, entry in vectors["projections"].items():
    argv = project(entry)
    assert argv == node["projections"][name]["argv"], (
        f"the canonical argv entries differ for {name}:\n"
        f"  python {argv}\n  node   {node['projections'][name]['argv']}")
    digest = execution_descriptor_sha256({**descriptor, "entrypoint": {
        **descriptor["entrypoint"], "argv": argv}})
    assert digest == node["projections"][name]["digest"], (
        f"the execution digest differs for {name}:\n"
        f"  python {digest}\n  node   {node['projections'][name]['digest']}")
    digests[name] = digest

# Each relation is asserted in BOTH languages, not once on the pair: two
# implementations that agreed on a WRONG projection would pass a cross-language
# comparison and nothing else.
for lang, by_name in (("python", digests),
                      ("node", {k: v["digest"]
                                for k, v in node["projections"].items()})):
    # The defect: a per-process socket path re-minted identity on every restart.
    assert by_name["declared_a"] == by_name["declared_b"], lang
    # Nothing is detected -- undeclared, the same two commands stay two agents.
    assert by_name["undeclared_a"] != by_name["undeclared_b"], lang
    # The option TOKEN is still identity, so declaring its value is not the
    # same as not passing the option.
    assert by_name["declared_a"] != by_name["no_option"], lang
    assert by_name["declared_a"] != by_name["undeclared_a"], lang
    # Both spellings collapse; neither is rewritten into the other.
    assert by_name["joined_a"] == by_name["joined_b"], lang
    assert by_name["joined_a"] != by_name["declared_a"], lang

# BACKWARD COMPATIBILITY, cross-language: with nothing declared the projection
# is the plain verbatim argv, and no key is added to the descriptor.
for name in ("undeclared_a", "no_option"):
    entry = vectors["projections"][name]
    verbatim = [
        {"task_input": True}
        if isinstance(item, dict) and item.get("task_sentinel") is True
        else {"literal": str(item)}
        for item in entry["command"]
    ]
    assert project(entry) == verbatim, name
    assert node["projections"][name]["argv"] == verbatim, name

# THE REFUSALS. A declaration that cannot be honoured must stop the process in
# front of whoever wrote it, in BOTH languages. One side accepting it silently
# would mean that side's digest still carries the per-process value.
for name in vectors["refusals"]:
    try:
        project(vectors["refusals"][name])
    except ConfigurationError:
        pass
    else:
        raise AssertionError(f"python accepted the {name} declaration")
    assert node["refusals"][name] == "ConfigurationError", (
        f"node did not refuse the {name} declaration: "
        f"{node['refusals'][name]!r}")

# THE ONE ASYMMETRY, asserted rather than left to a convention.
#
# `str()` and `String()` do not spell the same value the same way: Python
# writes True/1.0/None where JavaScript writes true/1/null. A token that is
# not already a string therefore CANNOT be stringified inside the projection
# without the two SDKs minting different identities for one command.
#
# Python never reaches its projection with such a token -- `RolloutRunner`
# stringifies at the entrypoint (`[a if a is TASK else str(a) for a in
# entrypoint]`), so `str()` happens where the caller still knows what the value
# meant -- and the projection's own `str()` is a no-op on that path. Node has
# no command entrypoint yet, so it has no such boundary to stringify at, and
# refuses the token instead of guessing which language's spelling to use.
#
# Node refuses in EVERY slot, including one a declaration would have voided
# (`non_string_value_for_a_declared_option`), where the token never would have
# been hashed at all. That slot is where a per-process number arrives most
# easily -- a port -- and accepting it there teaches the caller a habit that
# splits the digest the first time the same argv is built without the
# declaration.
#
# Both halves are checked, because each is how the other stays safe: Node must
# REFUSE, and Python must still project these commands its own way. If Python
# ever gains the same refusal, the second assertion goes red -- and the fix
# then is to move these vectors into "refusals" above, where both languages
# are required to refuse.
for name, entry in vectors["node_only_refusals"].items():
    assert node["node_only_refusals"][name] == "ConfigurationError", (
        f"node did not refuse the {name} command: "
        f"{node['node_only_refusals'][name]!r}. A non-string argv token "
        f"stringified in JavaScript is spelled differently than in Python, so "
        f"accepting it means the SAME agent attests a DIFFERENT identity "
        f"depending on which SDK registered it.")
    entries = project(entry)
    assert entries == entry["python_entries"], (
        f"python's projection of the {name} command moved:\n"
        f"  got      {entries}\n  expected {entry['python_entries']}\n"
        f"This is what python does with the argv node refuses -- including "
        f"the spellings only python produces (str(True) == 'True'). If python "
        f"now REFUSES this command too, move the vector into 'refusals', "
        f"where both languages must refuse.")

print("execution descriptor digest agrees:", fixed)
print("volatile-value projection agrees:",
      {k: v[:12] for k, v in sorted(digests.items())})
print("both languages refuse:", ", ".join(sorted(vectors["refusals"])))
print("node refuses the un-stringified argv python stringifies earlier:",
      ", ".join(sorted(vectors["node_only_refusals"])))
PY

# ─────────────────────────────────────────────────────────────────────────────
# THE MODEL BOUNDARY: one golden, two renderers, one on-disk record.
#
# (a) THE GOLDEN IS ONE FILE, TWICE. Each suite renders its own copy, so the
#     copies are compared byte for byte here -- two goldens that drifted would
#     let each SDK pass against a different contract.
# (b) BOTH RENDERERS, EVERY VECTOR. Node's output is compared to Python's by
#     the bytes Mission Control stores (sorted keys, indent 2 -- the digest a
#     retried upload must reproduce), and each to the golden's expectation.
#     The one vector the runtimes spell differently is checked on both sides
#     against its own expectation, and that the two really differ.
# (c) THE ON-DISK RECORD. The Node plugin records a real OpenClaw run -- its
#     llm_input and agent_end hooks, as the host fires them -- into a turn
#     directory; this package's reader renders that directory, and the result
#     must be exactly the trace the plugin itself uploads for the turn.
echo
echo "cross-language ABI check: the model boundary"

PY_GOLDEN="$ROOT/tests/fixtures/genai_capture_golden.json"
NODE_GOLDEN="$SDK/test/fixtures/genai-capture-golden.json"
if ! cmp -s "$PY_GOLDEN" "$NODE_GOLDEN"; then
  echo "✗ the two goldens differ:" >&2
  echo "  $PY_GOLDEN" >&2
  echo "  $NODE_GOLDEN" >&2
  echo "  They are ONE contract; copy the Python repo's file over the Node one." >&2
  exit 1
fi

TURN_DIR="$OUT/genai-turn/run-7"
"$NODE" --input-type=module -e "
import { readFileSync, writeFileSync } from 'node:fs';
import { renderTurnTrace, observationFromLines } from '$PKG/genai-otlp.js';
import { createTurnCapture } from '$PKG/capture.js';
import { turnWire } from '$PKG/turns.js';

const golden = JSON.parse(readFileSync('$NODE_GOLDEN', 'utf8'));
const out = { vectors: {}, lines: {} };
for (const v of golden.vectors) {
  out.vectors[v.name] = renderTurnTrace(v.observation, {
    agentName: v.agent_name, conversationId: v.conversation_id });
}
const a = golden.runtime_asymmetry;
out.asymmetry = renderTurnTrace(a.observation, {
  agentName: a.agent_name, conversationId: a.conversation_id });
for (const v of golden.observation_lines) {
  out.lines[v.name] = observationFromLines(v.lines);
}

// (c) a real OpenClaw run, recorded by the plugin's own turn store.
const turns = createTurnCapture({ root: '$OUT/genai-turn' });
turns.gateModelContent(() => true);
const ctx = { runId: 'run-7', sessionKey: 'conv-é-7' };
turns.onMessageReceived({ runId: 'run-7', content: 'Where is 4417?', sessionKey: 'conv-é-7' }, ctx);
turns.onLlmInput({
  runId: 'run-7', sessionId: 's', provider: 'anthropic', model: 'claude-sonnet-4-5',
  systemPrompt: 'Be exact. ¿Sí?', prompt: 'Where is 4417?', imagesCount: 0,
  historyMessages: [
    { role: 'user', content: 'hi', timestamp: 1758000000000 },
    { role: 'assistant', content: [{ type: 'text', text: 'hello \u{1f4e6}' }],
      provider: 'anthropic', model: 'claude-sonnet-4-5', stopReason: 'stop', timestamp: 1758000000001 },
  ],
  tools: [{ name: 'lookup', description: 'Look it up.',
            parameters: { type: 'object', properties: { id: { type: 'string' } } } }],
}, ctx);
turns.onAgentEnd({ runId: 'run-7', success: true, messages: [
  { role: 'user', content: 'hi', timestamp: 1758000000000 },
  { role: 'assistant', content: [{ type: 'text', text: 'hello \u{1f4e6}' }], stopReason: 'stop', timestamp: 1758000000001 },
  { role: 'user', content: 'Where is 4417?', timestamp: 1758000000002 },
  { role: 'assistant', stopReason: 'toolUse', provider: 'anthropic', model: 'claude-sonnet-4-5', timestamp: 1758000000050,
    content: [{ type: 'thinking', thinking: 'look it up' },
              { type: 'toolCall', id: 'toolu_1', name: 'lookup', arguments: { id: '4417', n: [1, 2, null] } }] },
  { role: 'toolResult', toolCallId: 'toolu_1', toolName: 'lookup', isError: false, timestamp: 1758000000090,
    content: [{ type: 'text', text: 'shipped\ttoday \"now\"' }] },
  { role: 'assistant', stopReason: 'stop', provider: 'anthropic', model: 'claude-sonnet-4-5', timestamp: 1758000000120,
    content: [{ type: 'text', text: 'It shipped today.' }] },
] }, ctx);
out.onDisk = turnWire('$TURN_DIR', { modelContentAgent: 'acme-support' }).genai_trace;
writeFileSync('$OUT/node-genai.json', JSON.stringify(out));
"

"$PYTHON" - "$PY_GOLDEN" "$OUT/node-genai.json" "$TURN_DIR" <<'PY'
import hashlib
import json
import sys

from agent_flywheel.genai_otlp import observation_from_lines, render_turn_trace
from agent_flywheel.outcomes import read_conversation_id, read_model_boundary

with open(sys.argv[1], encoding="utf-8") as fh:
    golden = json.load(fh)
with open(sys.argv[2], encoding="utf-8") as fh:
    node = json.load(fh)


def stored(value):
    """The bytes Mission Control stores for a turn's trace."""
    return json.dumps(value, sort_keys=True, default=str, indent=2).encode()


digests = {}
for v in golden["vectors"]:
    mine = render_turn_trace(v["observation"], agent_name=v["agent_name"],
                             conversation_id=v["conversation_id"])
    theirs = node["vectors"][v["name"]]
    assert stored(mine) == stored(theirs), (
        f"the two renderers wrote different bytes for {v['name']}:\n"
        f"  python {stored(mine)[:400]!r}\n  node   {stored(theirs)[:400]!r}")
    assert stored(mine) == stored(v["genai_trace"]), (
        f"both renderers agree on {v['name']} but not with the golden")
    digests[v["name"]] = hashlib.sha256(stored(mine)).hexdigest()[:12]

a = golden["runtime_asymmetry"]
mine = render_turn_trace(a["observation"], agent_name=a["agent_name"],
                         conversation_id=a["conversation_id"])
assert mine == a["genai_trace_python"], "python moved its side of the asymmetry"
assert node["asymmetry"] == a["genai_trace_node"], "node moved its side of the asymmetry"
assert stored(mine) != stored(node["asymmetry"]), (
    "the one runtime asymmetry no longer differs -- move the vector into "
    "`vectors`, where both renderers must agree")

for v in golden["observation_lines"]:
    assert observation_from_lines(v["lines"]) == v["observation"], v["name"]
    assert node["lines"][v["name"]] == v["observation"], v["name"]

# (c) the plugin's record, read by this package's reader
turn_dir = sys.argv[3]
from_disk = render_turn_trace(read_model_boundary(turn_dir),
                              agent_name="acme-support",
                              conversation_id=read_conversation_id(turn_dir))
assert from_disk is not None, "python read no model boundary from the plugin's turn"
assert stored(from_disk) == stored(node["onDisk"]), (
    "the Node plugin uploads a different trace than its own turn directory "
    f"renders to here:\n  python {stored(from_disk)[:400]!r}\n"
    f"  node   {stored(node['onDisk'])[:400]!r}")
ops = [a["value"]["stringValue"]
       for s in from_disk["resourceSpans"][0]["scopeSpans"][0]["spans"]
       for a in s["attributes"] if a["key"] == "gen_ai.operation.name"]
assert ops == ["invoke_agent", "chat", "execute_tool", "chat"], ops

print("genai capture golden agrees:", digests)
print("both languages spell the one asymmetry their own way")
print("a node-recorded model boundary renders identically in python")
PY
