#!/usr/bin/env bash
# The capture contract, checked across the language boundary.
#
# The Node plugins and the Python SDK are two ends of one ABI: whatever a
# plugin appends to $PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl, `Trajectory`
# has to read back with the same meaning. A JS suite that only checks itself
# would let the halves drift -- a plugin staying green while writing a line the
# reader counts as malformed, surfacing later as a silently short tool-call
# list.
#
# THIS FILE EXISTS SO CI AND A DEVELOPER RUN THE SAME CHECK. It used to live
# inline in the workflow, and a local approximation of it passed while the real
# one failed: the JS half had grown two records and the assertion had not. One
# definition, invoked from both places.
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
       && "$cand" -c "import percepteye_agent_flywheel" >/dev/null 2>&1; then
      PYTHON="$cand"; break
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "✗ no python with percepteye_agent_flywheel importable" >&2
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
from percepteye_agent_flywheel.outcomes import Trajectory

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
