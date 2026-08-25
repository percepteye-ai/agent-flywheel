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
# Requires: node on PATH, and the package importable (pip install -e .).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PKG="./packages/percepteye-agent-flywheel/src"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

PYTHON="${PYTHON:-python}"

node --input-type=module -e "
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
