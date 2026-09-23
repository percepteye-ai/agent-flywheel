# Vendored OpenTelemetry GenAI JSON schemas

These files are copied **byte for byte** from the OpenTelemetry GenAI semantic
conventions, so the conformance test can run offline and cannot drift with the
upstream branch:

| file | upstream path |
|:--|:--|
| `gen-ai-input-messages.json` | `model/gen-ai/gen-ai-input-messages.json` |
| `gen-ai-output-messages.json` | `model/gen-ai/gen-ai-output-messages.json` |
| `gen-ai-system-instructions.json` | `model/gen-ai/gen-ai-system-instructions.json` |
| `gen-ai-tool-call-arguments.json` | `model/gen-ai/gen-ai-tool-call-arguments.json` |
| `gen-ai-tool-call-result.json` | `model/gen-ai/gen-ai-tool-call-result.json` |

- Repository: <https://github.com/open-telemetry/semantic-conventions-genai>
- Commit: `8ffdf568e1b4391a99adb081db16e8102e36918e` (2026-09-22). The
  repository has no tags or releases; every `gen_ai.*` item is at stability
  *development*.
- License: Apache-2.0, as upstream.

The schemas carry no `$schema` key (they are generated), so
`tests/test_genai_conformance.py` pins **Draft 2020-12** explicitly. Every part
list ends in a `GenericPart` that requires only `type`, which makes the schema
alone nearly vacuous; the test therefore dispatches each known part `type` to
its specific `$defs` entry, and proves the gate can fail.

To move the pin, replace all five files from one upstream commit, update the
commit above and in the golden fixture's `standard.commit`, and rerun the suite.
