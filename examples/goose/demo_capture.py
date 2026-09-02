#!/usr/bin/env python3
"""DEMO 1 — the zero-instrumentation capture loop.

No control plane, no worker, no deploy, no API key. Everything a customer does
on day one, plus a stub model endpoint so it runs anywhere.
"""
import http.server, json, os, pathlib, shutil, subprocess, sys, threading

#: This repo, wherever it is checked out — the demo must not carry a
#: developer's absolute path.
SDK = pathlib.Path(__file__).resolve().parents[2]
DEMO = pathlib.Path(os.environ.get("DEMO_ROOT", "/tmp/percepteye-demo"))
#: Prefer the console script beside THIS interpreter, so the demo works
#: from any virtualenv the package is installed into.
PERCEPTEYE = str(pathlib.Path(sys.executable).with_name("percepteye"))
shutil.rmtree(DEMO, ignore_errors=True)
(DEMO / "goose-root" / "config").mkdir(parents=True)
(DEMO / "work").mkdir(parents=True)

# ── a stub model, so the demo needs no provider account ────────────────────
TURNS = [
    {"role": "assistant", "tool_calls": [{"index": 0, "id": "c1", "type": "function",
        "function": {"name": "write", "arguments": json.dumps(
            {"path": "hello.py", "content": "print('ready')\n"})}}]},
    {"role": "assistant", "tool_calls": [{"index": 0, "id": "c2", "type": "function",
        "function": {"name": "shell", "arguments": json.dumps(
            {"command": "python3 hello.py"})}}]},
    {"role": "assistant", "tool_calls": [{"index": 0, "id": "c3", "type": "function",
        "function": {"name": "shell", "arguments": json.dumps(
            {"command": "cat nope.txt"})}}]},          # one deliberate failure
    {"role": "assistant", "content": "Created hello.py; it prints ready."},
]
state = {"n": 0}

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, body, ctype):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self): self._send(b'{"object":"list","data":[]}', "application/json")
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n) or b"{}")
        delta = ({"role": "assistant", "content": "Demo"} if not body.get("tools")
                 else TURNS[min(state["n"], len(TURNS) - 1)])
        if body.get("tools"): state["n"] += 1
        chunks = [{"id":"c","object":"chat.completion.chunk","created":0,"model":"demo",
                   "choices":[{"index":0,"delta":delta,"finish_reason":None}]},
                  {"id":"c","object":"chat.completion.chunk","created":0,"model":"demo",
                   "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}]
        out = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(out.encode(), "text/event-stream")

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]

print("── 1. the ONLY install ──\n   $ pip install percepteye-agent-flywheel\n")
print("── 2. the ONLY command ──")
env = {**os.environ, "GOOSE_PATH_ROOT": str(DEMO / "goose-root")}
subprocess.run([PERCEPTEYE, "install-hooks"], env=env, check=True)

print("\n── 3. the customer runs goose THEIR way — nothing of ours on this line ──")
env.update({
    "PERCEPTEYE_HOME": str(DEMO / "percepteye"),      # where capture lands
    "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1", # the stub, standing in for their provider
    "OPENAI_API_KEY": "demo", "GOOSE_MODE": "auto",
})
r = subprocess.run(
    ["goose", "run", "-q", "--no-profile", "--with-builtin", "developer",
     "--name", "demo", "--provider", "openai", "--model", "demo", "--max-turns", "6",
     "-t", "Create hello.py that prints ready, then run it."],
    cwd=DEMO / "work", env=env, capture_output=True, text=True, timeout=180)
print("   goose said:", (r.stdout or r.stderr).strip().splitlines()[-1][:80] if (r.stdout or r.stderr).strip() else "(nothing)")

print("\n── 4. captured with NO worker, NO rollout, NO control plane ──")
found = sorted((DEMO / "percepteye").rglob("*.jsonl"))
for f in found:
    print(f"   {f.parent.name}/{f.name}")
    for line in f.read_text().splitlines():
        print("     " + line[:150])
if not found:
    print("   (nothing captured)")

print("\n── 5. the day-one payoff, no account and no network ──")
if found:
    subprocess.run([sys.executable,
                    str(SDK / "src/percepteye_agent_flywheel/hosts/scripts/percepteye_report.py"),
                    str(found[0].parent)])
