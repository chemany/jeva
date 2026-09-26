#!/usr/bin/env python3
"""Run jev-ultrafast's own agent with jeva as the decision model.

    python run.py "<goal>" "<url>"

jev-ultrafast is browser-use's reference agent (MIT). Its browser layer talks to a real Chrome
through browser-harness, and its loop carries the guards -- staleness checks, one-shot decisions, a
repeat detector. Only the decision endpoint changes here: `jeva_model.choose` replaces the TypeSafe
call. agent.py, browser.py and snapshot.js are used unmodified.

Requirements:
  * a clone of jev-ultrafast importable on sys.path (set JEVA_JEV_REPO, default /tmp/jev-ultrafast)
  * `browser-harness` installed (it is, in the 1cat-vllm venv)
  * a local jeva server on 127.0.0.1:8020

browser-harness normally wants the user's own Chrome with remote debugging approved by hand on every
connection, which is unusable unattended. It also accepts BU_CDP_WS, so this launches a throwaway
Chrome and points it there.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

JEV_REPO = os.environ.get("JEVA_JEV_REPO", "/tmp/jev-ultrafast")
CHROME_PORT = int(os.environ.get("JEVA_CHROME_PORT", "9700"))
CHROME_PROFILE = os.environ.get("JEVA_CHROME_PROFILE", "/tmp/jev-chrome")
CHROME_FLAGS = [
    "--headless=new", "--no-first-run", "--no-sandbox", "--disable-gpu",
    "--disable-gpu-compositing", "--disable-software-rasterizer",
    "--disable-accelerated-2d-canvas", "--disable-features=VizDisplayCompositor",
    "--hide-scrollbars", "--window-size=1400,1000",
]


def launch_chrome() -> tuple[subprocess.Popen, str]:
    """Start Chrome and return it with its browser-level websocket url."""
    import shutil
    shutil.rmtree(CHROME_PROFILE, ignore_errors=True)
    binary = (os.environ.get("CHROME") or shutil.which("google-chrome")
              or shutil.which("chromium") or shutil.which("chromium-browser"))
    if not binary:
        raise SystemExit("No Chrome/Chromium found; set $CHROME")
    proc = subprocess.Popen([binary, f"--remote-debugging-port={CHROME_PORT}",
                             f"--user-data-dir={CHROME_PROFILE}", *CHROME_FLAGS],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{CHROME_PORT}/json/version", timeout=2) as r:
                return proc, json.loads(r.read())["webSocketDebuggerUrl"]
        except Exception:                                             # noqa: BLE001
            time.sleep(0.25)
    proc.kill()
    raise SystemExit("Chrome did not become ready in 20s")


def main() -> int:
    goal = sys.argv[1] if len(sys.argv) > 1 else "Find one-way flights from Zurich to London on September 20, 2026, for two adults in Business."
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8899/flights.html"
    if not os.path.isdir(JEV_REPO):
        raise SystemExit(f"jev-ultrafast not found at {JEV_REPO}; set JEVA_JEV_REPO")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))    # jeva_model.py
    sys.path.insert(0, JEV_REPO)

    proc, ws = launch_chrome()
    os.environ["BU_CDP_WS"] = ws
    os.environ.setdefault("JEVA_BASE_URL", "http://127.0.0.1:8020/v1")
    os.environ.setdefault("JEVA_MODEL", "jeva")
    try:
        import jev_ultrafast.agent as agent_mod
        import jev_ultrafast.model as model_mod
        import jeva_model
        # agent.py binds `choose` and `field_text` at import time, so both names are replaced.
        model_mod.choose = jeva_model.choose
        agent_mod.choose = jeva_model.choose
        model_mod.field_text = jeva_model.field_text
        agent_mod.field_text = jeva_model.field_text

        started = time.time()
        with agent_mod.Agent(url, goal) as agent:
            for snap in agent.run():
                hist = snap.get("history") or []
                if hist:
                    h = hist[-1]
                    print(f"  {h['operation']:<9s} {str(h.get('target')):<5s} "
                          f"{h['action'][:38]:<40s} {h.get('text')!r}", flush=True)
                if snap["status"] in ("done", "blocked"):
                    page = snap.get("page") or {}
                    print(f"\n  {snap['status'].upper()} | {time.time() - started:.1f}s | "
                          f"{page.get('url', '')}")
                    return 0 if snap["status"] == "done" else 1
    finally:
        proc.terminate()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
