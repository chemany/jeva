"""jeva command line interface.

Getting running without cloning anything:

    pip install jeva          # runtime only, no dependencies
    jeva download             # fetch the quantised weights (~1.6 GB, cached)
    jeva demo                 # start a local server and make one decision
    jeva serve                # leave the server running on 127.0.0.1:8020

Only the *client* ships in the wheel. Weights are downloaded on demand, because a
1.6 GB GGUF has no business inside a Python distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PORT = 8020
DEFAULT_ALIAS = "jeva"
VARIANTS = {
    "Q4_K_M": "MiniCPM5-2B-WebDecider-v7-Q4_K_M.gguf",   # 1.6 GB, the recommended default
    "Q8_0": "MiniCPM5-2B-WebDecider-v7-Q8_0.gguf",       # 2.7 GB
    "F16": "MiniCPM5-2B-WebDecider-v7-F16.gguf",         # 5.0 GB
}
# 权重来源：按顺序尝试，第一个成功的胜出。设 JEVA_BASE_URL 可覆盖（单一来源）。
BASE_URLS = [
    # ModelScope 优先——国内直连，且一个仓库里同时放了合并版权重和 gguf/
    "https://modelscope.cn/models/imjasonli/jeva/resolve/master/gguf",
    "https://github.com/chemany/jeva/releases/download/v0.2.0",
]
DEFAULT_BASE_URL = os.environ.get("JEVA_BASE_URL", "")


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(os.environ.get("JEVA_CACHE", os.path.join(root, "jeva")))


# ── download ──────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    head = urllib.request.Request(url, method="HEAD")
    total = 0
    try:
        with urllib.request.urlopen(head, timeout=30) as r:
            total = int(r.headers.get("Content-Length") or 0)
    except Exception:                                         # noqa: BLE001
        pass
    print(f"↓ {url}")
    got, t0 = 0, time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = got * 100 / total
                speed = got / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  {pct:5.1f}%  {got/1e9:5.2f}/{total/1e9:.2f} GB  {speed:5.1f} MB/s",
                      end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_download(a) -> int:
    name = VARIANTS[a.variant]
    dest = Path(a.dest) / name if a.dest else cache_dir() / name
    if dest.exists() and not a.force:
        print(f"already cached: {dest}")
        print(f"sha256 {sha256(dest)}")
        return 0
    if a.url:
        sources = [a.url]
    elif a.base_url:
        sources = [f"{a.base_url.rstrip('/')}/{name}"]
    else:
        sources = [f"{b.rstrip('/')}/{name}" for b in BASE_URLS]
    last = None
    for url in sources:
        try:
            _download(url, dest)
            break
        except urllib.error.HTTPError as exc:
            sys.stdout.flush()                   # keep the progress line above the error
            print(f"  {exc.code} from {url}", file=sys.stderr)
            last = exc
        except Exception as exc:                 # noqa: BLE001
            sys.stdout.flush()
            print(f"  {type(exc).__name__} from {url}", file=sys.stderr)
            last = exc
    else:
        print(f"download failed (last: {last})", file=sys.stderr)
        print("  Tried: " + "\n         ".join(sources), file=sys.stderr)
        print("  Point --url (or JEVA_BASE_URL) at a mirror that has the file.", file=sys.stderr)
        return 1
    print(f"✓ {dest}  ({dest.stat().st_size/1e9:.2f} GB)")
    print(f"  sha256 {sha256(dest)}")
    return 0


# ── serve ─────────────────────────────────────────────────────────────────
def find_llama_server() -> str | None:
    for cand in (os.environ.get("LLAMA_SERVER"), "llama-server"):
        if cand and shutil.which(cand):
            return cand
    return None


def serve_command(a) -> list[str] | None:
    if a.backend == "vllm":
        if not shutil.which("vllm") and not _module_exists("vllm"):
            print("vLLM not found. pip install jeva[serve]", file=sys.stderr)
            return None
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", a.model, "--served-model-name", a.alias,
                "--host", "127.0.0.1", "--port", str(a.port),
                "--max-model-len", str(a.ctx), "--dtype", "float16"]
    binary = find_llama_server()
    if not binary:
        print("llama-server not found on PATH.\n"
              "  Install llama.cpp (https://github.com/ggml-org/llama.cpp), or set LLAMA_SERVER,\n"
              "  or use `jeva serve --backend vllm`.", file=sys.stderr)
        return None
    gguf = Path(a.gguf) if a.gguf else cache_dir() / VARIANTS[a.variant]
    if not gguf.exists():
        print(f"weights not found: {gguf}\n  run: jeva download --variant {a.variant}",
              file=sys.stderr)
        return None
    return [binary, "-m", str(gguf), "--alias", a.alias,
            "--host", "127.0.0.1", "--port", str(a.port),
            "-ngl", "99", "-c", str(a.ctx), "--jinja"]


def _module_exists(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def wait_healthy(port: int, timeout: float = 240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                return True
        except Exception:                                     # noqa: BLE001
            time.sleep(0.5)
    return False


def cmd_serve(a) -> int:
    cmd = serve_command(a)
    if not cmd:
        return 1
    print("$", " ".join(cmd), flush=True)
    env = dict(os.environ)
    libdir = os.path.dirname(os.path.abspath(cmd[0]))
    env["LD_LIBRARY_PATH"] = libdir + ":" + env.get("LD_LIBRARY_PATH", "")
    return subprocess.call(cmd, env=env)


# ── demo ──────────────────────────────────────────────────────────────────
DEMO_PAGE = dict(
    title="Flight Search", url="https://example.com/flights",
    text="Search flights. Round trip / One way. Where from? Where to? Departure.",
    elements=[
        dict(index="1", role="radio", label="Round trip", value="round",
             checked="true", operations=["CLICK"]),
        dict(index="2", role="radio", label="One way", value="oneway",
             checked="false", operations=["CLICK"]),
        dict(index="3", role="textbox", label="Where from?", value="Zurich",
             operations=["TYPE_TEXT", "CLICK"]),
        dict(index="4", role="textbox", label="Where to?", value="London",
             operations=["TYPE_TEXT", "CLICK"]),
        dict(index="5", role="textbox", label="Departure", value=None,
             operations=["TYPE_TEXT", "CLICK"]),
        dict(index="7", role="button", label="Search", operations=["CLICK"]),
    ],
)
DEMO_GOAL = ("Find one-way flights from Zurich to London on September 20, 2026, "
             "for one adult in Economy.")


def cmd_demo(a) -> int:
    from . import Element, Jeva, Option, Page, render_state, resolve

    proc = None
    if not wait_healthy(a.port, timeout=1):
        cmd = serve_command(a)
        if not cmd:
            return 1
        print(f"starting jeva on 127.0.0.1:{a.port} …", flush=True)
        proc = subprocess.Popen(cmd, env={**os.environ,
                                          "LD_LIBRARY_PATH": os.path.dirname(os.path.abspath(cmd[0]))
                                          + ":" + os.environ.get("LD_LIBRARY_PATH", "")})
        if not wait_healthy(a.port, timeout=300):
            print("server did not become healthy", file=sys.stderr)
            proc.terminate()
            return 1
        print("ready")
    else:
        print(f"using the server already listening on 127.0.0.1:{a.port}")

    page = Page(
        title=DEMO_PAGE["title"], url=DEMO_PAGE["url"], text=DEMO_PAGE["text"],
        elements=[Element(**{**e, "options": [Option(**o) for o in e.get("options", [])]})
                  for e in DEMO_PAGE["elements"]])
    jeva = Jeva(f"http://127.0.0.1:{a.port}/v1", model=a.alias)
    print("\n── state ─────────────────────────────────────────────")
    print(render_state(page, DEMO_GOAL, history=["CLICK Round trip"]))
    action = jeva.decide(page, DEMO_GOAL, history=["CLICK Round trip"])
    print("\n── decision ──────────────────────────────────────────")
    print(json.dumps(action.as_dict()), f"   ({jeva.last_latency_ms:.0f} ms)")
    target = resolve(page, action)
    print("resolves to:", target.label if target else "(terminal action)")
    print("\nThe goal asks for one way, so jeva does not touch the already-checked radio — "
          "it switches to the one the goal needs.")

    if proc and not a.keep_running:
        print("\nstopping the demo server (use --keep-running to leave it up)")
        proc.terminate()
    elif proc:
        print(f"\nserver left running on 127.0.0.1:{a.port} (Ctrl-C to stop)")
        return proc.wait()
    return 0


# ── check ─────────────────────────────────────────────────────────────────
def cmd_check(_a) -> int:
    from . import __version__
    ok = True
    print(f"jeva {__version__}  (python {sys.version.split()[0]})")
    binary = find_llama_server()
    print(f"  llama-server : {binary or 'NOT FOUND  → install llama.cpp or set LLAMA_SERVER'}")
    ok &= bool(binary)
    for variant, name in VARIANTS.items():
        p = cache_dir() / name
        state = f"cached ({p.stat().st_size/1e9:.2f} GB)" if p.exists() else "not downloaded"
        print(f"  {variant:<7s}      : {state}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_a.port}/health", timeout=2):
            print(f"  server :{_a.port}    : reachable")
    except Exception:                                         # noqa: BLE001
        print(f"  server :{_a.port}    : not running")
    print(f"  cache dir    : {cache_dir()}")
    return 0 if ok else 1


# ── parser ────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("jeva", description="A 2B browser-agent decision model")
    p.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = p.add_subparsers(dest="command")

    def add_serving_args(sp):
        sp.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
        sp.add_argument("--variant", choices=list(VARIANTS), default="Q4_K_M")
        sp.add_argument("--gguf", help="explicit GGUF path (llamacpp)")
        sp.add_argument("--model", help="HF repo id or dir (vllm)")
        sp.add_argument("--port", type=int, default=DEFAULT_PORT)
        sp.add_argument("--alias", default=DEFAULT_ALIAS)
        sp.add_argument("--ctx", type=int, default=4096, help="context size")

    d = sub.add_parser("download", help="fetch the weights into the local cache")
    d.add_argument("--variant", choices=list(VARIANTS), default="Q4_K_M")
    d.add_argument("--dest", help="destination directory (default: jeva cache)")
    d.add_argument("--url", help="explicit URL, overrides --base-url")
    d.add_argument("--base-url", default=DEFAULT_BASE_URL)
    d.add_argument("--force", action="store_true", help="re-download even if cached")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("serve", help="start a local OpenAI-compatible server")
    add_serving_args(s)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("demo", help="start a server if needed and make one decision")
    add_serving_args(m)
    m.add_argument("--keep-running", action="store_true",
                   help="leave the server running after the demo")
    m.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="drive a real browser towards a goal (agent loop)")
    r.add_argument("url", help="page to open")
    r.add_argument("goal", help="one natural-language goal")
    r.add_argument("--steps", type=int, default=25, help="step budget (default 25)")
    r.add_argument("--port", type=int, default=8020, help="jeva server port")
    r.add_argument("--model", default=DEFAULT_ALIAS, help="model alias the server was started with")
    r.add_argument("--chrome-port", type=int, default=9333, help="CDP port for the browser jeva launches")
    r.add_argument("--profile", default="/tmp/jeva-agent-profile", help="throwaway Chrome profile")
    r.add_argument("--chrome", help="explicit Chrome binary")
    r.add_argument("--cdp-ws", help="attach to a browser that is already running (reuse a login)")
    r.add_argument("--screenshots", help="directory to save step screenshots into")
    r.add_argument("--settle", type=float, default=0.6, help="seconds to wait after each action")
    r.add_argument("--json", action="store_true", help="print the full result as JSON")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", help="diagnose the local setup")
    c.add_argument("--port", type=int, default=DEFAULT_PORT)
    c.set_defaults(func=cmd_check)
    return p


def cmd_run(a) -> int:
    from .agent import Agent
    from .client import Jeva
    jeva = Jeva(f"http://127.0.0.1:{a.port}/v1", model=a.model)
    agent = Agent(a.url, a.goal, jeva=jeva, max_steps=a.steps, port=a.chrome_port,
                  profile=a.profile, chrome=a.chrome, cdp_ws=a.cdp_ws,
                  screenshot_dir=a.screenshots, settle=a.settle)
    result = agent.run()
    if a.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        for s in result.steps:
            note = "" if s.page_changed else "  (page unchanged)"
            print(f"  [{s.n:02d}] {s.operation:<9s} {s.target:<6s} {s.label[:44]:<46s} "
                  f"{s.text!r}{note}")
        print(f"\n  {result.status.upper()}: {result.reason}")
        print(f"  ended at {result.url}")
        print(f"  {len(result.steps)} steps, {result.elapsed_ms} ms, "
              f"{result.invalid_targets} invalid targets")
    # The model's DONE is a claim, not proof; the caller checks the page.
    return 0 if result.status == "done" else (1 if result.status == "blocked" else 2)


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "command", None):
        p.print_help()
        return 0
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
