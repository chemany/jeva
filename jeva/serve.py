"""Launch a local jeva server.

Two supported paths:

* **llama.cpp** (recommended, ~250 ms/decision for the Q4_K_M GGUF on a V100):

      python -m jeva.serve --backend llamacpp --gguf MiniCPM5-2B-WebDecider-Q4_K_M.gguf

* **vLLM** (if you already run vLLM; slower for this workload, ~950 ms):

      python -m jeva.serve --backend vllm --model MiniCPM5-2B-WebDecider

Both expose an OpenAI-compatible /v1/chat/completions endpoint, which is all Jeva needs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_ALIAS = "jeva"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve jeva locally")
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    ap.add_argument("--gguf", help="path to a jeva GGUF (llamacpp)")
    ap.add_argument("--model", help="HF repo id or local dir (vllm)")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--alias", default=DEFAULT_ALIAS)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server")
    a = ap.parse_args(argv)

    if a.backend == "llamacpp":
        if not a.gguf:
            ap.error("--gguf is required for the llamacpp backend")
        cmd = [a.llama_server, "-m", a.gguf, "--alias", a.alias,
               "--host", "127.0.0.1", "--port", str(a.port),
               "-ngl", "99", "-c", str(a.ctx_size), "--jinja"]
    else:
        if not a.model:
            ap.error("--model is required for the vllm backend")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", a.model, "--served-model-name", a.alias,
               "--port", str(a.port), "--max-model-len", str(a.ctx_size), "--dtype", "float16"]

    print(" ".join(cmd), flush=True)
    # llama.cpp needs its shared libraries on the loader path when run from a build tree
    env = dict(os.environ)
    libdir = os.path.dirname(os.path.abspath(a.llama_server))
    if libdir:
        env["LD_LIBRARY_PATH"] = libdir + ":" + env.get("LD_LIBRARY_PATH", "")
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
