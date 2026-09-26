"""Serve jeva over TypeSafe's System One protocol, so a TypeSafe client needs no adapter.

TypeSafe clients post ``{model, state, questions}`` to ``/v1/systemone`` and expect
``{answers: {question_id: {choice, probabilities, confidence}}}``. That is the contract
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) calls, and it is what this
module speaks, backed by the local jeva model.

Why this exists rather than an in-process shim: a client should be able to point its base URL here
and change nothing else. The shim in ``integrations/jev-ultrafast/`` proves the swap works, but it
edits the client's process; this does not.

Two honest limitations, both because jeva is not a System One model:

* **Probabilities are one-hot.** jeva emits one action, not a distribution, so the chosen option is
  reported as 1.0 and the rest as 0.0. TypeSafe's own validator requires a distribution over exactly
  the offered options summing to 1.0, which a one-hot satisfies; nothing should read it as
  calibrated. Measured on the content task, correct answers carried a median top-1 of 48% while
  wrong ones carried 80% -- a confidence signal here points the wrong way.
* **One question set at a time, answered as one action.** System One can fan out several independent
  questions over one state. jeva was trained to choose one operation and one target, so the
  operation question and the matching target question are answered together and any further
  questions are answered as "the same single choice".
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import Jeva
from .prompt import build_prompt
from .render import Element, Option, Page


def page_from_state(state: dict) -> Page:
    """Build a jeva Page from a System One ``state``.

    The element shape already matches -- ``index/label/role/value/checked/selected/expanded`` are the
    fields jev's own ``action_space`` produces, which is where jeva's renderer got its format.
    """
    elements = []
    for el in state.get("elements") or []:
        # jev labels a dropdown option "<field> → <option>"; keeping the prefix makes the renderer
        # print "Cabin → Cabin → Premium economy".
        options = [Option(index=str(o.get("index")),
                          label=str(o.get("label", "")).split(" → ")[-1],
                          value=o.get("value"))
                   for o in (el.get("options") or [])]
        elements.append(Element(
            index=str(el.get("index")), role=el.get("role", "element"),
            label=str(el.get("label", "")), value=el.get("value"),
            operations=list(el.get("operations") or []), options=options,
            checked=el.get("checked"), selected=el.get("selected"),
            expanded=el.get("expanded")))
    page = state.get("page") or {}
    return Page(elements=elements, url=page.get("url", ""), title=page.get("title", ""),
                text=page.get("text", ""))


def goal_from_questions(questions: dict) -> str:
    for question in questions.values():
        goal = ((question.get("instructions") or {}).get("goal") or "").strip()
        if goal:
            return goal
    return ""


def answer_question(ids: list[str], choice: str, confidence: float = 1.0) -> dict:
    """One System One answer: a one-hot over exactly the offered options."""
    probabilities = {key: 0.0 for key in ids}
    probabilities[choice] = 1.0
    return {"choice": choice, "probabilities": probabilities, "confidence": confidence}


def resolve(questions: dict, action) -> dict:
    """Map jeva's single action onto every question in the payload."""
    answers: dict[str, dict] = {}
    # The operation question decides; the target question is answered from the same action.
    target_key = f"{action.operation.lower()}_target"
    for qid, question in questions.items():
        ids = list((question.get("criteria") or {}).keys())
        if not ids:
            continue
        if qid == target_key and action.target in ids:
            answers[qid] = answer_question(ids, action.target)
        elif action.operation in ids:
            answers[qid] = answer_question(ids, action.operation)
        elif qid == target_key and ids:
            # The model named a target this page does not offer. Reporting the first option would be
            # a guess, so say nothing and let the client treat the response as invalid.
            continue
        else:
            answers[qid] = answer_question(ids, ids[0])
    return answers


class Handler(BaseHTTPRequestHandler):
    jeva: Jeva = None                                   # set by serve()
    model_name = "jeva"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):                       # keep the console quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                   # noqa: N802
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(200, {"status": "ok", "model": self.model_name})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):                                  # noqa: N802
        if not self.path.rstrip("/").endswith("systemone"):
            self._send(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode() or "{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": {"message": "body is not JSON"}})
            return

        questions = body.get("questions") or {}
        state = body.get("state") or {}
        if not questions:
            self._send(400, {"error": {"message": "no questions"}})
            return
        try:
            page = page_from_state(state)
            goal = goal_from_questions(questions)
            started = time.perf_counter()
            action = self.jeva.decide(page, goal)
            answers = resolve(questions, action)
            import os
            if os.environ.get("JEVA_SYSTEMONE_DEBUG"):
                with open(os.environ["JEVA_SYSTEMONE_DEBUG"], "a") as fh:
                    fh.write(json.dumps({
                        "goal": goal, "prompt": build_prompt(page, goal),
                        "elements": [{"i": e.index, "role": e.role, "label": e.label,
                                      "ops": list(e.operations)} for e in page.elements],
                        "action": {"op": action.operation, "target": action.target,
                                   "text": action.text},
                        "questions": {k: list((q.get("criteria") or {}).keys())
                                      for k, q in questions.items()},
                        "answers": answers}, ensure_ascii=False) + "\n")
        except Exception as exc:                        # noqa: BLE001
            # TypeSafe clients treat a failed response as "no action executed", which is right.
            self._send(502, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return
        if not answers:
            self._send(422, {"error": {"message":
                                       f"jeva answered {action.operation} {action.target!r}, which "
                                       f"this state does not offer"}})
            return
        self._send(200, {
            "answers": answers,
            "model": body.get("model") or self.model_name,
            "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            # Additive, non-standard. System One questions are pure choices and never carry free
            # text, so a TYPE_TEXT value cannot come back through `answers`. jeva does produce one,
            # and a client that knows to look here can use it; a client that does not ignores this
            # key and behaves exactly as it would against TypeSafe.
            "jeva": {"operation": action.operation, "target": action.target, "text": action.text},
        })


def serve(*, port: int = 8021, model: str = "jeva",
          base_url: str = "http://127.0.0.1:8020/v1", host: str = "127.0.0.1") -> None:
    """Run the System One compatible front end until interrupted."""
    handler = type("BoundHandler", (Handler,), {"jeva": Jeva(base_url, model=model),
                                                "model_name": model})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"jeva systemone listening on http://{host}:{port}/v1/systemone "
          f"(backend {base_url})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def probe(base_url: str, state: dict, questions: dict, timeout: int = 60) -> dict:
    """Send one System One request, for tests and smoke checks."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps({"model": "jeva", "state": state, "questions": questions}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
