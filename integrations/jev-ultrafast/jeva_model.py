"""jeva as the decision model for jev-ultrafast.

Drops in where `model.choose()` used TypeSafe: same inputs (`state`, `goal`, `history`), same output
shape (`choice`, `operation`, `target`, `probabilities`, ...), so `agent.py` and `browser.py` run
untouched. Only the endpoint changes -- that was the point.

Two things are deliberately not identical to TypeSafe:

* **No probabilities.** jeva emits one action, not a distribution. The answer is reported as a
  one-hot, which is what jev's own `validate_choice` accepts (it requires the distribution to sum to
  1.0). Nothing downstream reads it as calibrated, and it is not one.
* **The prompt is jeva's own format.** jeva was fine-tuned on a compact indexed element table
  (`[3] textbox: Where from? "Zurich" {ops: TYPE_TEXT,CLICK}`); feeding it TypeSafe's JSON question
  object instead measured 9% where the flat form measured 82% on the same task. So the questions are
  rendered back into that table rather than sent as JSON.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

try:
    from .model import action_space                 # installed inside jev_ultrafast
except ImportError:                                 # loaded next to it, as integrations/ does
    from jev_ultrafast.model import action_space

DEFAULT_URL = os.environ.get("JEVA_BASE_URL", "http://127.0.0.1:8020/v1")
DEFAULT_MODEL = os.environ.get("JEVA_MODEL", "jeva")

OPERATION_DESCRIPTIONS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field. The caller supplies the value.",
    "SELECT": "Select an observed dropdown value.",
    "DONE": "Every requirement is visibly satisfied.",
    "BLOCKED": "No supported operation can progress.",
}
SYSTEM = "You are a decision model. Reply with JSON only."


def build_prompt(state: dict, goal: str, targets: dict, controls: dict) -> str:
    """Render jev's observation into the compact table jeva was trained on."""
    page = state.get("page") or {}
    lines = [f"Page: {page.get('title') or 'page'}  ({page.get('url') or ''})",
             f"Goal: {goal}", ""]
    for el in state.get("elements") or []:
        bits = [f"[{el['index']}] {el.get('role', 'element')}: {el.get('label', '')}"]
        val = el.get("value")
        if el.get("role") in ("button", "link", "tab", "menuitem"):
            if val:
                bits.append(f'"{val}"')
        else:
            bits.append("not set" if val in (None, "") else f'"{val}"')
        for key in ("checked", "selected", "expanded"):
            if el.get(key) not in (None, ""):
                bits.append(f"{key}={el[key]}")
        ops = list(el.get("operations") or [])
        if ops:
            bits.append("{ops: " + ",".join(ops) + "}")
        if el.get("options"):
            bits.append("options: " + " | ".join(
                f"[{o['index']}] {str(o.get('label', '')).split(' → ')[-1]}" for o in el["options"]))
        lines.append(" ".join(bits))
    if page.get("text"):
        lines.append("")
        lines.append("Page text: " + " ".join(str(page["text"]).split())[:600])
    recent = [h for h in (state.get("recent_actions") or []) if h]
    lines.append("Recent actions: " + ("; ".join(
        f"{h.get('kind', '')} {h.get('action', '')}".strip() for h in recent) or "none"))

    lines += ["", "Available operations:"]
    for key, desc in OPERATION_DESCRIPTIONS.items():
        if key in ("DONE", "BLOCKED") or key in targets:
            lines.append(f"  {key}: {desc}")
    lines.append("")
    lines.append("Note: to enter or change text in a field you MUST use TYPE_TEXT "
                 "(CLICK on a text field only opens it). To choose a dropdown value use SELECT.")
    for op in ("CLICK", "TYPE_TEXT", "SELECT"):
        if op not in targets:
            continue
        crit = []
        for index, a in targets[op].items():
            text = f"[{index}] {str(a.get('label', '')).split(' → ')[0]}"
            cur = a.get("current_value", a.get("value", ""))
            if cur:
                text += f" (current: {crit_clean(cur)})"
            crit.append(text)
        if crit:
            lines.append(f"Targets for {op}: " + " | ".join(crit))
    lines += ["", 'Reply with JSON only, e.g. {"operation":"CLICK","target":"3","text":""}']
    return "\n".join(lines)


# jev asks a second model for the value of a TYPE_TEXT field. jeva returns the value in the same
# decision it returns the operation, so that model is unnecessary -- this carries the text across
# from ``choose`` to ``field_text``, exactly as jev's own ``pending_text`` carries it across a retry.
_PENDING = {"text": ""}


def field_text(context):                                            # noqa: ARG001
    """Return the value jeva already produced. Raises when it produced none."""
    text = _PENDING.get("text") or ""
    if not text.strip():
        raise ValueError("jeva gave no text for a fill action; nothing typed.")
    return text, {"model": DEFAULT_MODEL, "latency_ms": 0, "usage": {}}


def crit_clean(value) -> str:
    text = " ".join(str(value).split())
    return text[:60] + ("…" if len(text) > 60 else "")


def post(url: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def choose(state, goal, history):
    """Same contract as ``model.choose`` but answered by the local jeva server."""
    elements, targets, controls = action_space(state["actions"])
    prompt = build_prompt({"page": {k: state[k] for k in ("url", "title", "text")},
                           "elements": elements,
                           "recent_actions": [
                               {"action": h.get("action"), "kind": h.get("kind"),
                                "text": h.get("text"), "page_changed": h.get("page_changed")}
                               for h in history[-10:]]},
                          goal, targets, controls)

    started = time.perf_counter()
    body = {"model": DEFAULT_MODEL, "max_tokens": 96, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},   # else `content` comes back empty
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]}
    try:
        payload = post(DEFAULT_URL, body)
        content = (payload["choices"][0]["message"].get("content") or "").strip()
        answer = json.loads(content[content.find("{"):content.rfind("}") + 1])
    except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"jeva did not answer ({type(exc).__name__}); no action executed.") from None

    operation = str(answer.get("operation") or "").upper()
    target = str(answer.get("target") or "")
    text = answer.get("text") or ""
    _PENDING["text"] = text

    if operation in ("DONE", "BLOCKED"):
        choice = operation
    elif operation in targets and target in targets[operation]:
        choice = targets[operation][target]["id"]
    elif operation == "WAIT" and "WAIT" in controls:
        choice = controls["WAIT"]["id"]
    else:
        # jeva named an operation or target this page does not offer. jev never executes a target it
        # did not offer, so this is reported as the model having nothing to do rather than guessed.
        raise RuntimeError(
            f"jeva answered {operation} {target!r}, which this page does not offer; no action executed.")

    return {
        "choice": choice,
        "operation": operation,
        "target": target or None,
        "confidence": 1.0,                 # one-hot; see the module docstring
        "probabilities": {choice: 1.0},
        "operation_probabilities": {operation: 1.0},
        "target_probabilities": {target: 1.0} if target else {},
        "target_confidence": 1.0 if target else None,
        "raw_answers": answer,
        "model": DEFAULT_MODEL,
        "usage": payload.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }
