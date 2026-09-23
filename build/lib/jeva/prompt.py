"""The prompt jeva was trained on. Treat both halves as part of the model.

jeva is specialised to this exact wording. Changing the system prompt or the layout of the
user message measurably degrades output — the ablation is in docs/pipeline.md. If you need a
different contract, fine-tune rather than reword.
"""
from __future__ import annotations

from .render import Page, available_operations, render_state, target_criteria

SYSTEM = """You are a browser agent. Choose exactly ONE next action.
Output compact JSON only, no prose:
{"operation":"<one of the operations>","target":"<an offered target index, or empty>","text":"<only for TYPE_TEXT, else empty>"}

Hard rules:
- Never repeat a step that is already satisfied. A field that already holds the required value needs no
  further typing; a radio/checkbox already in the requested state needs no further click.
- If a suggestion/option list is open, click the matching option instead of typing again.
- DONE only when visible evidence proves every requirement of the goal is met.
- BLOCKED only when a challenge or a missing control prevents progress."""

NOTE = ("Note: to enter or change text in a field you MUST use TYPE_TEXT (CLICK on a text field "
        "only opens it). To choose a dropdown value use SELECT. Clicking a radio/checkbox that "
        "already has the requested state does nothing.")

DEFAULT_RULES = "Advance the goal by exactly one operation."


def build_prompt(page: Page, goal: str, history=(), rules: str = DEFAULT_RULES,
                 state: str | None = None) -> str:
    """Build the user message. Pass ``state`` to reuse a state you already rendered."""
    state_text = state if state is not None else render_state(page, goal, history)
    ops = available_operations(page)
    lines = [state_text, "", "Available operations:"]
    lines += [f"  {k}: {v}" for k, v in ops.items()]
    lines += ["", NOTE]
    for op in ("CLICK", "TYPE_TEXT", "SELECT"):
        if op not in ops:
            continue
        crit = target_criteria(page, op)
        if not crit:
            continue
        # target_criteria 的值里已含 "[index]" 前缀，别再包一层
        lines.append(f"Targets for {op}: " + " | ".join(crit.values()))
    lines += ["", f"Policy rules: {rules}", "",
              'Reply with JSON only, e.g. {"operation":"CLICK","target":"3","text":""}']
    return "\n".join(lines)
