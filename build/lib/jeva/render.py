"""Render a page observation into the plain-text state jeva was trained on.

jeva reads a *compact plain-text* state, one line per interactive element. The five rules
below are not cosmetic — each was measured to change accuracy (see docs/pipeline.md):

1. Never render an empty string for a missing value. Write ``not set``. An empty value is read
   as "present".
2. Always emit ``{ops: ...}``. It is the only structural signal separating a select-only
   dropdown from a typeable textbox.
3. For a dropdown, list only the **unselected** options, keyed ``<element>:<option>``.
4. Render ``checked=true/false`` for radios and checkboxes.
5. Keep it compact plain text. A JSON blob degrades accuracy substantially.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

OP_ORDER = {"SELECT": 0, "TYPE_TEXT": 1, "CLICK": 2}
_BUTTONISH = {"button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "option"}

OPERATIONS = ("CLICK", "TYPE_TEXT", "SELECT", "WAIT", "DONE", "BLOCKED")

OPERATION_DESCRIPTIONS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field. The caller supplies the value.",
    "SELECT": "Select an observed dropdown value.",
    "WAIT": "Wait because an action is already in progress or the page is still loading.",
    "DONE": "Every requirement is visibly satisfied.",
    "BLOCKED": "No supported operation can progress.",
}


@dataclass
class Option:
    """One entry of a dropdown. ``index`` is the ``"<element>:<option>"`` key."""

    index: str
    label: str
    value: Optional[str] = None


@dataclass
class Element:
    """One interactive element of the observed page.

    ``operations`` is what the element accepts; ``options`` is only for dropdowns and should
    contain the *unselected* entries.
    """

    index: str
    role: str = "element"
    label: str = ""
    value: Optional[str] = None
    operations: Sequence[str] = field(default_factory=list)
    options: Sequence[Option] = field(default_factory=list)
    checked: Optional[object] = None
    selected: Optional[object] = None
    expanded: Optional[object] = None

    def render(self) -> str:
        bits = [f"[{self.index}] {self.role}: {_clean(self.label)}"]
        val = self.value
        if self.role in _BUTTONISH:
            if val not in (None, ""):
                bits.append(f'"{val}"')
        else:
            bits.append("not set" if val in (None, "") else f'"{val}"')   # rule 1
        for key in ("checked", "selected", "expanded"):
            v = getattr(self, key)
            if v not in (None, ""):
                bits.append(f"{key}={v}")                                  # rule 4
        if self.operations:
            ordered = sorted(self.operations, key=lambda o: OP_ORDER.get(o, 9))
            bits.append("{ops: " + ",".join(ordered) + "}")                # rule 2
        if self.options:
            bits.append("options: " + " | ".join(f"[{o.index}] {_clean(o.label)}"
                                                 for o in self.options))   # rule 3
        return " ".join(bits)


@dataclass
class Page:
    """What the browser layer observed: url, title, optional visible text, elements."""

    elements: Sequence[Element]
    url: str = ""
    title: str = ""
    text: str = ""
    page_text_chars: int = 600


def _clean(s) -> str:
    return " ".join(str(s or "").split())


def render_state(page: Page, goal: str, history: Iterable[str] = ()) -> str:
    lines = [f"Page: {_clean(page.title) or 'page'}  ({_clean(page.url)})",
             f"Goal: {_clean(goal)}", ""]
    lines += [e.render() for e in page.elements]
    if page.text:
        lines += ["", "Page text: " + _clean(page.text)[:page.page_text_chars]]
    acts = [_clean(h) for h in history if _clean(h)]
    lines.append("Recent actions: " + ("; ".join(acts) if acts else "none"))
    return "\n".join(lines)


def target_criteria(page: Page, operation: str) -> dict[str, str]:
    """The labels offered to the model for one operation, keyed by target index."""
    out: dict[str, str] = {}
    for e in page.elements:
        if operation == "SELECT":
            for o in e.options:
                out[o.index] = f"[{o.index}] {_clean(e.label)} → {_clean(o.label)}"
        elif operation in e.operations:
            text = f"[{e.index}] {_clean(e.label)}"
            if e.value not in (None, ""):
                text += f" (current: {_clean(e.value)})"
            out[str(e.index)] = text
    return out


def available_operations(page: Page) -> dict[str, str]:
    """The operations that have at least one target, plus terminal operations."""
    ops: dict[str, str] = {}
    for op in ("CLICK", "TYPE_TEXT", "SELECT"):
        if target_criteria(page, op):
            ops[op] = OPERATION_DESCRIPTIONS[op]
    ops["DONE"] = OPERATION_DESCRIPTIONS["DONE"]
    ops["BLOCKED"] = OPERATION_DESCRIPTIONS["BLOCKED"]
    return ops
