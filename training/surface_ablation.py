"""Ablate the incidental surface of an observation, one axis at a time.

The title ablation found a real defect -- the same page and goal answered differently under different
page titles, because each page's title had been fixed during collection and the model learned the
pairing. That fix removes one channel. This looks for the others.

Every case is a page, a goal, and the set of steps that advance that page toward that goal. An axis
changes something that carries no information about which of those steps to take, so the answer must
stay inside the set. Which of the page's requirements is tackled first is left free.

* **label** -- element labels replaced by synonyms ("Where from?" / "Origin" / "From" / "Departure
  city"). Mapping the goal onto the page cannot depend on which word was chosen.
* **distractors** -- extra elements inserted before the form, shifting every index. The index has to
  be read off the row that was chosen, not remembered.
* **order** -- the form shuffled and renumbered.
* **goal** -- the goal reworded without changing what is asked.

    python surface_ablation.py --url http://127.0.0.1:8020/v1 --model jeva
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

sys.path.insert(0, "/root/code/jeva")
from jeva.prompt import SYSTEM, build_prompt                      # noqa: E402
from jeva.render import Element, Option, Page                     # noqa: E402

GOAL = ("Find one-way flights from Zurich to London on September 20, 2026, for two adults in "
        "Business.")

SYNONYMS = {
    "Round trip": ["Round-trip", "Return", "Round trip only"],
    "One way": ["One-way", "Single", "One way only"],
    "Where from?": ["Origin", "From", "Departure city", "Leaving from", "出发地"],
    "Where to?": ["Destination", "To", "Arrival city", "Going to", "目的地"],
    "Departure": ["Date", "When", "Travel date", "Departure date", "出发日期"],
    "Cabin": ["Cabin class", "Travel class", "Seat class", "舱位"],
    "Passengers": ["Travellers", "Guests", "Passenger count", "乘客"],
    "Search": ["Submit", "Find flights", "Go", "搜索"],
}
ROLES = list(SYNONYMS)

GOAL_REWORDINGS = [
    "Book a one-way flight Zurich to London, September 20 2026, 2 adults, Business class.",
    "I need to fly from Zurich to London on 2026-09-20, one way, two adults, business class.",
    "Search one-way flights: Zurich to London, 20 Sep 2026, 2 passengers, Business.",
    "One-way trip Zurich to London departing September 20, 2026 for two adults in business class.",
    "Zurich - London, 20/09/2026, one way, 2 adults, business.",
]

# What a correct first step looks like per role. The index comes from the element list.
STEP_KIND = {"One way": "CLICK", "Where from?": "TYPE_TEXT", "Where to?": "TYPE_TEXT",
             "Departure": "TYPE_TEXT"}
SELECT_KIND = {"Cabin": "2", "Passengers": "1"}       # option number: Business, 2 adults

# Any synonym names the same control, so scoring has to accept it as that control. Without this the
# harness marks the correct step wrong as soon as the label it was expected under is replaced.
CANONICAL = {syn: role for role, syns in SYNONYMS.items() for syn in syns}
CANONICAL.update({role: role for role in SYNONYMS})


def base_elements() -> list[Element]:
    return [
        Element(index="1", role="radio", label="Round trip", value="round", checked="false",
                operations=["CLICK"]),
        Element(index="2", role="radio", label="One way", value="oneway", checked="false",
                operations=["CLICK"]),
        Element(index="3", role="textbox", label="Where from?", value="",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="4", role="textbox", label="Where to?", value="",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="5", role="textbox", label="Departure", value="",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="6", role="combobox", label="Cabin", value="Economy", operations=["SELECT"],
                options=[Option(index="6:1", label="Premium economy"),
                         Option(index="6:2", label="Business")]),
        Element(index="7", role="combobox", label="Passengers", value="1 adult", operations=["SELECT"],
                options=[Option(index="7:1", label="2 adults"), Option(index="7:2", label="3 adults")]),
        Element(index="8", role="button", label="Search", operations=["CLICK"]),
    ]


def relabel(elements: list[Element], mapping: dict[str, str]) -> list[Element]:
    return [Element(index=e.index, role=e.role, label=mapping.get(e.label, e.label), value=e.value,
                    checked=e.checked, operations=list(e.operations), options=list(e.options))
            for e in elements]


def acceptable(elements: list[Element]) -> set[tuple[str, str]]:
    """The (operation, target) pairs that move this page toward the goal.

    A radio is a step when it is not already set -- its value is the option it stands for, not a
    filled-in field. A text field is a step when it is empty. A dropdown is a step only for the
    option the goal asks for. Choosing the wrong radio, or the wrong dropdown option, is excluded.
    """
    out = set()
    for e in elements:
        e = Element(index=e.index, role=e.role, label=CANONICAL.get(e.label, e.label),
                    value=e.value, checked=e.checked, operations=list(e.operations),
                    options=list(e.options))
        if e.label in STEP_KIND:
            if e.role == "radio":
                if e.checked != "true":
                    out.add(("CLICK", e.index))
            elif not e.value:
                out.add((STEP_KIND[e.label], e.index))
        elif e.label in SELECT_KIND and e.options:
            out.add(("SELECT", f"{e.index}:{SELECT_KIND[e.label]}"))
    return out


def ask(prompt: str, url: str, model: str) -> tuple[str, str]:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 64,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = (json.loads(resp.read().decode())["choices"][0]["message"].get("content") or "").strip()
    try:
        obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return obj.get("operation", "?"), str(obj.get("target", "?"))
    except json.JSONDecodeError:
        return "?", "?"


def push_down(elements: list[Element], k: int) -> list[Element]:
    """Insert k decoys at the top and shift every index, option indices included."""
    def shift(idx: str) -> str:
        if ":" in idx:
            head, tail = idx.split(":", 1)
            return f"{int(head) + k}:{tail}"
        return str(int(idx) + k)

    decoys = [Element(index=str(i), role="link", label=f"Promo {i}", operations=["CLICK"])
              for i in range(1, k + 1)]
    moved = [Element(index=shift(e.index), role=e.role, label=e.label, value=e.value,
                     checked=e.checked, operations=list(e.operations),
                     options=[Option(index=shift(o.index), label=o.label) for o in e.options])
             for e in elements]
    return decoys + moved


def renumber(elements: list[Element]) -> list[Element]:
    """Renumber 1..n in place so the list is a legal observation again."""
    out = []
    for i, e in enumerate(elements, 1):
        options = [Option(index=f"{i}:{o.index.split(':')[1]}" if ":" in o.index else o.index,
                          label=o.label)
                   for o in e.options]
        out.append(Element(index=str(i), role=e.role, label=e.label, value=e.value,
                           checked=e.checked, operations=list(e.operations), options=options))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1")
    ap.add_argument("--model", default="jeva")
    a = ap.parse_args()
    rng = random.Random(5)

    cases: list[tuple[str, list[Element], str, set]] = []
    base = base_elements()
    base_ok = acceptable(base)

    for role in ROLES:
        for syn in SYNONYMS[role][:2]:
            el = relabel(base, {role: syn})
            cases.append((f"label: {role!r} -> {syn!r}", el, GOAL, acceptable(el)))

    for k in (1, 3, 6):
        el = push_down(base, k)
        cases.append((f"distractors: {k} inserted before the form", el, GOAL, acceptable(el)))

    for trial in range(3):
        shuffled = list(base)
        rng.shuffle(shuffled)
        el = renumber(shuffled)
        cases.append((f"order: shuffle #{trial + 1}", el, GOAL, acceptable(el)))

    for i, g in enumerate(GOAL_REWORDINGS):
        cases.append((f"goal: reword #{i + 1}", base, g, base_ok))

    hits = 0
    group = None
    for desc, elements, goal, valid in cases:
        head = desc.split(":")[0]
        if head != group:
            group = head
            print(f"\n  ── {group} ──")
        operation, target = ask(
            build_prompt(Page(elements=elements, url="https://example.com/flights", title="x"), goal),
            a.url, a.model)
        good = (operation, target) in valid
        hits += good
        print(f"    {desc:<48s} -> {operation:<10s} {target:<5s} "
              f"{'' if good else 'NOT A GOAL STEP'}")
    print(f"\n  {a.model}: {hits}/{len(cases)} valid steps")
    return 0 if hits == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
