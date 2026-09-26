"""Acceptance test: does the same page get the same, valid decision when only the title changes?

Found while wiring jev-ultrafast to the System One front end. Same elements, same goal, eight
different page titles -- and "which operation for this text field" was answered correctly twice.
During collection each page's title was fixed, so the title acted as a proxy for the layout and the
model learned the pairing.

Two things are scored, because the first version of this test scored only one of them and got the
conclusion wrong:

* **invariance** -- a title says nothing about what to do, so every title must produce the same
  answer. This is the property that was broken.
* **validity** -- the answer must be an action that advances the goal. An earlier version demanded
  one specific operation and scored `CLICK 2` (tick the "One way" radio, which the goal requires and
  which is not ticked) as a failure, although the end-to-end run takes exactly that step and
  completes. Choice of which page requirement to satisfy first is not correctness.

Titles are generated fresh on each run. A fixed list drawn from the training pool once reported 9/10
for a model that was in fact answering 7/12 on titles outside it: it had memorised the pool and was
reading the title as a task identifier.

    python title_ablation.py --url http://127.0.0.1:8020/v1 --model jeva
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

sys.path.insert(0, "/root/code/jeva")
sys.path.insert(0, "/root/code/jeva-content")
from jeva.prompt import SYSTEM, build_prompt                      # noqa: E402
from jeva.render import Element, Option, Page                     # noqa: E402

GOAL = ("Find one-way flights from Zurich to London on September 20, 2026, for two adults in "
        "Business.")

# The element table is held fixed; only the title line varies.
ELEMENTS = [
    Element(index="1", role="radio", label="Round trip", value="round", checked="false",
            operations=["CLICK"]),
    Element(index="2", role="radio", label="One way", value="oneway", checked="false",
            operations=["CLICK"]),
    Element(index="3", role="textbox", label="Where from?", value="", operations=["TYPE_TEXT", "CLICK"]),
    Element(index="4", role="textbox", label="Where to?", value="", operations=["TYPE_TEXT", "CLICK"]),
    Element(index="5", role="textbox", label="Departure", value="", operations=["TYPE_TEXT", "CLICK"]),
    Element(index="6", role="combobox", label="Cabin", value="Economy", operations=["SELECT"],
            options=[Option(index="6:1", label="Premium economy"), Option(index="6:2", label="Business")]),
    Element(index="7", role="combobox", label="Passengers", value="1 adult", operations=["SELECT"],
            options=[Option(index="7:1", label="2 adults"), Option(index="7:2", label="3 adults")]),
    Element(index="8", role="button", label="Search", operations=["CLICK"]),
]

# Actions that move this page toward the goal. Clicking a text field is not among them: the prompt
# says CLICK only opens it, and the goal needs text in it.
ACCEPTABLE = {
    ("CLICK", "2"),          # tick One way -- the goal asks for one-way and neither radio is ticked
    ("TYPE_TEXT", "3"),      # Zurich
    ("TYPE_TEXT", "4"),      # London
    ("TYPE_TEXT", "5"),      # September 20, 2026
    ("SELECT", "6:2"),       # Business
    ("SELECT", "7:1"),       # 2 adults
}


def ask(prompt: str, url: str, model: str, temperature: float = 0) -> tuple[str, str]:
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": 64,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1")
    ap.add_argument("--model", default="jeva")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--count", type=int, default=12)
    a = ap.parse_args()

    from surface import random_title
    rng = random.Random(a.seed)
    answers = []
    for _ in range(a.count):
        title = random_title(rng)
        page = Page(elements=ELEMENTS, url="https://example.com/flights", title=title)
        operation, target = ask(build_prompt(page, GOAL), a.url, a.model, a.temperature)
        answers.append((operation, target))
        valid = (operation, target) in ACCEPTABLE
        print(f"  {title[:30]:<32s} -> {operation:<10s} {target:<5s} "
              f"{'' if valid else 'NOT A GOAL STEP'}")
    distinct = len(set(answers))
    valid_n = sum(1 for x in answers if x in ACCEPTABLE)
    print(f"\n  {a.model}: {distinct} distinct answer(s) for one page "
          f"({'invariant' if distinct == 1 else 'TITLE-DEPENDENT'}), "
          f"{valid_n}/{a.count} valid")
    return 0 if distinct == 1 and valid_n == a.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
