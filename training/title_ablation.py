"""Acceptance test: does the same page get the same decision when only the title changes?

Found while wiring jev-ultrafast to the System One front end. Same elements, same goal, eight
different page titles -- and "which operation for this text field" was answered correctly twice.
During collection each page's title was fixed, so the title acted as a proxy for the layout and the
model learned the pairing.

A page title carries no information about which operation a field needs, so a model whose answer
depends on it cannot be trusted with an unfamiliar site. This harness reports the spread directly.

    python title_ablation.py --url http://127.0.0.1:8020/v1 --model jeva
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "/root/code/jeva")
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
TITLES = ["Flight Search", "Search flights", "Flights", "Flight search", "Book a flight",
          "Airline tickets", "Google Flights", "飞机票搜索", "Travel", "Plan your trip"]

# The first thing this page needs is text in a field, so the first operation must be TYPE_TEXT.
EXPECTED_OPERATION = "TYPE_TEXT"


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
        return json.loads(text[text.find("{"):text.rfind("}") + 1]).get("operation", "?"), text
    except json.JSONDecodeError:
        return "?", text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1")
    ap.add_argument("--model", default="jeva")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    hits, answers = 0, []
    for title in TITLES:
        page = Page(elements=ELEMENTS, url="https://example.com/flights", title=title)
        operation, _raw = ask(build_prompt(page, GOAL), a.url, a.model, a.temperature)
        good = operation == EXPECTED_OPERATION
        hits += good
        answers.append(operation)
        print(f"  {title:<18s} -> {operation:<10s} {'ok' if good else 'WRONG'}")
    distinct = len(set(answers))
    print(f"\n  {a.model}: {hits}/{len(TITLES)} correct, {distinct} distinct answers for one page")
    print("  A model that reads the state answers this the same way every time.")
    return 0 if hits == len(TITLES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
