#!/usr/bin/env python3
"""A minimal jeva call with the built-in demo observation.

    python examples/quickstart.py                     # expects a server on 127.0.0.1:8020
    JEVA_URL=http://127.0.0.1:8020/v1 python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jeva import Element, Jeva, Page, render_state, resolve        # noqa: E402

PAGE = Page(
    title="Flight Search",
    url="https://example.com/flights",
    text="Search flights. Round trip / One way. Where from? Where to? Departure.",
    elements=[
        Element(index="1", role="radio", label="Round trip", value="round",
                checked="true", operations=["CLICK"]),
        Element(index="2", role="radio", label="One way", value="oneway",
                checked="false", operations=["CLICK"]),
        Element(index="3", role="textbox", label="Where from?", value="Zurich",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="4", role="textbox", label="Where to?", value="London",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="5", role="textbox", label="Departure", value=None,
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="6", role="combobox", label="Cabin", value="Economy", operations=["SELECT"],
                options=[]),
        Element(index="7", role="button", label="Search", operations=["CLICK"]),
    ],
)

GOAL = ("Find one-way flights from Zurich to London on September 20, 2026, "
        "for one adult in Economy.")

if __name__ == "__main__":
    jeva = Jeva(os.environ.get("JEVA_URL", "http://127.0.0.1:8020/v1"),
                model=os.environ.get("JEVA_MODEL", "jeva"))
    print("── state ─────────────────────────────────────────────")
    print(render_state(PAGE, GOAL, history=["CLICK Round trip"]))
    action = jeva.decide(PAGE, GOAL, history=["CLICK Round trip"])
    print("\n── decision ──────────────────────────────────────────")
    print(action.as_dict(), f"   ({jeva.last_latency_ms:.0f} ms)")
    target = resolve(PAGE, action)
    print("resolves to:", target.label if target else "(terminal action)")
