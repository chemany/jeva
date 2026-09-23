#!/usr/bin/env python3
"""Real-site test: drive a live website with jeva as the decision maker.

jeva was trained only on the synthetic fixtures in site/. This script points it at a real page
whose markup, labels and layout it has never seen, and reports what it decides.

    python evals/real_site_test.py                       # default: httpbin.org/forms/post
    python evals/real_site_test.py <url> "<goal>" [steps]

Nothing is asserted: the point is to see the behaviour, not to pass a suite.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # repo root -> jeva package
sys.path.insert(0, _HERE)                           # evals/ -> browser.py

from browser import CDP, Browser, action_space, launch_chrome, to_page   # noqa: E402
from jeva import Jeva, render_state, resolve                             # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "https://httpbin.org/forms/post"
GOAL = (sys.argv[2] if len(sys.argv) > 2 else
        "Order one large pizza with mushrooms and cheese, for delivery at 12:30. "
        "Name Jason Zhang, phone 555-0100, email jason@example.com.")
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 14


def main():
    proc, ws = launch_chrome(port=9350, profile="/tmp/jeva-real-site")
    cdp = CDP(ws)
    jeva = Jeva(os.environ.get("JEVA_URL", "http://127.0.0.1:8020/v1"),
                model=os.environ.get("JEVA_MODEL", "jeva"))
    history = []
    print("=" * 100)
    print(f"目标: {GOAL}")
    print(f"真实站点: {URL}")
    print("=" * 100)
    b = None
    try:
        b = Browser(cdp, URL)
        page = b.observe(settle=1.5)
        elements, targets, controls = action_space(page["actions"])
        jp = to_page(page, elements)
        print(f"\n观察到的元素 {len(elements)} 个（jeva 从未见过这个页面）：")
        for e in elements[:14]:
            print(f"  {e['index']:>3s} {e.get('role','?'):<10s} {str(e.get('label'))[:44]:<46s} "
                  f"value={str(e.get('value'))[:18]!r} ops={','.join(e.get('operations') or [])}")
        print(f"\n按 {STEPS} 步运行：\n" + "-" * 100)
        for i in range(STEPS):
            decision = jeva.decide(jp, GOAL, history)
            op, tgt, text = decision.operation, decision.target, decision.text or None
            label = ""
            act = (targets.get(op) or {}).get(tgt) or controls.get(op)
            if act:
                label = str(act.get("label", ""))[:46]
            print(f"[{i+1:02d}] {op:<9s} target={tgt:<6s} {label:<48s} "
                  f"text={text!r}  ({jeva.last_latency_ms:.0f}ms)")
            if os.environ.get("SHOW_RAW") == "1":
                print(f"      raw: {decision.raw.strip()[:160]}")
                print(f"      TYPE_TEXT 候选: {sorted((targets.get('TYPE_TEXT') or {}).keys())}")
                print(f"      CLICK 候选:     {sorted((targets.get('CLICK') or {}).keys())}")
            if op in ("DONE", "BLOCKED"):
                print(f"\n→ 模型判定 {op}")
                break
            if act is None:
                print(f"\n→ 目标 {tgt!r} 不在本页候选里，停下")
                break
            b.act(act, page, text=text)
            history.append(f"{op} {label}")
            page = b.observe(settle=1.0)
            elements, targets, controls = action_space(page["actions"])
            jp = to_page(page, elements)
        print("-" * 100)
        print(f"结束页: {page['url'][:110]}")
        body = (page.get("text") or "")[:400].replace("\n", " ")
        print(f"页面文本: {body}")
    finally:
        if b:
            b.close()
        proc.terminate()


if __name__ == "__main__":
    main()
