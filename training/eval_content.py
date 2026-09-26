"""Measure the content task: does the model pick the same first article the teacher did?

Run against any OpenAI-compatible endpoint, so the same script scores jeva, the teacher, and the
rule-based baseline -- a number means nothing without the two reference points next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "/root/code/jeva")
from jeva.render import Element, Page                                 # noqa: E402
from jeva.prompt import SYSTEM, build_prompt                          # noqa: E402

GOAL = "What is the first news article on this page?"


def to_element(i, c):
    meta = f"{c.get('host') or ''} {c['size']:.0f}px col{c['column']} row{c['y']}".strip()
    return Element(index=str(i), role="textblock", label=c["t"], operations=["READ"], meta=meta)


def judge(cands, content, url, title):
    """Return the model's chosen index, or None."""
    page = Page(elements=[to_element(i, c) for i, c in enumerate(cands, 1)], url=url, title=title)
    prompt = build_prompt(page, GOAL)
    return prompt


def ask(prompt, url, model):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 64,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=180).read())
    text = (out["choices"][0]["message"].get("content") or "").strip()
    try:
        parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:                                                 # noqa: BLE001
        return None, text[:40]
    if parsed.get("operation") != "READ":
        return None, text[:40]
    try:
        return int(parsed.get("target")), text[:40]
    except (TypeError, ValueError):
        return None, text[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/code/jeva-content/raw.jsonl")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1")
    ap.add_argument("--model", default="jeva")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.raw).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    ok = 0
    for r in rows:
        prompt = judge(r["cands"], r["content"], r["url"], r["title"])
        got, raw = ask(prompt, a.url, a.model)
        hit = got == r["first"]
        ok += hit
        if a.show or not hit:
            picked = r["cands"][got - 1]["t"][:34] if got and 1 <= got <= len(r["cands"]) else f"<{raw}>"
            print(f"    {'✓' if hit else '✗'} {r['url'].split('//')[1][:24]:<26s} "
                  f"应 [{r['first']:2d}] 得 [{got}]  {picked}")
    n = len(rows)
    print(f"\n  {a.model} @ {a.split}: {ok}/{n} = {ok/n*100:.0f}%" if n else "  no rows")


if __name__ == "__main__":
    main()
