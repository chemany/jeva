"""Mix the content examples with the action examples into one training file.

One model, two jobs: act on a page, and answer what a page says. The content set is small
(hundreds, not thousands), so it is upsampled -- otherwise it is 3% of the gradient and the model
learns it the way a person learns a language from a footnote.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ACTION = "/root/code/jeva-content/sft_action_surface.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", default="/root/code/jeva-content/content_train.jsonl")
    ap.add_argument("--action", default=ACTION)
    ap.add_argument("--out", default="/root/code/jeva-content/mixed_train.jsonl")
    ap.add_argument("--content-repeat", type=int, default=3)
    ap.add_argument("--action-cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    content = [json.loads(l) for l in Path(a.content).read_text().splitlines() if l.strip()]
    action = [json.loads(l) for l in Path(a.action).read_text().splitlines() if l.strip()]
    if a.action_cap and len(action) > a.action_cap:
        action = rng.sample(action, a.action_cap)

    rows = []
    for _ in range(max(1, a.content_repeat)):
        rows += content
    rows += action
    rng.shuffle(rows)
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n_c = len(rows) - len(action)
    print(f"  内容 {len(content)} × {a.content_repeat} = {n_c} + 动作 {len(action)} "
          f"= {len(rows)} 条 → {Path(a.out).name}")
    print(f"  内容占比 {n_c/len(rows)*100:.0f}%")


if __name__ == "__main__":
    main()
