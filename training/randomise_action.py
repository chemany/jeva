"""Rewrite the incidental strings in the action training set.

`sft_v7.jsonl` was collected with one fixed page title per site, so its prompts carry the same
spurious title-to-layout pairing that makes today's model flip on an unfamiliar title. Training on
them unmodified would put the correlation straight back.

Only the `Page: <title>  (<url>)` line is rewritten. The element table, the goal, the page text and
the answer are untouched.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from layout import remap_target, shuffle_layout, verify_identity
from surface import randomise_surface


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/code/jeva/evals/data/sft_v7.jsonl")
    ap.add_argument("--out", default="/root/code/jeva-content/sft_action_surface.jsonl")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--verify", action="store_true", help="check the layout transform is lossless first")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(line) for line in Path(a.src).read_text().splitlines() if line.strip()]
    if a.verify:
        verify_identity([r["prompt"] for r in rows], limit=500)
    for r in rows:
        prompt, remap = shuffle_layout(r["prompt"], rng)
        r["prompt"] = randomise_surface(prompt, rng)
        if remap and r.get("target"):
            r["target"] = remap_target(str(r["target"]), remap)
            r["target_key"] = r["target"]
            r["state"] = r["prompt"]
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    titles = {ln.split("  (")[0][6:] for r in rows for ln in r["prompt"].splitlines()
              if ln.startswith("Page: ")}
    print(f"  {len(rows)} 条动作样本 → {Path(a.out).name}（标题 {len(titles)} 种，元素顺序已重排）")


if __name__ == "__main__":
    main()
