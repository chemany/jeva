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

from build_dataset import TITLE_POOL, URL_POOL, randomise_surface


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/code/jeva/evals/data/sft_v7.jsonl")
    ap.add_argument("--out", default="/root/code/jeva-content/sft_action_surface.jsonl")
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(line) for line in Path(a.src).read_text().splitlines() if line.strip()]
    for r in rows:
        r["prompt"] = randomise_surface(r["prompt"], rng)
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    titles = {ln.split("  (")[0][6:] for r in rows for ln in r["prompt"].splitlines()
              if ln.startswith("Page: ")}
    print(f"  {len(rows)} 条动作样本 → {Path(a.out).name}（标题 {len(titles)} 种）")


if __name__ == "__main__":
    main()
