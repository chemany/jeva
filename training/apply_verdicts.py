#!/usr/bin/env python3
"""Apply my per-screenshot verdicts to the homepage benchmark.

The verdicts below were made by looking at `bench/<i>.png` for each row, one at a time. That is the
only way this task can be judged: "the first news item" is not a property that any script can read
off the DOM, which is exactly why the task needed a teacher and why the teacher's own definition
turned out to matter.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

BENCH = Path("/root/code/jeva-content/home_bench.jsonl")

# correct  = the model's block is the page's lead news item
# wrong    = it is something else (site name, utility link, column name, a later item)
# ticker   = it picked a top scrolling headline instead of the lead headline -- defensible, but a
#            different definition from the one used for `correct`
# undefined= the homepage has no single lead item (image carousel or topic banner), so no answer
#            is right and the row should not count either way
VERDICT = {
    0: "correct", 1: "correct", 2: "correct", 3: "ticker", 4: "correct", 5: "undefined",
    6: "correct", 7: "correct", 8: "correct", 9: "correct", 10: "wrong", 11: "wrong",
    12: "correct", 13: "undefined", 14: "correct", 15: "wrong", 16: "undefined", 17: "wrong",
    18: "wrong", 19: "correct", 20: "correct", 21: "correct", 22: "correct", 23: "correct",
    24: "correct", 25: "ticker", 26: "undefined", 27: "wrong", 28: "correct", 29: "wrong",
}
NOTE = {
    3: "选了顶部滚动快讯；主标题是「视频｜习近平同美国总统特朗普会谈」",
    10: "选了栏目名「光华锐评」",
    11: "选了滚动条里的一条",
    15: "选了站名「东方财富」",
    17: "选了语言切换「English」",
    18: "选了「退还2199元订金」",
    25: "选了顶部快讯；主标题是「吴心伯：稳定中美关系…」",
    27: "选了列表第③条；第①条是「俄美乌三方会晤或近期举行」",
    29: "选了「习近平在北京考察调研」",
    5: "sohu 首页是轮播大图，无明确主标题",
    13: "jiemian 首页是轮播，无法判定",
    16: "hexun 顶部无该内容，无法判定",
    26: "xinhuanet 首页是专题横幅，无单条新闻",
}


def main() -> None:
    rows = [json.loads(line) for line in BENCH.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["verdict"] = VERDICT.get(r["i"], "unset")
        r["note"] = NOTE.get(r["i"], "")
    BENCH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                     encoding="utf-8")

    c = collections.Counter(r["verdict"] for r in rows)
    n = len(rows)
    judged = n - c["undefined"]
    print(f"  ══ 首页基准：{n} 个不同首页，逐张人工核对 ══")
    print(f"     correct   {c['correct']:2d}   wrong {c['wrong']:2d}   "
          f"ticker {c['ticker']:2d}   undefined {c['undefined']:2d}")
    print(f"  模型正确率           {c['correct']}/{n} = {c['correct'] / n * 100:.0f}%")
    print(f"  只算口径明确的       {c['correct']}/{judged} = "
          f"{c['correct'] / judged * 100:.0f}%   （排除 {c['undefined']} 个无明确答案的首页）")
    print("\n  失败项：")
    for r in rows:
        if r["verdict"] in ("wrong", "ticker"):
            print(f"    [{r['i']:02d}] {r['site']:<24s} {r['model']['text'][:24]:<26s} {r['note'][:38]}")


if __name__ == "__main__":
    main()
