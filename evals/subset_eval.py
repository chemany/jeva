"""Evaluate on the task subsets that the random suite under-samples.

    SITE=... LLM_DECISION_URL=... LLM_DECISION_MODEL=... python evals/subset_eval.py <subset> [n]

The random suite averages over every task family, so a gap that only shows up on
goals which need several manual steps stays hidden: it surfaces as one lost task
out of forty. These subsets keep only the goals where the page pre-fills nothing,
so the decider has to do the work itself.

  note      a note that has to be typed
  when      a delivery time that has to be entered
  contact   name, phone and email given as a bare list, all three left blank
  compound  all of the above at once -- closest to an ordinary order form
"""
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                    # evals/ -> browser.py
sys.path.insert(0, os.path.dirname(_HERE))                   # repo root -> jeva package

import collect as C                                          # noqa: E402
from browser import CDP, launch_chrome                       # noqa: E402
from run_eval import run_task                                # noqa: E402
from jeva import Jeva                                        # noqa: E402

SUBSET = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def wanted(spec, url):
    """只保留「必须真的动手」的任务：页面不预填，模型才必须自己补上。"""
    contact = spec.get("contact") or {}
    blank = lambda k: f"{k}=" not in url                        # noqa: E731
    if SUBSET == "note":
        return spec["note"] and blank("note")
    if SUBSET == "when":
        return spec["when"] and blank("when")
    if SUBSET == "contact":
        return len(contact) == 3 and all(blank(k) for k in contact)
    if SUBSET == "compound":
        return (spec["note"] and blank("note") and spec["when"] and blank("when")
                and len(contact) == 3 and all(blank(k) for k in contact))
    raise SystemExit(f"未知子集: {SUBSET}")


jeva = Jeva(os.environ["LLM_DECISION_URL"], model=os.environ["LLM_DECISION_MODEL"])
rng = random.Random(7)
tasks, tries = [], 0
while len(tasks) < N and tries < 40000:
    tries += 1
    t = C.gen_multi(rng)
    if wanted(t[2], t[0]):
        tasks.append(t)

if not tasks:
    raise SystemExit("这个子集生成不出任务")

proc, ws = launch_chrome(port=9362, profile="/tmp/jeva-subset")
cdp = CDP(ws)
ok, fails = 0, []
try:
    for i, t in enumerate(tasks, 1):
        result, status, steps, final = run_task(t, cdp, jeva)
        ok += result
        if not result and len(fails) < 3:
            fails.append((t[1], steps, status, final, t[2]))
        if i % 10 == 0:
            print(f"  {i}/{N} 通过 {ok}", flush=True)
finally:
    try:
        cdp.close()
    except Exception:                                             # noqa: BLE001
        pass
    proc.terminate()

print(f"\n[{SUBSET}] {os.environ['LLM_DECISION_MODEL']}: 通过 {ok}/{len(tasks)} = {ok/len(tasks)*100:.0f}%")
for goal, steps, status, final, spec in fails:
    print(f"  ❌ {goal[:92]}")
    print(f"     status={status}")
    print(f"     提交后 URL: {str(final)[:150]}")
    print(f"     要求 where={spec.get('when')} note={spec.get('note')!r} contact={spec.get('contact')}")
