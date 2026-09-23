#!/usr/bin/env python3
"""在随机任务上评估 agent（LLM 决策 + 真实浏览器），得到统计意义上的成功率。

与 collect.py 共用任务生成器与 verify()，但决策来自 LLM（不含特权提示），
流程与真实部署一致：状态 → 模型 → 动作 → 执行 → 再观察。
"""
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # repo root -> jeva package
sys.path.insert(0, _HERE)                           # evals/ -> browser.py, collect.py

from browser import CDP, Browser, action_space, launch_chrome, to_page   # noqa: E402
import collect as C                                                      # noqa: E402

MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))


def run_task(task, cdp, backend):
    from jeva.render import render_state

    url, goal, spec = task
    b = Browser(cdp, url)
    history, steps = [], []
    status = "ready"
    final = url
    try:
        page = b.observe(screenshot=False)
        for _ in range(MAX_STEPS):
            actions = [a for a in page["actions"] if a.get("kind") != "wait"]
            elements, targets, controls = action_space(actions)
            jpage = to_page(page, elements)
            state = render_state(jpage, goal, history)

            decision = backend.decide(jpage, goal, history)
            op, tgt, text = decision.operation, decision.target, decision.text or None
            if op in ("DONE", "BLOCKED"):
                status = "blocked" if op == "BLOCKED" else "done"
                break
            action = (targets.get(op) or {}).get(tgt) or controls.get(op)
            if action is None:
                status = "invalid_action"
                break
            b.act(action, page, text=text)
            history.append({"action": action["label"], "kind": action["kind"], "text": text})
            steps.append(f"{op}:{action['label'][:24]}")
            page = b.observe(screenshot=False)
        final = page["url"]
    except Exception as exc:                                      # noqa: BLE001
        status = f"error:{type(exc).__name__}"
        if os.environ.get("EVAL_DEBUG") == "1":
            import traceback
            traceback.print_exc()
    finally:
        try:
            b.close()
        except Exception:                                         # noqa: BLE001
            pass
    return C.verify(spec, final, status), status, steps, final


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(os.environ.get("SEED", "999"))
    out_json = os.environ.get("OUT_JSON", "")
    rng = random.Random(seed)

    from jeva import Jeva
    backend = Jeva(os.environ["LLM_DECISION_URL"].rstrip("/").removesuffix("/chat/completions"),
                   model=os.environ.get("LLM_DECISION_MODEL", "jeva"))
    proc, ws = launch_chrome()
    cdp = CDP(ws)

    ok_n = 0
    by_kind = {}
    fails = []
    t0 = time.time()
    try:
        for i in range(n):
            task = rng.choice(C.GENS)(rng)
            kind = task[2]["kind"]
            ok, status, steps, final = run_task(task, cdp, backend)
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][1] += 1
            by_kind[kind][0] += bool(ok)
            ok_n += bool(ok)
            if not ok:
                fails.append({"kind": kind, "goal": task[1][:90], "status": status,
                              "steps": steps, "final": final.split("/")[-1][:80]})
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n}] 成功 {ok_n} ({ok_n/(i+1)*100:.0f}%) | {time.time()-t0:.0f}s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:                                         # noqa: BLE001
            proc.kill()

    ms = sorted(backend.stats["ms"])
    print("=" * 92)
    print(f"随机任务评估（{n} 个，seed={seed}）| 成功率 {ok_n}/{n} = {ok_n/n*100:.1f}%")
    for k, (a, b) in sorted(by_kind.items()):
        print(f"  {k:<9s} {a}/{b} ({a/b*100:.0f}%)")
    if ms:
        print(f"决策延迟 p50 {ms[len(ms)//2]:.0f}ms  p90 {ms[int(len(ms)*0.9)]:.0f}ms  "
              f"| 解析失败 {backend.stats['unparsed']} 目标非法 {backend.stats['bad_target']}")
    print(f"耗时 {time.time()-t0:.0f}s")
    if fails:
        print("失败样例（最多 6 个）:")
        for f in fails[:6]:
            print(f"  ❌ {f['kind']:<8s} {f['goal'][:64]!r} status={f['status']} steps={f['steps']}")
    if out_json:
        json.dump({"n": n, "ok": ok_n, "rate": ok_n / n, "by_kind": by_kind, "fails": fails},
                  open(out_json, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
