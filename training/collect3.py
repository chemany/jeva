"""Full corpus with a clean site-level split.

Collection is cheap now that the teacher's reasoning is off (`reasoning_effort: none` took a page
from ~85 s to ~5 s). The split is by SITE, not by page: a model that has seen sina's layout family
and is tested on ifeng's has demonstrated something, whereas a page-level split mostly measures
memorisation.
"""
from __future__ import annotations
import json, os, sys, threading, time, urllib.request
sys.path.insert(0, "/root/code/jeva")
from jeva.browser import CDP, Browser, close_chrome, launch_chrome       # noqa: E402

TEACHERS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8002/v1"]
TRAIN = ["https://sports.sina.com.cn/", "https://finance.sina.com.cn/", "https://news.sina.com.cn/",
         "https://ent.sina.com.cn/", "https://tech.sina.com.cn/", "https://mil.news.sina.com.cn/",
         "https://news.163.com/", "https://www.thepaper.cn/", "https://www.sohu.com/",
         "https://www.people.com.cn/", "https://www.chinanews.com.cn/"]
HOLDOUT = ["https://news.ifeng.com/", "https://www.chinadaily.com.cn/", "https://www.xinhuanet.com/",
           "https://www.guancha.cn/", "https://www.cctv.com/"]
LOADS = 5
PORT_BASE = 9470
OUT = "/root/code/jeva-content/raw3.jsonl"
lock = threading.Lock()
done = {"train": 0, "holdout": 0, "fail": 0}


def label(cands, url, endpoint, tries=2):
    listing = "\n".join(f'[{i}] {c["size"]:.0f}px col{c["column"]} y={c["y"]} host={c.get("host","")} :: {c["t"]}'
                        for i, c in enumerate(cands, 1))
    prompt = ("Mark which blocks are news article links. Exclude navigation, section tabs, "
              "hot-search lists, advertisements and promotions.\n\n"
              f"Page: {url}\nBlocks:\n{listing}\n\nReply with JSON only: {{\"content\": [indices]}}")
    for _ in range(tries):
        try:
            body = json.dumps({"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 2000,
                               "reasoning_effort": "none",
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            text = json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"].get("content") or ""
            parsed = json.loads(text[text.find("{"):text.rfind("}") + 1]).get("content")
            if isinstance(parsed, list):
                idx = sorted({int(i) for i in parsed if str(i).strip().isdigit()})
                if idx:
                    return [i for i in idx if 1 <= i <= len(cands)]
        except Exception:                                                 # noqa: BLE001
            time.sleep(2)
    return None


def one(url, split, endpoint, port):
    proc = cdp = None
    try:
        proc, ws = launch_chrome(port=port, profile=f"/tmp/c3-{port}")
        cdp = CDP(ws)
        b = Browser(cdp, url, width=1400, height=1000)
        time.sleep(3.5); b.dismiss_overlays(); time.sleep(0.5)
        cands = b.all_blocks(30)
        if len(cands) < 8:
            raise RuntimeError(f"只有 {len(cands)} 个候选")
        idx = label(cands, url, endpoint)
        if not idx:
            raise RuntimeError("老师未标注")
        rec = {"split": split, "site": url.split("//")[1].split("/")[0], "url": url,
               "title": b.evaluate("document.title"), "cands": cands,
               "content": idx, "first": min(idx)}
        with lock:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[split] += 1
        return True
    except Exception as exc:                                              # noqa: BLE001
        with lock:
            done["fail"] += 1
        return False
    finally:
        try:
            if cdp: cdp.close()
        except Exception: pass
        try:
            if proc: close_chrome(proc, cdp)
        except Exception: pass


def main():
    jobs = [(u, "train") for u in TRAIN for _ in range(LOADS)]
    jobs += [(u, "holdout") for u in HOLDOUT for _ in range(LOADS)]
    print(f"  {len(jobs)} 页（训练 {len(TRAIN)} 站 / 留出 {len(HOLDOUT)} 站，各 {LOADS} 次）", flush=True)
    pending = list(jobs)

    def worker(ep, port):
        while True:
            with lock:
                if not pending: return
                url, split = pending.pop(0)
            ok = one(url, split, ep, port)
            with lock:
                n = done["train"] + done["holdout"]
                if n % 10 == 0 or not ok:
                    print(f"    [{n:3d}/{len(jobs)}] {'✓' if ok else '✗'} {split:<7s} "
                          f"{url.split('//')[1][:30]}", flush=True)

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(TEACHERS[i], PORT_BASE + i), daemon=True)
           for i in range(2)]
    for t in ths: t.start()
    for t in ths: t.join()
    print(f"\n  完成: 训练 {done['train']} / 留出 {done['holdout']} / 失败 {done['fail']}"
          f"，耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
