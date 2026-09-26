"""Wide corpus, site-level split, long-nav portals included.

v8 failed for a diagnosable reason: every training site was a portal homepage with at most 11
chrome blocks before the article, while the holdout's hardest page had 18 -- and those 18 were
phrased like headlines ("学习进行时", "思客智库", "数字经济"). The model had never seen that shape.
This corpus adds government and agency portals, which is where that shape lives, and moves the
holdout to ten sites that appear nowhere in training.
"""
from __future__ import annotations
import json, os, sys, threading, time, urllib.request
sys.path.insert(0, "/root/code/jeva")
from jeva.browser import CDP, Browser, close_chrome, launch_chrome       # noqa: E402

TEACHERS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8002/v1"]
TRAIN = [
    "https://sports.sina.com.cn/", "https://finance.sina.com.cn/", "https://news.sina.com.cn/",
    "https://ent.sina.com.cn/", "https://tech.sina.com.cn/", "https://mil.news.sina.com.cn/",
    "https://auto.sina.com.cn/", "https://house.sina.com.cn/",
    "https://news.163.com/", "https://www.sohu.com", "https://www.thepaper.cn/",
    "https://www.people.com.cn/", "https://www.chinanews.com.cn/", "https://www.china.com.cn/",
    "https://www.yicai.com/", "https://www.jiemian.com/", "https://www.stcn.com/",
    "https://www.cls.cn/", "https://www.eastmoney.com/", "https://www.hexun.com/",
    "https://www.gov.cn/", "https://www.cctv.com/", "https://www.youth.cn/",
    "https://www.cnstock.com/", "https://www.ce.cn/", "https://www.bjnews.com.cn/",
    "https://www.10jqka.com.cn/", "https://www.chinadaily.com.cn/", "https://finance.eastmoney.com/",
    "https://www.chinanews.com/",
]
HOLDOUT = [
    "https://www.xinhuanet.com/", "https://news.ifeng.com/", "https://www.guancha.cn/",
    "https://www.21jingji.com/", "https://www.ynet.com/", "https://www.gxnews.com.cn/",
    "https://www.jschina.com.cn/", "https://www.qianlong.com/", "https://www.southcn.com/",
    "https://www.chinanews.com.cn/cj/",
]
LOADS = 8
PORT_BASE = 9480
OUT = "/root/code/jeva-content/raw4.jsonl"
lock = threading.Lock()
stat = {"train": 0, "holdout": 0, "fail": 0}


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
        proc, ws = launch_chrome(port=port, profile=f"/tmp/c4-{port}")
        cdp = CDP(ws)
        b = Browser(cdp, url, width=1400, height=1000)
        time.sleep(3.5); b.dismiss_overlays(); time.sleep(0.5)
        cands = b.all_blocks(30)
        if len(cands) < 8:
            raise RuntimeError(f"候选仅 {len(cands)}")
        idx = label(cands, url, endpoint)
        if not idx:
            raise RuntimeError("老师未标注")
        rec = {"split": split, "site": url.split("//")[1].split("/")[0], "url": url,
               "title": b.evaluate("document.title"), "cands": cands, "content": idx, "first": min(idx)}
        with lock:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stat[split] += 1
        return True
    except Exception:                                                     # noqa: BLE001
        with lock:
            stat["fail"] += 1
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
                n = stat["train"] + stat["holdout"] + stat["fail"]
                if n % 25 == 0:
                    print(f"    [{n:3d}/{len(jobs)}] 训练 {stat['train']} / 留出 {stat['holdout']} "
                          f"/ 失败 {stat['fail']}", flush=True)

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(TEACHERS[i], PORT_BASE + i), daemon=True) for i in range(2)]
    for t in ths: t.start()
    for t in ths: t.join()
    print(f"\n  完成: 训练 {stat['train']} / 留出 {stat['holdout']} / 失败 {stat['fail']}"
          f"，耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
