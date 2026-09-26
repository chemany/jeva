"""Collect genuinely distinct pages: crawl a site's own section links instead of reloading its home.

The earlier rounds reloaded each homepage N times, assuming the content rotates. It does not -- news
homepages are static for hours. Measured afterwards: 150 collected pages from 17 URLs contained only
17 distinct contents, and the "56 page" holdout was 8 distinct samples. Every accuracy number from
those rounds was computed on a handful of samples.

This walks the homepage's own internal links and collects each one, so pages differ in layout and
not just in timestamp.
"""
from __future__ import annotations
import json, sys, threading, time, urllib.parse, urllib.request
sys.path.insert(0, "/root/code/jeva")
from jeva.browser import CDP, Browser, close_chrome, launch_chrome       # noqa: E402
from collect4 import label                                              # noqa: E402

TRAIN = [
    "https://news.sina.com.cn/", "https://sports.sina.com.cn/", "https://finance.sina.com.cn/",
    "https://ent.sina.com.cn/", "https://tech.sina.com.cn/", "https://news.163.com/",
    "https://www.thepaper.cn/", "https://www.sohu.com", "https://www.people.com.cn/",
    "https://www.chinanews.com.cn/", "https://www.yicai.com/", "https://www.jiemian.com/",
    "https://www.stcn.com/", "https://www.cls.cn/", "https://www.eastmoney.com/",
    "https://www.hexun.com/", "https://www.gov.cn/", "https://www.cctv.com/",
    "https://www.youth.cn/", "https://www.cnstock.com/", "https://www.bjnews.com.cn/",
    "https://www.gmw.cn/", "https://www.cnr.cn/", "https://www.ce.cn/",
    "https://www.china.com.cn/", "https://www.10jqka.com.cn/", "https://www.qstheory.cn/",
    "https://www.workercn.cn/", "https://www.stdaily.com/", "https://www.cssn.cn/",
]
HOLDOUT = [
    "https://www.xinhuanet.com/", "https://news.ifeng.com/", "https://www.guancha.cn/",
    "https://www.21jingji.com/", "https://www.ynet.com/", "https://www.jschina.com.cn/",
    "https://www.qianlong.com/", "https://www.southcn.com/", "https://www.gxnews.com.cn/",
    "https://www.chinadaily.com.cn/",
]
PER_SITE = 6                          # homepage + 5 section pages
OUT = "/root/code/jeva-content/raw8.jsonl"
TEACHERS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8002/v1"]
lock = threading.Lock(); stat = {"ok": 0, "fail": 0}


def section_links(browser, base, want=5):
    """Internal links that look like sections: same registrable host, no query, not a deep article."""
    try:
        return browser.evaluate("""((base) => {
          const host = new URL(base).hostname.split('.').slice(-3).join('.');
          const out = [], seen = new Set();
          for (const a of document.querySelectorAll('a[href]')) {
            const u = new URL(a.href, base);
            if (!u.hostname.endsWith(host)) continue;
            if (u.search || u.hash) continue;
            const p = u.pathname.replace(/\\/+$/, '');
            if (p === '' || p === '/') continue;
            const seg = p.split('/').filter(Boolean);
            if (seg.length > 2) continue;            // deeper than a section index
            if (/\\d{4}|\\d{6,}|\\.s?html?$/i.test(p)) continue;
            const key = u.origin + u.pathname;
            if (seen.has(key)) continue;
            seen.add(key); out.push(key);
            if (out.length >= 40) break;
          }
          return JSON.stringify(out);
        })(""" + json.dumps(base) + ")")
    except Exception:                                                     # noqa: BLE001
        return "[]"


def collect_one(url, split, endpoint, port):
    proc = cdp = None
    try:
        proc, ws = launch_chrome(port=port, profile=f"/tmp/c8-{port}")
        cdp = CDP(ws)
        b = Browser(cdp, url, width=1400, height=1000)
        time.sleep(3.5); b.dismiss_overlays(); time.sleep(0.5)
        cands = b.all_blocks(30)
        if len(cands) < 8:
            raise RuntimeError("候选太少")
        idx = label(cands, url, endpoint)
        if not idx:
            raise RuntimeError("未标注")
        rec = {"split": split, "site": url.split("//")[1].split("/")[0], "url": url,
               "title": b.evaluate("document.title"), "cands": cands, "content": idx, "first": min(idx)}
        with lock:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stat["ok"] += 1
        return rec
    except Exception:                                                     # noqa: BLE001
        with lock:
            stat["fail"] += 1
        return None
    finally:
        try:
            if cdp: cdp.close()
        except Exception: pass
        try:
            if proc: close_chrome(proc, cdp)
        except Exception: pass


def one_site(base, split, endpoint, port):
    home = collect_one(base, split, endpoint, port)
    if not home:
        return
    # Reuse the same browser for the section pages of this site.
    proc = cdp = None
    try:
        proc, ws = launch_chrome(port=port + 100, profile=f"/tmp/c8b-{port}")
        cdp = CDP(ws)
        b = Browser(cdp, base, width=1400, height=1000)
        time.sleep(2.5)
        links = json.loads(section_links(b, base))[:PER_SITE - 1]
    except Exception:                                                     # noqa: BLE001
        links = []
    finally:
        try:
            if cdp: cdp.close()
        except Exception: pass
        try:
            if proc: close_chrome(proc, cdp)
        except Exception: pass
    for link in links:
        collect_one(link, split, endpoint, port)


def main():
    jobs = [u for u in TRAIN] + [u for u in HOLDOUT]
    splits = {**{u: "train" for u in TRAIN}, **{u: "holdout" for u in HOLDOUT}}
    print(f"  {len(jobs)} 站 × 每站最多 {PER_SITE} 个不同页面", flush=True)
    pending = list(jobs)

    def worker(ep, port):
        while True:
            with lock:
                if not pending: return
                base = pending.pop(0)
            one_site(base, splits[base], ep, port)
            with lock:
                print(f"    [{stat['ok']+stat['fail']:3d} 页] 成功 {stat['ok']} 失败 {stat['fail']}",
                      flush=True)

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(TEACHERS[i], 9560 + i * 200), daemon=True) for i in range(2)]
    for t in ths: t.start()
    for t in ths: t.join()
    print(f"\n  完成 {stat['ok']} 页（失败 {stat['fail']}），{(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
