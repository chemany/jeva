"""采集「页面文本块 → 第一篇文章」的训练数据。

确定性部分只做通用枚举（不做任何站点专属过滤）；老师是本机 27B，只在离线打标签时用一次。
"""
import json, os, sys, time, urllib.request
sys.path.insert(0, "/root/code/jeva")
from jeva.browser import launch_chrome, close_chrome, CDP, Browser

TEACHER = "http://127.0.0.1:8000/v1"
PAGES = [
    ("https://sports.sina.com.cn/", "sina-sports"),
    ("https://finance.sina.com.cn/", "sina-finance"),
    ("https://news.163.com/", "163-news"),
    ("https://www.thepaper.cn/", "thepaper"),
    ("https://www.chinanews.com.cn/", "chinanews"),
    ("https://www.ifeng.com/", "ifeng"),
    ("https://www.sohu.com/", "sohu"),
    ("https://news.sina.com.cn/", "sina-news"),
]


def blocks(browser, limit=30):
    """通用候选：所有可见链接/标题块，按原始阅读顺序（y, x）。不做字号或域名筛选。"""
    items = browser.all_blocks(limit)
    return sorted(items, key=lambda c: (c["y"], c["x"]))


def label(cands, url, tries=2):
    """老师标注：哪些块是正文文章。max_tokens 必须够大，推理与答案共享预算。"""
    listing = "\n".join(
        f'[{i}] {c["size"]:.0f}px col{c["column"]} y={c["y"]} host={c.get("host","")} :: {c["t"]}'
        for i, c in enumerate(cands, 1))
    prompt = ("Mark which blocks are news article links. Exclude navigation, section tabs, "
              "hot-search lists, advertisements and promotions.\n\n"
              f"Page: {url}\nBlocks:\n{listing}\n\n"
              'Reply with JSON only: {"content": [indices]}')
    for _ in range(tries):
        body = json.dumps({"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 8000,
                           "reasoning_effort": "low",
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        r = urllib.request.Request(TEACHER + "/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(r, timeout=600).read())["choices"][0]["message"]["content"] or ""
        s = out[out.find("{"):out.rfind("}")+1]
        try:
            got = json.loads(s).get("content")
            if isinstance(got, list):
                return [int(i) for i in got if str(i).isdigit() or isinstance(i, int)]
        except Exception:
            pass
        time.sleep(2)
    return None


def main():
    out_path = "/root/code/jeva-content/data.jsonl"
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try: done.add(json.loads(line)["page"])
            except Exception: pass
    recs = []
    port = 9440
    for url, name in PAGES:
        if name in done:
            print(f"  跳过 {name}（已采）"); continue
        proc, ws = launch_chrome(port=port, profile=f"/tmp/ca{port}")
        cdp = CDP(ws)
        try:
            b = Browser(cdp, url, width=1400, height=1000)
            time.sleep(3.5)
            b.dismiss_overlays(); time.sleep(0.6)
            cands = blocks(b)
            if len(cands) < 5:
                print(f"  ✗ {name}: 候选太少（{len(cands)}）"); port += 1; continue
            t0 = time.time()
            idx = label(cands, url)
            if not idx:
                print(f"  ✗ {name}: 老师没给出标注"); port += 1; continue
            first = min(i for i in idx if 1 <= i <= len(cands))
            rec = {"page": name, "url": url, "title": b.evaluate("document.title"),
                   "cands": cands, "content": idx, "first": first,
                   "target_text": cands[first-1]["t"], "labeled_ms": int((time.time()-t0)*1000)}
            recs.append(rec)
            print(f"  ✓ {name:<14s} {len(cands):2d} 候选 → 正文 {len(idx)} 个，第一条 [{first}] "
                  f"{cands[first-1]['t'][:40]}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__} {str(e)[:70]}")
        finally:
            try: cdp.close()
            except Exception: pass
            close_chrome(proc, cdp)
        port += 1
    with open(out_path, "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  本次写入 {len(recs)} 条 → {out_path}")


if __name__ == "__main__":
    main()
