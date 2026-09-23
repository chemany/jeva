#!/usr/bin/env python3
"""蒸馏数据采集：用确定性求解器（oracle）驱动真实浏览器，产出 SFT 语料。

为什么不用 LLM 当老师：我已有确定性策略（原 gen_dataset.py 那套规则）。让它在**真实 Chrome**
上操作，就能得到「真实 DOM 快照 + 保证正确的动作」，零 LLM 调用、零人工标注。

关键设计：
  1. **老师看到额外提示**（"已满足/未完成"，由 oracle 从目标+DOM 算出），
     **但录制下来的监督只有 (state, action)** —— 学生要自己学会状态追踪。
     这是标准的带特权信息蒸馏。
  2. 提示词用 llm_backend.build_prompt 生成，**与推理时逐字一致**，避免训练/部署分布漂移。
  3. 轨迹成功后整条保留；失败轨迹丢弃（环境即验证器）。
"""
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # repo root -> jeva package
sys.path.insert(0, _HERE)                           # evals/ -> browser.py

from browser import CDP, Browser, action_space, launch_chrome, to_page   # noqa: E402

SITE = os.environ.get("SITE", "http://127.0.0.1:8899")
# 多站点：每个任务随机选一个站点根，逼模型学“规则”而不是记住某套标签
SITES = [x.strip() for x in os.environ.get("SITES", "").split(",") if x.strip()] or [SITE]
MAX_STEPS = 10

# ── 字段别名：让 oracle 能认出标签变体（同时给学生词汇泛化）───────────
ALIAS = {
    "from": {"Where from?", "Origin", "From", "Departing from", "Leaving from"},
    "to": {"Where to?", "Destination", "To", "Going to", "Arriving at"},
    "dep": {"Departure", "Depart date", "When", "Travel date", "Leaving on"},
    "cabin": {"Cabin", "Cabin class", "Travel class", "Seat class"},
    "pax": {"Passengers", "Travellers", "Guests", "Pax"},
    "search": {"Search", "Find", "Go", "Submit", "Search flights"},
    "city": {"City", "Where?", "Destination", "Location"},
    "max": {"Max price per night", "Max price", "Price limit", "Budget"},
    "sort": {"Sort by", "Sort", "Order by"},
    "cancel": {"Free cancellation only", "Free cancellation", "Cancellable only"},
    "search_hotels": {"Search hotels", "Find hotels", "Search"},
    "name": {"Full name", "Name", "Your name"},
    "email": {"Email", "Email address", "E-mail"},
    "msg": {"Message", "Your message", "Details", "Enquiry", "Comments"},
    "agree": {"I agree to the privacy policy", "I accept the privacy policy", "Agree to terms"},
    "send": {"Send", "Submit", "Send message"},
    "q": {"Search", "Search Wikipedia", "Query"},
    "trip_round": {"Round trip", "Return", "Round-trip"},
    "trip_oneway": {"One way", "One-way", "Single"},
}

CABINS = ["Economy", "Premium economy", "Business", "First"]
PAX = ["1 adult", "2 adults", "3 adults"]
MAXPRICE = ["$50", "$100", "$200", "$500"]
SORTS = ["Relevance", "Price low to high", "Rating"]
CITIES = ["Zurich", "Paris", "Berlin", "Tokyo", "Osaka", "London", "New York"]
TOPICS = ["Photosynthesis", "Photosystem II", "Cellular respiration", "Chlorophyll", "Photoperiodism"]


def lab_of(el):
    return (el.get("label") or "").strip()


def find(elements, field, ops=None):
    """按别名集找元素；ops 过滤可执行的操作。"""
    names = ALIAS.get(field, {field})
    for e in elements:
        if lab_of(e) in names and (ops is None or any(o in e["operations"] for o in ops)):
            return e
    return None


# ── 任务实例生成 ────────────────────────────────────────────────────
def gen_flight(rng):
    trip = rng.choice(["round", "oneway"])
    frm, to = rng.sample(CITIES, 2)
    dep = rng.choice(["September 20, 2026", "October 5, 2026", "March 3, 2026"])
    cabin = rng.choice(CABINS)
    pax = rng.choice(PAX)
    # 随机初始状态（有些字段预填、有的预填错值）
    init = {"trip": rng.choice(["round", "oneway"]), "cabin": rng.choice(CABINS), "pax": rng.choice(PAX)}
    for k, v in (("from", frm), ("to", to), ("dep", dep)):
        if rng.random() < 0.45:
            init[k] = rng.choice([v, rng.choice(CITIES)])      # 可能预填成错值
    url = f"{SITE}/flights.html?" + "&".join(f"{k}={v.replace(' ', '+')}" for k, v in init.items())
    goal = (f"Find {trip} flights from {frm} to {to} on {dep}, "
            f"for {pax} in {cabin}, then submit the search.")
    spec = {"kind": "flight", "trip": trip, "from": frm, "to": to, "dep": dep, "cabin": cabin, "pax": pax}
    return url, goal, spec


def gen_hotel(rng):
    city = rng.choice(CITIES)
    mx = rng.choice(MAXPRICE)
    sort = rng.choice(SORTS[1:])          # 故意用非默认排序
    cancel = True
    init = {"city": rng.choice(["", city]), "max": rng.choice(MAXPRICE),
            "sort": rng.choice(SORTS), "cancel": rng.choice(["0", "1"])}
    url = f"{SITE}/hotels.html?" + "&".join(f"{k}={v.replace(' ', '+').replace('$','%24')}"
                                            for k, v in init.items() if v != "")
    goal = (f"Find hotels in {city} with free cancellation only, at most {mx} per night, "
            f"sorted by {sort.lower()}, then submit the search.")
    spec = {"kind": "hotel", "city": city, "max": mx, "sort": sort, "cancel": cancel}
    return url, goal, spec


def gen_contact(rng):
    name = rng.choice(["Jason Zhang", "Li Wei", "Maria Lopez", "Chen Yu"])
    email = rng.choice(["jason@example.com", "li.wei@example.org", "maria@example.net"])
    msg = rng.choice(["Please send a quotation for your services.",
                      "We would like pricing and lead times.",
                      "Could you send the technical datasheet?"])
    init = {}
    for k, v in (("name", name), ("email", email), ("msg", msg)):
        if rng.random() < 0.4:
            init[k] = v
    if rng.random() < 0.4:
        init["agree"] = "1"
    url = f"{SITE}/contact.html?" + "&".join(f"{k}={v.replace(' ', '+')}" for k, v in init.items())
    goal = (f"Send the contact form from {email} with the name {name} and the message "
            f"'{msg}' then accept the privacy policy and submit.")
    spec = {"kind": "contact", "name": name, "email": email, "msg": msg}
    return url, goal, spec


def gen_wiki(rng):
    topic = rng.choice(TOPICS)
    init = {"q": rng.choice(["", topic[:-1]])} if rng.random() < 0.5 else {"q": ""}
    url = f"{SITE}/wiki.html?" + "&".join(f"{k}={v.replace(' ', '+')}" for k, v in init.items())
    goal = f"Find and open the Wikipedia article about {topic}."
    spec = {"kind": "wiki", "topic": topic}
    return url, goal, spec


def gen_blocked(rng):
    return f"{SITE}/blocked.html", "Export the full catalogue to CSV.", {"kind": "blocked"}


GENS = [gen_flight, gen_flight, gen_hotel, gen_hotel, gen_contact, gen_wiki, gen_blocked]


# ── oracle：从目标 + 当前 DOM 推出正确动作（同时产出"已满足/未完成"提示）──
SUCCESS_URL = ("flight-results", "hotel-results", "thanks", "article")


def oracle(elements, targets, spec, url=""):
    kind = spec["kind"]
    # 已在成功页 → DONE（整条轨迹另有 verify() 把关，判错会被丢弃）
    if any(u in url for u in SUCCESS_URL):
        return "DONE", None, None, None, ["the goal is visibly satisfied"], []
    sat, miss = [], []

    if kind == "blocked":
        return "BLOCKED", None, None, ["a human-verification challenge blocks progress"], sat, miss

    if kind == "flight":
        checks = [
            ("trip_round" if spec["trip"] == "round" else "trip_oneway", "Trip type",
             lambda e: e is not None and e.get("checked") == "true",
             lambda: ("CLICK", _click_key(targets, "CLICK", None, ALIAS, spec["trip"]), None)),
            ("from", "Origin", lambda e: (e.get("value") or "").strip() == spec["from"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "from"), spec["from"])),
            ("to", "Destination", lambda e: (e.get("value") or "").strip() == spec["to"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "to"), spec["to"])),
            ("dep", "Departure", lambda e: (e.get("value") or "").strip() == spec["dep"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "dep"), spec["dep"])),
            ("cabin", "Cabin", lambda e: (e.get("value") or "").strip() == spec["cabin"],
             lambda: ("SELECT", _opt_key(elements, targets, "cabin", spec["cabin"]), None)),
            ("pax", "Passengers", lambda e: (e.get("value") or "").strip() == spec["pax"],
             lambda: ("SELECT", _opt_key(elements, targets, "pax", spec["pax"]), None)),
        ]
        action = _resolve(checks, elements, sat, miss)
        if action:
            return action[0], action[1], action[2], None, sat, miss
        if any(u in url for u in SUCCESS_URL):
            return "DONE", None, None, None, sat, miss
        return "CLICK", _key_of(elements, targets, "CLICK", "search"), None, None, sat, miss

    if kind == "hotel":
        checks = [
            ("city", "City", lambda e: (e.get("value") or "").strip() == spec["city"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "city"), spec["city"])),
            ("max", "Max price", lambda e: (e.get("value") or "").strip() == spec["max"],
             lambda: ("SELECT", _opt_key(elements, targets, "max", spec["max"]), None)),
            ("cancel", "Free cancellation", lambda e: e.get("checked") == "true",
             lambda: ("CLICK", _key_of(elements, targets, "CLICK", "cancel"), None)),
            ("sort", "Sort order", lambda e: (e.get("value") or "").strip() == spec["sort"],
             lambda: ("SELECT", _opt_key(elements, targets, "sort", spec["sort"]), None)),
        ]
        action = _resolve(checks, elements, sat, miss)
        if action:
            return action[0], action[1], action[2], None, sat, miss
        if any(u in url for u in SUCCESS_URL):
            return "DONE", None, None, None, sat, miss
        return "CLICK", _key_of(elements, targets, "CLICK", "search_hotels"), None, None, sat, miss

    if kind == "contact":
        checks = [
            ("name", "Name", lambda e: (e.get("value") or "").strip() == spec["name"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "name"), spec["name"])),
            ("email", "Email", lambda e: (e.get("value") or "").strip() == spec["email"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "email"), spec["email"])),
            ("msg", "Message", lambda e: (e.get("value") or "").strip() == spec["msg"],
             lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "msg"), spec["msg"])),
            ("agree", "Privacy consent", lambda e: e.get("checked") == "true",
             lambda: ("CLICK", _key_of(elements, targets, "CLICK", "agree"), None)),
        ]
        action = _resolve(checks, elements, sat, miss)
        if action:
            return action[0], action[1], action[2], None, sat, miss
        if any(u in url for u in SUCCESS_URL):
            return "DONE", None, None, None, sat, miss
        return "CLICK", _key_of(elements, targets, "CLICK", "send"), None, None, sat, miss

    if kind == "wiki":
        e = find(elements, "q")
        topic = spec["topic"]
        if e and (e.get("value") or "").strip().lower() == topic.lower():
            sat.append("the search box already holds the requested text")
        else:
            miss.append("the search text has not been entered yet")
            return "TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "q"), topic, None, sat, miss
        # 选项列表里点同名项
        for el in elements:
            if el.get("role") == "option" and lab_of(el).lower() == topic.lower():
                return "CLICK", str(el["index"]), None, None, sat, miss
        miss.append("the matching suggestion is not visible yet")
        return "WAIT", None, None, None, sat, miss

    return "BLOCKED", None, None, ["unknown task"], sat, miss


def _resolve(checks, elements, sat, miss):
    for _f, label, pred, mk in checks:
        e = find(elements, _f)
        if e and pred(e):
            sat.append(label)
        else:
            miss.append(label)
    # 第一个未满足的 = 下一个动作
    for _f, label, pred, mk in checks:
        e = find(elements, _f)
        if not (e and pred(e)):
            return mk()
    return None


def _key_of(elements, targets, op, field):
    e = find(elements, field, ops=[op])
    if e is None:
        return None
    return str(e["index"]) if str(e["index"]) in targets.get(op, {}) else None


def _opt_key(elements, targets, field, want):
    e = find(elements, field)
    if e is None:
        return None
    for k, a in targets.get("SELECT", {}).items():
        if k.startswith(f"{e['index']}:") and want.lower() in (a.get("label") or "").lower():
            return k
    return None


def _click_key(targets, op, _x, _alias, trip):
    """按 ALIAS 里的票种别名找 radio（site1/2/3 的标签各不相同）。"""
    names = ALIAS["trip_oneway" if trip == "oneway" else "trip_round"]
    for k, a in targets.get("CLICK", {}).items():
        lab = (a.get("label") or "").lower()
        if any(nm.lower() in lab for nm in names):
            return k
    return None


# ── 轨迹回放 + 采集 ────────────────────────────────────────────────
def run_one(task, cdp, verbose=False):
    from jeva.prompt import build_prompt
    from jeva.render import render_state

    url, goal, spec = task
    b = Browser(cdp, url)
    recs, history = [], []
    status = "ready"
    try:
        page = b.observe(screenshot=False)
        for step in range(MAX_STEPS):
            actions = [a for a in page["actions"] if a.get("kind") != "wait"]
            elements, targets, controls = action_space(actions)
            jpage = to_page(page, elements)
            state = render_state(jpage, goal, history)

            op, key, text, note, sat, miss = oracle(elements, targets, spec, page["url"])
            if os.environ.get("COLLECT_DEBUG") == "1":
                print(f"      step{step}: {op} key={key} text={text!r} sat={sat} miss={miss}", flush=True)

            # 老师看得到“已满足/未完成”，学生看不到 —— 这就是特权信息
            hint = ("Already satisfied: " + ("; ".join(sat) if sat else "nothing") +
                    ". Not yet done: " + ("; ".join(miss) if miss else "nothing") + ".")
            prompt_hint = build_prompt(jpage, goal, history, state=state + "\n" + hint)
            prompt_plain = build_prompt(jpage, goal, history, state=state)

            if op in ("DONE", "BLOCKED"):
                recs.append({"prompt": prompt_plain, "prompt_with_hint": prompt_hint,
                             "state": state, "hint": hint, "operation": op,
                             "target": "", "text": "", "target_key": ""})
                status = "blocked" if op == "BLOCKED" else "done"
                break
            if key is None or op not in targets and op != "WAIT":
                status = "stuck"
                break
            action = targets[op][key] if op in targets else controls.get(op)
            if action is None:
                status = "stuck"
                break
            # 记录：state（不含 hint）→ 动作
            recs.append({"prompt": prompt_plain, "prompt_with_hint": prompt_hint,
                         "state": state, "hint": hint, "operation": op, "target": key,
                         "text": text or "", "target_key": key})
            b.act(action, page, text=text)
            history.append({"action": action["label"], "kind": action["kind"], "text": text})
            page = b.observe(screenshot=False)
        final = page["url"]
    except Exception as exc:                                       # noqa: BLE001
        status = f"error:{type(exc).__name__}"
        final = ""
        if os.environ.get("COLLECT_DEBUG") == "1":
            import traceback
            traceback.print_exc()
    finally:
        try:
            b.close()
        except Exception:                                          # noqa: BLE001
            pass
    ok = verify(spec, final, status)
    if verbose:
        print(f"    status={status} ok={ok} steps={len(recs)} final={final.split('/')[-1][:70]}")
    return ok, recs, status, final



def verify(spec, url, status):
    import urllib.parse
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    n = lambda s: "".join(c for c in str(s).lower() if c.isalnum())
    if spec["kind"] == "blocked":
        return status == "blocked" and "blocked.html" in url
    if status != "done" and "results" not in url and "thanks" not in url and "article" not in url:
        return False
    if spec["kind"] == "flight":
        return ("flight-results" in url and q.get("trip", [""])[0] == spec["trip"]
                and n(spec["from"]) in n(q.get("from", [""])[0]) and n(spec["to"]) in n(q.get("to", [""])[0])
                and q.get("cabin", [""])[0] == spec["cabin"])
    if spec["kind"] == "hotel":
        return ("hotel-results" in url and q.get("max", [""])[0] == spec["max"]
                and q.get("sort", [""])[0] == spec["sort"] and q.get("cancel", [""])[0] == "1"
                and n(spec["city"]) in n(q.get("city", [""])[0]))
    if spec["kind"] == "contact":
        return ("thanks" in url and n(spec["name"]) in n(q.get("name", [""])[0])
                and n(spec["email"]) in n(q.get("email", [""])[0]) and q.get("agree", [""])[0] == "1")
    if spec["kind"] == "wiki":
        return "article" in url and n(spec["topic"]) in n(q.get("topic", [""])[0])
    return False


def main():
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "data", "sft_raw.jsonl")
    seed = int(os.environ.get("SEED", "7"))
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    proc, ws = launch_chrome()
    cdp = CDP(ws)

    n_ok = n_ex = 0
    by_kind = {}
    t0 = time.time()
    try:
        with open(out, "w") as f:
            for i in range(n_tasks):
                globals()["SITE"] = rng.choice(SITES)
                task = rng.choice(GENS)(rng)
                ok, recs, status, final = run_one(task, cdp)
                by_kind.setdefault(task[2]["kind"], [0, 0])
                by_kind[task[2]["kind"]][1] += 1
                if ok and recs:
                    n_ok += 1
                    by_kind[task[2]["kind"]][0] += 1
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n_ex += 1
                if (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{n_tasks}] 成功轨迹 {n_ok} | 样本 {n_ex} | "
                          f"{time.time()-t0:.0f}s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:                                          # noqa: BLE001
            proc.kill()

    print("=" * 90)
    print(f"任务 {n_tasks} | 成功轨迹 {n_ok} ({n_ok/max(1,n_tasks)*100:.0f}%) | SFT 样本 {n_ex} | "
          f"耗时 {time.time()-t0:.0f}s")
    for k, (a, b) in sorted(by_kind.items()):
        print(f"  {k:<9s} 成功 {a}/{b}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
