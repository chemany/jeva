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

from browser import (CDP, Browser, _normalize_for_input, action_space,   # noqa: E402
                     launch_chrome, to_page)

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
    # 同一个概念在三个站点皮肤里写法不同，必须合成一个键；拆成两个键会因字典后写覆盖前写
    # 而丢掉另一批标签（site2 用 "Your name"、site3 用 "Email address"）。
    "name": {"Full name", "Name", "Your name", "Your name:", "Full name:", "Customer name:"},
    "email": {"Email", "Email address", "E-mail", "E-mail address:", "Email:"},
    "msg": {"Message", "Your message", "Details", "Enquiry", "Comments"},
    "agree": {"I agree to the privacy policy", "I accept the privacy policy", "Agree to terms"},
    "send": {"Send", "Submit", "Send message"},
    "q": {"Search", "Search Wikipedia", "Query"},
    "note": {"Note for the kitchen", "Note", "Comments"},
    "when": {"Preferred delivery time:", "Delivery time", "Time"},
    "phone": {"Phone number:", "Phone", "Telephone:", "Telephone"},
    "trip_round": {"Round trip", "Return", "Round-trip"},
    "trip_oneway": {"One way", "One-way", "Single"},
}

CABINS = ["Economy", "Premium economy", "Business", "First"]
PAX = ["1 adult", "2 adults", "3 adults"]
MAXPRICE = ["$50", "$100", "$200", "$500"]
SORTS = ["Relevance", "Price low to high", "Rating"]
CITIES = ["Zurich", "Paris", "Berlin", "Tokyo", "Osaka", "London", "New York"]
# (value, label, goal_word)：label 与 goal_word 故意不同，制造「目标用词 ≠ 控件标签」的情况
TOPPINGS = [("cheese", "Extra Cheese", "cheese"),
            ("mushroom", "Mushroom", "mushrooms"),
            ("olives", "Black Olives", "olives"),
            ("pepperoni", "Pepperoni", "pepperoni"),
            ("onion", "Onion", "onions")]
FLAGS = [("nonstop", "Nonstop delivery only", "direct delivery"),
         ("refundable", "Free cancellation", "a refundable booking"),
         ("gift", "Gift wrapping", "gift wrapping")]
NOTES = [None, "Ring the bell", "Leave at the door", "Call on arrival", "Extra napkins please"]
# 同一要求的不同措辞 + 不同从句位置，避免模型只学会「读目标最后一句」
# 时间要求：含 24 小时与 AM/PM 两种写法，压一压浏览器的原生校验
TIMES = ["12:30", "6:15 PM", "19:45", "9:00 AM", "13:05", "8:30 pm"]
# (插在中间的写法, 另起一句的写法)，两种都要自然通顺，模型读到的是自然语言
TIME_CLAUSES = [("for delivery at {t}", "Delivery is at {t}."),
                ("with delivery no earlier than {t}", "Delivery must be no earlier than {t}."),
                ("for delivery around {t}", "Delivery should be around {t}.")]

PEOPLE = ["Jason Zhang", "Li Wei", "Maria Lopez", "Chen Yu"]
PHONES = ["555-0100", "+41 44 555 01 23", "13800138000", "020 7946 0958"]
EMAILS = ["jason@example.com", "li.wei@example.org", "maria@example.net"]
# 真实用户常把多个字段写成裸列表（"Name X, phone Y, email Z"），而不是逐个动词交代
BARE_ORDER = ["Name {name}, phone {phone}, email {email}.",
              "Contact {name}, {phone}, {email}.",
              "Reach {name} at {phone} or {email}."]
VERBOSE_ORDER = ["Use the name {name}, the phone number {phone} and the email {email}."]

NOTE_PHRASES = ['Add the note "{n}".', 'Include a note saying "{n}".',
                'Put "{n}" in the notes.', 'Leave a note that says "{n}".']

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


def gen_multi(rng):
    """多要求目标 + 标签≠目标用词 + 提交后跳到无可交互元素的页面。

    三个缺口一次覆盖：
      1. 目标要求勾选 2-3 个复选框（不是单个），且部分已预勾选（测试“别重复点”）
      2. 目标用词与控件标签不同（目标说 cheese，标签是 "Extra Cheese"；目标说
         "direct delivery"，标签是 "Nonstop delivery only"）
      3. 结果页 multi-results.html 没有任何可交互元素 → 必须直接判 DONE
    """
    want = rng.sample(TOPPINGS, rng.choice([2, 2, 3]))
    flag = rng.choice(FLAGS) if rng.random() < 0.6 else None
    note = rng.choice(NOTES)
    when = rng.choice(TIMES) if rng.random() < 0.55 else None
    contact = None
    if rng.random() < 0.45:
        contact = {"name": rng.choice(PEOPLE), "phone": rng.choice(PHONES),
                   "email": rng.choice(EMAILS)}
    words = [t[2] for t in want] + ([flag[2]] if flag else [])
    items = _human_join(words)
    if when and rng.random() < 0.5:
        mid, _tail = rng.choice(TIME_CLAUSES)
        order = f"Order a pizza with {items}, {mid.format(t=when)}, then place the order."
    else:
        order = f"Order a pizza with {items}, then place the order."
        if when:
            order += " " + rng.choice(TIME_CLAUSES)[1].format(t=when)
    goal = order
    if contact:
        style = rng.choice(BARE_ORDER + BARE_ORDER + VERBOSE_ORDER)
        goal += " " + style.format(**contact)
    if note:
        phrase = rng.choice(NOTE_PHRASES).format(n=note)
        goal = f"{goal} {phrase}" if rng.random() < 0.5 else f"{phrase} {goal}"

    pre = []
    for _v, label, _w in want:                      # 有时已经勾好，有时勾了不该勾的
        if rng.random() < 0.35:
            pre.append(_v)
    if rng.random() < 0.2:
        wrong = rng.choice([t for t in TOPPINGS if t not in want])
        pre.append(wrong[0])
    when_pre = when if when and rng.random() < 0.3 else None      # 有时已经填好了
    contact_pre = [k for k in (contact or {}) if rng.random() < 0.25]
    url = f"{SITE}/multi.html" + ("?" + "&".join(f"pre={v}" for v in pre) if pre else "")
    if when_pre:
        url += ("&" if "?" in url else "?") + "when=" + when_pre.replace(" ", "+")
    for k in contact_pre:
        url += "&" + k + "=" + str(contact[k]).replace(" ", "+")
    if note and rng.random() < 0.35:            # 多数情况下 note 是空的，必须真的去填
        url += ("&" if "?" in url else "?") + "note=" + note.replace(" ", "+")
    spec = {"kind": "multi",
            "want": [(label, v) for v, label, _w in want],
            "flag": (flag[1], flag[0]) if flag else None,
            "when": when,
            "contact": contact,
            "note": note}
    return url, goal, spec


def _human_join(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def gen_blocked(rng):
    return f"{SITE}/blocked.html", "Export the full catalogue to CSV.", {"kind": "blocked"}


GENS = [gen_flight, gen_flight, gen_hotel, gen_hotel, gen_contact, gen_wiki, gen_blocked,
        gen_multi, gen_multi, gen_multi, gen_multi]


# ── oracle：从目标 + 当前 DOM 推出正确动作（同时产出"已满足/未完成"提示）──
SUCCESS_URL = ("flight-results", "hotel-results", "multi-results", "thanks", "article")


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

    if kind == "multi":
        checks = []
        for label, _value in spec["want"]:
            checks.append((label, label,
                           lambda e: e is not None and e.get("checked") == "true",
                           (lambda lb: lambda: ("CLICK", _key_by_label(elements, targets, lb), None))(label)))
        if spec["flag"]:
            label = spec["flag"][0]
            checks.append((label, label,
                           lambda e: e is not None and e.get("checked") == "true",
                           (lambda lb: lambda: ("CLICK", _key_by_label(elements, targets, lb), None))(label)))
        for key, field in (("name", "name"), ("phone", "phone"), ("email", "email")):
            want = (spec.get("contact") or {}).get(key)
            if want:
                checks.append((key, f"contact:{key}",
                               (lambda w: lambda e: e is not None and _n(e.get("value")) == _n(w))(want),
                               (lambda w, f: lambda: ("TYPE_TEXT",
                                                      _key_of(elements, targets, "TYPE_TEXT", f),
                                                      w))(want, field)))
        if spec["when"]:
            checks.append(("when", "Time",
                           lambda e: e is not None and _tval(e.get("value")) == _tval(spec["when"]),
                           lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "when"),
                                    spec["when"])))
        if spec["note"]:
            checks.append(("note", "Note",
                           lambda e: e is not None and (e.get("value") or "").strip() == spec["note"],
                           lambda: ("TYPE_TEXT", _key_of(elements, targets, "TYPE_TEXT", "note"), spec["note"])))
        action = _resolve(checks, elements, sat, miss)
        if action:
            return action[0], action[1], action[2], None, sat, miss
        if any(u in url for u in SUCCESS_URL) or not elements:
            return "DONE", None, None, None, sat, miss
        return "CLICK", _key_by_label(elements, targets, "Place order"), None, None, sat, miss

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


def _n(s):
    """比较文本时的宽松归一：忽略大小写、空格与标点。"""
    return "".join(c for c in str(s).lower() if c.isalnum())


def _tval(x):
    """把时间写法折成浏览器最终会存下的样子（6:15 PM 与 18:15 视为同一个要求）。"""
    return _normalize_for_input(str(x)) if x else None


def _key_by_label(elements, targets, label):
    """按控件标签（而非 ALIAS 键）取 CLICK 目标键。"""
    for e in elements:
        if lab_of(e) == label:
            k = str(e["index"])
            if k in (targets.get("CLICK") or {}):
                return k
    return None


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
    if spec["kind"] == "multi":
        if "multi-results" not in url:
            return False
        got = set(q.get("extras", []))
        if not all(v in got for _l, v in spec["want"]):
            return False
        if spec["flag"] and spec["flag"][1] not in q.get("flags", []):
            return False
        if spec["when"] and _tval(q.get("when", [""])[0]) != _tval(spec["when"]):
            return False
        for key in ("name", "phone", "email"):
            want = (spec.get("contact") or {}).get(key)
            if want and n(want) not in n(q.get(key, [""])[0]):
                return False
        if spec["note"] and n(spec["note"]) not in n(q.get("note", [""])[0]):
            return False
        return True

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
