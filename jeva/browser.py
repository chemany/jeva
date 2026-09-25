"""Browser driver for collecting and evaluating jeva trajectories.

Self-contained: a small Chrome DevTools Protocol client, a page observer that evaluates
``snapshot.js``, and an action executor that resolves a target index back to the DOM node.

The observation format is the one jeva is trained on: an indexed table of interactive elements
plus the visible page text. Only indices ever cross the model boundary — selectors, coordinates
and JavaScript stay in here.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

SNAPSHOT_JS = Path(__file__).with_name("snapshot.js").read_text()


# Re-running the snapshot before acting refreshes the node table that ``act`` resolves against,
# and returning only the marker keeps it cheap.
MARKER_JS = f"(() => {{ const state={SNAPSHOT_JS}; return state?.marker ?? null; }})()"


# A background tab has its animation frames throttled, and Page.captureScreenshot waits for a
# fresh frame to composite. Sleeping instead does not guarantee one exists, and the capture then
# blocks until the CDP timeout. Waiting for two frames explicitly -- with a hard bound, so a
# throttled tab cannot stall the loop -- is what keeps the screenshot cheap.
FRAME_JS = """new Promise(resolve => {
  let frames = 0, settled = false;
  const finish = () => { if (!settled) { settled = true; resolve(true); } };
  setTimeout(finish, 500);
  const step = () => {
    if (settled) return;
    if (++frames >= 2) finish(); else requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
})"""


# App-promo modals, cookie walls and ad overlays are an obstacle, not part of the goal, and on
# Chinese portals they cover exactly the content you came for. Deterministic on purpose: whether a
# box is a full-screen overlay is a geometry question, not a judgement call, and a click on the
# close affordance is what a person would do -- removing the node (the obvious shortcut) can break
# whatever state the page keeps around it.
DISMISS_OVERLAYS_JS = r"""(() => {
  const vw = innerWidth, vh = innerHeight, vArea = vw * vh;
  const closed = [];
  const isOverlay = e => {
    for (let n = e; n && n !== document.body; n = n.parentElement) {
      const s = getComputedStyle(n);
      if (!['fixed', 'absolute', 'sticky'].includes(s.position)) continue;
      const z = s.zIndex === 'auto' ? 0 : Number(s.zIndex);
      if (z < 100) continue;
      const r = n.getBoundingClientRect();
      // Two tests, both needed. Area alone catches sina's "关闭置顶" strip across the top, which is
      // a real page control: a modal is a box, a banner is a thin strip, so require real height too.
      // Under-dismissing is the safe direction here -- guessing wrong clicks a control nobody asked for.
      const areaOk = r.width * r.height >= vArea * 0.04;
      const boxOk = r.height >= vh * 0.15 && r.width >= vw * 0.15;
      if (areaOk && boxOk) return n;
    }
    return null;
  };
  // Start from the close affordance, not from the overlay: scanning big boxes misses the dialog
  // that a full-screen mask hides inside, and cannot tell a dialog from page furniture.
  const affordances = [...document.querySelectorAll('body *')].filter(c => {
    const r = c.getBoundingClientRect();
    if (!r.width || !r.height || r.width > 72 || r.height > 72) return false;
    if (c.children.length > 0 && !(c.innerText || '').trim()) return false;
    const name = [c.getAttribute('aria-label'), c.title, c.className, c.innerText]
      .map(v => String(v || '')).join(' ');
    return /close|dismiss|关闭|取消|×|✕|✖/i.test(name);
  });
  for (const c of affordances) {
    const box = isOverlay(c);
    if (!box) continue;                                  // "关闭置顶" in a header is not an overlay
    c.click();
    closed.push(((c.getAttribute('aria-label') || c.innerText || 'x') + '').trim().slice(0, 20));
    if (closed.length >= 3) break;
  }
  return JSON.stringify(closed);
})()"""


# "What is the first news item" is a question about typography and position, not about meaning:
# the headline is the largest text in the main column, and the ticker above it is set small on
# purpose. Ranking by font size against the page's own median beats asking a model, and it is the
# same rule a person applies when they glance at a page. Header, nav, footer and aside are skipped
# because they hold navigation, not content -- that is what put sina's rolling ticker ahead of the
# real headline when the DOM order was used instead.
PROMINENT_JS = r"""(() => {
  const vw = innerWidth, vh = innerHeight;
  const seen = new Set(), items = [];
  const skip = e => e.closest('header, nav, footer, aside, [role="navigation"], [role="banner"]') ||
                    /(^|[-_])(ad|ads|advert|banner|promo)([-_]|$)/i.test(String(e.className) + ' ' + String(e.id));
  for (const e of document.querySelectorAll('a, h1, h2, h3, h4')) {
    if (skip(e)) continue;
    const r = e.getBoundingClientRect();
    if (!r.width || !r.height || r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) continue;
    if (e.checkVisibility && !e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) continue;
    const t = (e.innerText || '').trim().replace(/\s+/g, ' ');
    if (t.length < 8 || t.length > 90 || seen.has(t)) continue;
    seen.add(t);
    const cs = getComputedStyle(e);
    // Headlines are often an <h2> wrapping an <a>; e.href is undefined on the heading itself.
    const inner = e.querySelector ? e.querySelector('a[href]') : null;
    const url = e.href || (inner && inner.href) || '';
    let host = ''; try { host = url ? new URL(url).hostname : ''; } catch (_) {}
    items.push({t, href: url, host, size: parseFloat(cs.fontSize) || 0,
                weight: Number(cs.fontWeight) || 400,
                x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width)});
  }
  if (!items.length) return '[]';
  const sizes = items.map(i => i.size).sort((a, b) => a - b);
  const median = sizes[Math.floor(sizes.length / 2)] || 12;
  const big = items.filter(i => i.size >= median * 1.25);
  const pool = big.length >= 3 ? big : items;
  pool.sort((a, b) => a.y - b.y || a.x - b.x || b.size - a.size);
  return JSON.stringify({median, count: items.length, items: pool.slice(0, 12)});
})()"""


def fingerprint(info: dict) -> str:
    """Identity of the observable page state.

    Two observations with the same fingerprint offer the same choices, so an action between them
    changed nothing the model could have used. ``actions`` carries the current values, which is
    what makes typing and ticking visible here.
    """
    content = {k: info.get(k) for k in ("url", "text", "actions", "marker")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()
_OP_KINDS = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}

# Chrome flags that are required on machines whose GPU driver is unreliable; harmless elsewhere.
CHROME_FLAGS = [
    "--headless=new", "--no-first-run", "--no-default-browser-check", "--no-sandbox",
    "--disable-gpu", "--disable-gpu-compositing", "--disable-software-rasterizer",
    "--disable-accelerated-2d-canvas", "--disable-features=VizDisplayCompositor",
    "--hide-scrollbars", "--proxy-server='direct://'", "--proxy-bypass-list=*",
]


class CDP:
    """Synchronous CDP client: one websocket, request/response ping-pong."""

    def __init__(self, ws_url: str):
        import websockets
        self._ws_mod = websockets
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._ws = self._run(self._connect(ws_url))
        self._n = 0

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=180)

    async def _connect(self, url):
        return await self._ws_mod.connect(url, max_size=None)

    async def _call(self, method, params, session_id):
        self._n += 1
        mid = self._n
        msg = {"id": mid, "method": method, "params": params or {}}
        if session_id:
            msg["sessionId"] = session_id
        await self._ws.send(json.dumps(msg))
        while True:
            data = json.loads(await self._ws.recv())
            if data.get("id") == mid:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})

    def __call__(self, method, session_id=None, **params):
        return self._run(self._call(method, params, session_id))


def _normalize_for_input(text: str) -> str | None:
    """Narrow a planner value to what a native picker will accept.

    ``<input type=time>`` only takes ``HH:MM``, so "12:30 PM" has to be rewritten. The AM/PM
    suffix carries no information the page lacks, because the clock format comes from the page,
    not from jeva.
    """
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\s*", text, re.IGNORECASE)
    if not match:
        return None
    hour, minute, meridiem = int(match.group(1)), match.group(2), (match.group(3) or "").lower()
    if hour > 23:
        return None
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}"


# Chrome 147+ stopped serving /json/* for the default profile, so the websocket path has to come
# from the file Chrome itself writes next to the profile: line 1 is the port, line 2 the path.
CHROME_PROFILES = ("~/.config/google-chrome", "~/.config/chromium", "~/.config/chromium-browser",
                   "~/.config/microsoft-edge")


def chrome_ws_url(profile: str | None = None) -> str:
    """Find the DevTools websocket of an already-running Chrome, for reusing its logged-in session.

    Requires remote debugging to be allowed once in that browser (chrome://inspect/#remote-debugging).
    """
    candidates = [profile] if profile else list(CHROME_PROFILES)
    tried = []
    for raw in candidates:
        base = Path(raw).expanduser()
        tried.append(str(base))
        marker = base / "DevToolsActivePort"
        if not marker.is_file():
            continue
        port, _path = (marker.read_text().splitlines() + ["", ""])[:2]
        if not port.strip().isdigit():
            continue
        return f"ws://127.0.0.1:{port.strip()}" + (_path.strip() or "/")
    raise RuntimeError(
        "No running Chrome with remote debugging found. Checked: " + ", ".join(tried) +
        ". Allow it once at chrome://inspect/#remote-debugging, or pass --cdp-ws explicitly.")


def launch_chrome(port: int = 9333, profile: str = "/tmp/jeva-chrome-profile", *,
                  fresh: bool = True, headless: bool = True, window: str = "1280,900",
                  url: str | None = None):
    """Start a Chrome and return ``(process, websocket url)``.

    ``fresh=True`` wipes the profile first, which is right for a throwaway run. Pass
    ``fresh=False`` with a directory of your own to keep cookies between runs: sign in once in
    that profile (``jeva login``) and later headless runs reuse the session -- no per-run
    permission prompt, unlike attaching to a Chrome you are using.

    Headless is the default precisely because it needs no approval from anyone.
    """
    if fresh:
        shutil.rmtree(profile, ignore_errors=True)
    flags = [f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
             f"--window-size={window}", *CHROME_FLAGS]
    if not headless:
        flags = [f for f in flags if not f.startswith("--headless")]
    if url:
        flags.append(url)
    binary = (os.environ.get("CHROME") or shutil.which("google-chrome")
              or shutil.which("chromium") or shutil.which("chromium-browser"))
    if not binary:
        raise RuntimeError("No Chrome/Chromium found; set $CHROME to its path")
    proc = subprocess.Popen([binary, *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                return proc, json.loads(r.read())["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.25)
    proc.kill()
    raise RuntimeError("Chrome did not become ready in 20s")


def close_chrome(proc, cdp, timeout: float = 20) -> None:
    """Shut Chrome down the way it expects, so cookies reach the profile on disk.

    Chrome writes cookies in batches; SIGTERM skips that flush. A run that signed in and was
    killed can therefore come back signed out, which looks exactly like the profile not working.
    ``Browser.close`` is the graceful request, and termination is only the fallback.
    """
    try:
        cdp("Browser.close")
    except Exception:                                                 # noqa: BLE001
        pass
    if proc is not None:
        try:
            proc.wait(timeout=timeout)
        except Exception:                                             # noqa: BLE001
            try:
                proc.terminate()
            except Exception:                                         # noqa: BLE001
                pass


class StalePage(ValueError):
    """A decision no longer refers to the page it was made against."""


class Browser:
    """Owns one tab. ``observe()`` returns the page state; ``act()`` executes an index."""

    def __init__(self, cdp: CDP, url: str, *, width: int = 1120, height: int = 780):
        self.cdp = cdp
        self.target = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                  deviceScaleFactor=1, mobile=False)
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        # Without the Page domain enabled, Page.captureScreenshot never returns.
        self.call("Page.enable")
        self.after_input = None
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)

    def call(self, method, **params):
        return self.cdp(method, session_id=self.session, **params)

    def evaluate(self, expression, await_promise=False):
        res = self.call("Runtime.evaluate", expression=expression,
                        returnByValue=True, awaitPromise=await_promise)
        if res.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return res.get("result", {}).get("value")

    def observe(self, screenshot: bool = False, settle: float = 0.05):
        """``screenshot`` is accepted for API compatibility and ignored.

        After a fill, give an autocomplete list a moment to render before snapshotting —
        otherwise the option elements the model needs to click are still missing.
        """
        last_kind = self.after_input
        self.after_input = None
        if last_kind:
            self.wait_for_frame()
        else:
            time.sleep(settle)
        for attempt in range(10):
            try:
                info = self.evaluate(SNAPSHOT_JS)
                if info is None:
                    raise StalePage("Document is navigating")
                info["fingerprint"] = fingerprint(info)
                return info
            except (StalePage, RuntimeError):
                if attempt == 9:
                    raise
                time.sleep(0.1)
        raise StalePage("Page did not settle")

    def dismiss_overlays(self) -> list[str]:
        """Click the close button of anything covering the viewport, and report what was closed.

        Conservative by construction: an overlay with no recognisable close affordance is left
        alone, because guessing (removing it) can break the page instead of clearing an ad.
        """
        try:
            closed = self.evaluate(DISMISS_OVERLAYS_JS)
        except (StalePage, RuntimeError):
            return []
        try:
            result = json.loads(closed) if closed else []
        except json.JSONDecodeError:
            result = []
        if result:
            self.wait_for_frame()
        return result

    @staticmethod
    def _same_site(host: str, page_host: str) -> bool:
        """True when ``host`` belongs to the same site family as the page.

        Sina's sports page ranks a Weibo hot-search widget above its own articles: those blocks are
        smaller and higher up, so typography alone promotes them. They link to a different domain,
        and news links stay inside the publication -- so this is the signal that separates a
        sidebar widget from the content the page actually publishes.
        """
        if not host or not page_host or host == page_host:
            return True
        parts = page_host.split(".")
        for i in range(len(parts) - 1):
            suffix = ".".join(parts[i:])
            if host == suffix or host.endswith("." + suffix):
                return True
        return False

    def prominent(self, limit: int = 10) -> list[dict]:
        """Text blocks ranked by visual prominence: the page's own typography decides, not DOM order."""
        try:
            raw = self.evaluate(PROMINENT_JS)
        except (StalePage, RuntimeError):
            return []
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return []
        items = (data or {}).get("items") or []
        median = (data or {}).get("median")
        # Label the layout column each block sits in. Ranking is by position, so a sidebar item
        # can land between two main-column headlines -- on sina they are four pixels apart -- and
        # a caller asking for "the first news item" needs to be able to tell them apart. Which
        # column is *the* content column is a layout question, so it is reported, not guessed.
        # Numbered left to right, not in order of appearance: a sidebar high on the page used to
        # become "column 0", which reads as if it were the main content.
        columns: list[float] = []
        for it in items:
            for c in columns:
                if abs(it["x"] - c) < 60:
                    it["_col"] = c
                    break
            else:
                columns.append(it["x"])
                it["_col"] = it["x"]
        for it in items:
            it["column"] = sorted(columns).index(it.pop("_col"))
        import urllib.parse
        page_host = urllib.parse.urlparse(self.evaluate("location.href") or "").hostname or ""
        for it in items:
            it["external"] = not self._same_site(it.get("host", ""), page_host)
        # Content the page publishes outranks a widget that merely sits higher on the screen.
        items.sort(key=lambda i: (i["external"], i["y"], i["x"]))
        return [{**i, "median": median} for i in items[:limit]]

    def wait_for_frame(self) -> None:
        """Block until the page has painted, so the next read reflects the action just taken.

        A navigation tears the pending promise down and the CDP layer raises; that is exactly the
        case ``observe()`` retries, so it is swallowed here rather than escaping the loop.
        """
        try:
            self.evaluate(FRAME_JS, await_promise=True)
        except (StalePage, RuntimeError):
            pass

    def targets(self) -> dict:
        """Currently open page targets, keyed by target id (browser-level, not session-level)."""
        infos = self.cdp("Target.getTargets")["targetInfos"]
        return {t["targetId"]: t for t in infos if t.get("type") == "page"}

    def follow_new_tab(self, before: dict, settle: float = 0.6) -> str | None:
        """Move this session onto a tab the last action opened, and return its url.

        News portals open almost every navigation link with ``target="_blank"`` (sina, for one,
        marks all four of its 财经 links that way). Without this the click looks like it worked --
        the page did change -- while the session stays on the old document forever.

        A new tab pointing at the page we are already on is not progress: sina's finance channel
        links "新浪财经" back to itself, and following that loops forever on a page that never
        changes meaningfully. Such a tab is closed and ignored.
        """
        def same_page(a: str, b: str) -> bool:
            """Compare host+path only: sina links http://finance.sina.com.cn from its own
            https://finance.sina.com.cn page, and the scheme difference is not navigation."""
            import urllib.parse
            pa, pb = urllib.parse.urlparse(a), urllib.parse.urlparse(b)
            return bool(pa.netloc) and pa.netloc == pb.netloc and pa.path.rstrip("/") == pb.path.rstrip("/")

        current = ""
        try:
            current = self.evaluate("location.href") or ""
        except Exception:                                             # noqa: BLE001
            pass
        for _ in range(20):
            for tid, t in self.targets().items():
                url = t.get("url", "")
                if tid in before or tid == self.target or url in ("", "about:blank"):
                    continue
                if current and same_page(url, current):
                    try:
                        self.cdp("Target.closeTarget", targetId=tid)
                    except Exception:                                 # noqa: BLE001
                        pass
                    continue
                previous = self.target
                self.target = tid
                self.session = self.cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
                self.after_input = None
                try:
                    self.call("Page.enable")
                except Exception:                                     # noqa: BLE001
                    pass
                time.sleep(settle)
                try:
                    self.cdp("Target.closeTarget", targetId=previous)  # do not leak tabs
                except Exception:                                     # noqa: BLE001
                    pass
                return url
            time.sleep(0.15)
        return None

    def fresh(self, page: dict) -> bool:
        """True when the page still matches the observation this decision came from.

        Checked before acting and before accepting a terminal ``DONE`` / ``BLOCKED``, so a claim
        is never trusted against a page that has moved on.
        """
        try:
            return self.evaluate(MARKER_JS) == page.get("marker")
        except StalePage:
            return False

    def act(self, action: dict, page: dict | None = None, text: str | None = None):
        """``page`` is accepted for API compatibility; freshness is re-checked in the page."""
        """Execute one action taken from a previous observation."""
        kind = action["kind"]
        self.after_input = kind
        if kind == "wait":
            time.sleep(0.1)
            return {"executed": action["id"]}
        if type(action.get("node")) is not int:
            raise ValueError("Invalid observed node")
        target = self.evaluate(
            "(a => { const e = window.__jevFast?.nodes.get(a.node);"
            " if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled=\"true\"],[inert]')"
            "     || !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;"
            " if (a.kind === 'select') { if (e.tagName !== 'SELECT' ||"
            "   ![...e.options].some(o => o.value === a.value && !o.disabled)) return null;"
            "   e.value = a.value; e.dispatchEvent(new Event('input',{bubbles:true}));"
            "   e.dispatchEvent(new Event('change',{bubbles:true})); return {x:0,y:0}; }"
            " const r = e.getBoundingClientRect(), x = r.x + r.width/2, y = r.y + r.height/2;"
            " if (!r.width || !r.height || x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return null;"
            " return {x, y}; })(" + json.dumps(action) + ")")
        if target is None:
            raise StalePage("Target changed, is covered, or is not selectable. Observe again.")
        if kind != "select":
            x, y = target["x"], target["y"]
            for event in ("mousePressed", "mouseReleased"):
                self.call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
            if kind == "fill" and text:
                before = self._focused_value()
                # commands=["selectAll"] is what actually clears the field; the modifier alone
                # is not enough over CDP.
                self.call("Input.dispatchKeyEvent", type="keyDown", key="a", code="KeyA",
                          modifiers=2, commands=["selectAll"])
                self.call("Input.dispatchKeyEvent", type="keyUp", key="a", code="KeyA", modifiers=2)
                self.call("Input.insertText", text=text)
                # Date/time/color inputs reject insertText entirely, so retry through the native
                # setter. Compare with the pre-write value because the field may already have
                # held something (or nothing) and insertText silently did nothing either way.
                if self._focused_value() == before:
                    candidates = [text]
                    normalized = _normalize_for_input(text)
                    if normalized and normalized != text:
                        candidates.append(normalized)
                    for candidate in candidates:
                        self._set_native_value(candidate)
                        if self._focused_value() == candidate:
                            break
        return {"executed": action["id"]}

    def _focused_value(self):
        return self.evaluate("(() => (document.activeElement || {}).value)()")

    def _set_native_value(self, text):
        return self.evaluate(
            "(t => { const e = document.activeElement; if (!e) return null;"
            " const proto = e.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype"
            "                                              : HTMLInputElement.prototype;"
            " const d = Object.getOwnPropertyDescriptor(proto, 'value');"
            " d && d.set.call(e, t);"
            " e.dispatchEvent(new Event('input', {bubbles: true}));"
            " e.dispatchEvent(new Event('change', {bubbles: true}));"
            " return e.value; })(" + json.dumps(text) + ")")

    def close(self):
        if self.target:
            try:
                self.cdp("Target.closeTarget", targetId=self.target)
            finally:
                self.target = None


def action_space(actions):
    """Index observed controls the way the model sees them.

    Returns ``(elements, targets, controls)`` where ``targets[OPERATION]`` maps a target index to
    the underlying action, and ``controls`` holds non-target operations such as WAIT.
    """
    elements, indices, targets, controls = [], {}, {}, {}
    for action in actions:
        kind = action.get("kind")
        if kind not in _OP_KINDS:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in
                       ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=str(action.get("label", "")).split(" → ")[0],
                           operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = _OP_KINDS[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action.get("label", ""),
                                       "value": action.get("value")})
        group[target] = action
    return elements, targets, controls


def to_page(page_info, elements):
    """Build a ``jeva.Page`` from an observation + the element table."""
    from jeva.render import Element, Option, Page
    els = []
    for e in elements:
        els.append(Element(
            index=str(e["index"]),
            role=e.get("role", "element"),
            label=e.get("label", ""),
            value=e.get("value"),
            operations=list(e.get("operations") or []),
            options=[Option(index=o["index"], label=str(o["label"]).split(" → ")[-1],
                            value=o.get("value")) for o in (e.get("options") or [])],
            checked=e.get("checked"), selected=e.get("selected"), expanded=e.get("expanded")))
    return Page(elements=els, url=page_info.get("url", ""), title=page_info.get("title", ""),
                text=page_info.get("text", ""))
