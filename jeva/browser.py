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
    # Only what fits the viewport is observable, so a laptop-sized window matters:
    # the 800x600 headless default hides the bottom of an ordinary form.
    "--window-size=1280,900",
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


def launch_chrome(port: int = 9333, profile: str = "/tmp/jeva-chrome-profile"):
    """Start a throwaway headless Chrome and return (process, websocket url)."""
    shutil.rmtree(profile, ignore_errors=True)
    flags = [f"--remote-debugging-port={port}", f"--user-data-dir={profile}", *CHROME_FLAGS]
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

    def wait_for_frame(self) -> None:
        """Block until the page has painted, so the next read reflects the action just taken.

        A navigation tears the pending promise down and the CDP layer raises; that is exactly the
        case ``observe()`` retries, so it is swallowed here rather than escaping the loop.
        """
        try:
            self.evaluate(FRAME_JS, await_promise=True)
        except (StalePage, RuntimeError):
            pass

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
