"""The agent loop: observe, decide with jeva, act, repeat.

jeva itself only *decides*. This module owns the parts that make a decision safe to execute:

* the decision is resolved against the observation it was made from, so a target index can never
  point at an element that appeared later;
* a page that changed mid-step raises :class:`StalePage`, and the loop re-observes instead of
  clicking into the void;
* three consecutive actions that leave the page unchanged end the run as ``blocked`` -- a decider
  that has stopped making progress should not spend the whole step budget;
* ``DONE`` / ``BLOCKED`` are the model's own claims. They are reported as ``status`` and the final
  page text is in the result, so the caller can verify. This module does not pretend otherwise.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .browser import CDP, Browser, StalePage, action_space, launch_chrome, to_page
from .client import Jeva

# An observation is only useful if a decision can act on it; these bound the loop instead of
# letting a confused decider run forever.
DEFAULT_MAX_STEPS = 25
NO_PROGRESS_LIMIT = 3


@dataclass
class Step:
    """One executed action, recorded before the next observation so a stale read cannot erase it."""

    n: int
    operation: str
    target: str
    text: str
    label: str
    url: str
    page_changed: bool = False
    latency_ms: int = 0

    def as_dict(self) -> dict:
        return {"n": self.n, "operation": self.operation, "target": self.target, "text": self.text,
                "label": self.label, "url": self.url, "page_changed": self.page_changed,
                "latency_ms": self.latency_ms}


@dataclass
class RunResult:
    """What happened. ``status`` is the model's claim; ``verify`` the caller's job."""

    status: str                      # "done" | "blocked" | "failed"
    reason: str
    url: str = ""
    title: str = ""
    text: str = ""
    steps: list[Step] = field(default_factory=list)
    elapsed_ms: int = 0
    screenshots: list[str] = field(default_factory=list)
    invalid_targets: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "done"

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "url": self.url, "title": self.title,
                "steps": [s.as_dict() for s in self.steps], "elapsed_ms": self.elapsed_ms,
                "screenshots": self.screenshots, "invalid_targets": self.invalid_targets,
                "text": self.text[:4000]}


class Agent:
    """Drive one page towards one goal.

        with Agent("https://example.com/form", "Order a large pizza ...") as agent:
            result = agent.run()
    """

    def __init__(self, url: str, goal: str, *, jeva: Jeva | None = None,
                 max_steps: int = DEFAULT_MAX_STEPS, port: int = 9333,
                 profile: str = "/tmp/jeva-agent-profile", chrome: str | None = None,
                 screenshot_dir: str | None = None, cdp_ws: str | None = None,
                 settle: float = 0.6):
        if not goal.strip():
            raise ValueError("Supply a goal")
        self.url = url
        self.goal = goal.strip()
        self.jeva = jeva or Jeva()
        self.max_steps = max_steps
        self.settle = settle
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        if self.screenshot_dir:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._proc = None
        if cdp_ws:
            # Attach to a browser the caller already runs -- the way to reuse a logged-in session.
            self.cdp = CDP(cdp_ws)
        else:
            import os
            if chrome:
                os.environ["CHROME"] = chrome
            self._proc, ws = launch_chrome(port=port, profile=profile)
            self.cdp = CDP(ws)
        self.browser = Browser(self.cdp, url)
        self.history: list[Step] = []
        self.invalid_targets = 0
        self._screenshots: list[str] = []

    # -- internals ---------------------------------------------------------------------------
    def _observe(self):
        return self.browser.observe(screenshot=False, settle=self.settle)

    def _screenshot(self, tag: str) -> None:
        """Best effort. A capture that loses a race with navigation is not worth failing a run."""
        if not self.screenshot_dir:
            return
        try:
            png = base64.b64decode(self.browser.call("Page.captureScreenshot", format="png")["data"])
            path = self.screenshot_dir / f"{len(self._screenshots):02d}-{tag}.png"
            path.write_bytes(png)
            self._screenshots.append(str(path))
        except Exception:                                             # noqa: BLE001
            pass

    def _resolve(self, page, action):
        """Map (operation, target) onto the observed action table, or explain why it does not fit."""
        elements, targets, _controls = action_space(page["actions"])
        if action.operation in ("DONE", "BLOCKED"):
            return None, elements, targets, ""
        if action.operation not in targets:
            return None, elements, targets, f"operation {action.operation} is not offered on this page"
        act = targets[action.operation].get(action.target or "")
        if act is None:
            return None, elements, targets, f"target {action.target!r} is not a valid {action.operation} target"
        label = str(act.get("label", "")).split(" → ")[0]
        return act, elements, targets, label

    # -- the loop ----------------------------------------------------------------------------
    def run(self) -> RunResult:
        started = time.perf_counter()
        result = RunResult(status="failed", reason="run did not start")
        history_lines: list[str] = []
        try:
            page = self._observe()
            self._screenshot("start")
            for n in range(1, self.max_steps + 1):
                elements, _targets, _controls = action_space(page["actions"])
                action = self.jeva.decide(to_page(page, elements), self.goal, history_lines)
                latency = int(self.jeva.last_latency_ms or 0)

                if action.terminal:
                    # Never accept a terminal claim about a page that has since changed.
                    if not self.browser.fresh(page):
                        page = self._observe()
                        continue
                    result = RunResult(
                        status="done" if action.operation == "DONE" else "blocked",
                        reason=f"model reported {action.operation}",
                        url=page.get("url", ""), title=page.get("title", ""),
                        text=str(page.get("text") or ""), steps=self.history,
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                        screenshots=self._screenshots, invalid_targets=self.invalid_targets)
                    self._screenshot("end")
                    return result

                act, _elements, _targets, label = self._resolve(page, action)
                if act is None:
                    self.invalid_targets += 1
                    history_lines.append(f"{action.operation} {action.target} (rejected)")
                    if self.invalid_targets >= 3:
                        result = RunResult(
                            status="failed", reason=f"repeated invalid targets: {label}",
                            url=page.get("url", ""), steps=self.history,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            screenshots=self._screenshots, invalid_targets=self.invalid_targets)
                        return result
                    page = self._observe()
                    continue

                step = Step(n=n, operation=action.operation, target=action.target,
                            text=action.text, label=label, url=page.get("url", ""),
                            latency_ms=latency)
                # Record before acting: a stale observation after the action must not lose the step.
                self.history.append(step)
                try:
                    self.browser.act(act, page, text=action.text)
                except StalePage as exc:
                    step.label = f"{label} (stale: {exc})"
                    page = self._observe()
                    continue

                previous = page
                page = self._observe()
                step.page_changed = page.get("fingerprint") != previous.get("fingerprint")
                history_lines.append(f"{action.operation} {label}"
                                     + (f" = {action.text!r}" if action.text else ""))
                self._screenshot(f"step{n}")

                recent = self.history[-NO_PROGRESS_LIMIT:]
                if len(recent) == NO_PROGRESS_LIMIT and all(not s.page_changed for s in recent):
                    result = RunResult(
                        status="blocked", reason=f"{NO_PROGRESS_LIMIT} actions changed nothing",
                        url=page.get("url", ""), title=page.get("title", ""),
                        text=str(page.get("text") or ""), steps=self.history,
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                        screenshots=self._screenshots, invalid_targets=self.invalid_targets)
                    return result

            result = RunResult(
                status="blocked", reason=f"step budget of {self.max_steps} exhausted",
                url=page.get("url", ""), title=page.get("title", ""),
                text=str(page.get("text") or ""), steps=self.history,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                screenshots=self._screenshots, invalid_targets=self.invalid_targets)
        except Exception as exc:                                      # noqa: BLE001
            result = RunResult(status="failed", reason=f"{type(exc).__name__}: {exc}",
                               steps=self.history,
                               elapsed_ms=int((time.perf_counter() - started) * 1000),
                               screenshots=self._screenshots, invalid_targets=self.invalid_targets)
        finally:
            self.close()
        return result

    def close(self) -> None:
        try:
            self.browser.close()
        except Exception:                                             # noqa: BLE001
            pass
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:                                         # noqa: BLE001
                pass
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def run(url: str, goal: str, **kwargs) -> RunResult:
    """One-shot convenience wrapper around :class:`Agent`."""
    return Agent(url, goal, **kwargs).run()
