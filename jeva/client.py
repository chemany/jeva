"""Minimal client for jeva. Talks to any OpenAI-compatible endpoint."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

from .prompt import DEFAULT_RULES, SYSTEM, build_prompt
from .render import Page, render_state


@dataclass
class Vote:
    """The result of sampling one decision several times."""

    action: Action            # the most common sample
    samples: list[Action]
    agreement: float          # share of samples equal to ``action``
    counts: dict[str, int]

    @property
    def unanimous(self) -> bool:
        return self.agreement >= 1.0

    @property
    def alternatives(self) -> list[str]:
        return [k for k in self.counts if k != json.dumps(self.action.as_dict(), sort_keys=True)]


@dataclass
class Action:
    """One decision. ``target`` is an index taken from the state you supplied."""

    operation: str
    target: str = ""
    text: str = ""
    raw: str = ""

    def as_dict(self) -> dict:
        return {"operation": self.operation, "target": self.target, "text": self.text}

    @property
    def terminal(self) -> bool:
        return self.operation in ("DONE", "BLOCKED")


class Jeva:
    """Stateless decision client.

    >>> jeva = Jeva("http://127.0.0.1:8020/v1", model="jeva")
    >>> action = jeva.decide(page, goal="Find one-way flights from Zurich to London.")
    >>> action.operation, action.target
    ('CLICK', '2')
    """

    def __init__(self, base_url: str, model: str = "jeva", api_key: str = "EMPTY",
                 timeout: int = 120, max_tokens: int = 64, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_latency_ms: Optional[float] = None

    def decide(self, page: Page, goal: str, history: Iterable[str] = (),
               rules: str = DEFAULT_RULES, temperature: Optional[float] = None,
               seed: Optional[int] = None) -> Action:
        """One decision. ``temperature``/``seed`` override the instance defaults, which is what
        makes independent samples possible (at temperature 0 every sample is identical)."""
        prompt = build_prompt(page, goal, history, rules)
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            # Required. Without it the model emits a reasoning trace and `content` comes back empty.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        }
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        import time
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        self.last_latency_ms = (time.time() - t0) * 1000
        return parse_action(payload["choices"][0]["message"].get("content") or "")

    def vote(self, page: Page, goal: str, history: Iterable[str] = (),
             rules: str = DEFAULT_RULES, n: int = 3, temperature: float = 0.8) -> "Vote":
        """Sample the same state ``n`` times and report how much the samples agree.

        This is agreement, not a calibrated probability: the model was never trained to produce
        one, and a confident-but-wrong sample would look exactly the same. It exists so a caller
        can refuse to act on a decision the model will not reproduce.
        """
        samples = [self.decide(page, goal, history, rules, temperature=temperature, seed=1000 + i)
                   for i in range(n)]
        keys = [json.dumps(s.as_dict(), sort_keys=True) for s in samples]
        counts: dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        winner_key, winner_count = max(counts.items(), key=lambda kv: kv[1])
        return Vote(action=samples[keys.index(winner_key)], samples=samples,
                    agreement=winner_count / n, counts=counts)

    # convenience: render + decide in one call
    def decide_from_elements(self, elements, goal: str, *, url: str = "", title: str = "",
                             text: str = "", history: Iterable[str] = ()) -> Action:
        return self.decide(Page(elements=list(elements), url=url, title=title, text=text),
                           goal, history)


def parse_action(content: str) -> Action:
    """Extract the JSON action. Falls back to BLOCKED on unparseable output."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return Action(operation="BLOCKED", raw=content)
    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return Action(operation="BLOCKED", raw=content)
    return Action(operation=str(data.get("operation") or "BLOCKED").upper(),
                  target="" if data.get("target") is None else str(data["target"]),
                  text=str(data.get("text") or ""),
                  raw=content)


def resolve(page: Page, action: Action):
    """Turn an action into the element it refers to, purely by index lookup.

    jeva never emits selectors, coordinates or JavaScript — the caller resolves the index
    against the same observation it sent in.
    """
    for e in page.elements:
        if str(e.index) == action.target:
            return e
        for o in e.options:
            if o.index == action.target:
                return e
    return None
