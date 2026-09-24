"""jeva — a 2B browser-agent decision model.

Two ways in:

``Jeva``
    give it a page observation and a goal, get back one action as JSON. The model only decides.
``Agent``
    give it a URL and a goal; it launches Chrome, observes, decides, acts, and guards every
    decision against a page that moved underneath it.
"""
from .agent import Agent, RunResult, Step, run
from .browser import Browser, launch_chrome
from .client import Action, Jeva, parse_action, resolve
from .prompt import DEFAULT_RULES, SYSTEM, build_prompt
from .render import (OPERATIONS, OPERATION_DESCRIPTIONS, Element, Option, Page,
                     available_operations, render_state, target_criteria)

__version__ = "0.1.0"
__all__ = [
    "Jeva", "Action", "parse_action", "resolve",
    # driving a real browser with jeva as the decision maker
    "Agent", "RunResult", "Step", "run", "Browser", "launch_chrome",
    "Page", "Element", "Option", "render_state", "available_operations", "target_criteria",
    "build_prompt", "SYSTEM", "DEFAULT_RULES",
    "OPERATIONS", "OPERATION_DESCRIPTIONS",
    "__version__",
]
