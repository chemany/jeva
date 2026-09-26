"""jeva — a 2B browser-agent decision model, as an API.

``Jeva`` takes a page observation and a goal and returns one action as JSON. That is the whole
package: it decides, it does not drive.

Driving belongs to the agent software that calls it. This repo integrates
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) -- its browser layer, its
loop and its guards, with jeva swapped in as the decision model: see
``integrations/jev-ultrafast/``. The browser harness under ``evals/`` exists for collecting and
evaluating, not for production driving.
"""
from .client import Action, Jeva, parse_action, resolve
from .prompt import DEFAULT_RULES, SYSTEM, build_prompt
from .render import (OPERATIONS, OPERATION_DESCRIPTIONS, Element, Option, Page,
                     available_operations, render_state, target_criteria)

__version__ = "0.1.0"
__all__ = [
    "Jeva", "Action", "parse_action", "resolve",
    "Page", "Element", "Option", "render_state", "available_operations", "target_criteria",
    "build_prompt", "SYSTEM", "DEFAULT_RULES",
    "OPERATIONS", "OPERATION_DESCRIPTIONS",
    "__version__",
]
