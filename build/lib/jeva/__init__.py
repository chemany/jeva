"""jeva — a 2B browser-agent decision model.

Give it a page observation and a goal; it returns one action as JSON.
See README.md for the required state format and prompt.
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
