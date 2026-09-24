"""Backwards-compatible import path.

The runtime moved into the package (``jeva.browser``) so it can be used outside this
repo's eval scripts -- see the ``browser-agent`` skill. Kept here because the eval
scripts and their docs reference ``browser.<name>``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jeva.browser import (                                        # noqa: E402,F401
    CDP, CHROME_FLAGS, SNAPSHOT_JS, Browser, StalePage, _normalize_for_input,
    action_space, launch_chrome, to_page,
)
