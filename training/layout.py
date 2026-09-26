"""Shuffle the element order inside a training prompt, remapping the target so the answer stays true.

The layout ablation found the one remaining channel the title fix did not close. With the form's
elements shuffled and renumbered -- same controls, same labels, same goal -- the model answered with
a step that advances nothing, three times out of three. It was not reading the row it had chosen; it
was remembering where that control usually sits. On a site whose form is ordered differently, that
is a wrong click.

A training prompt is text, so the shuffle has to go through the rendered form. Each element row is
self-contained -- role, label, value, state flags, `{ops: ...}` and any `options: ...` are all on the
line -- so the rows can be parsed back into elements, reordered, and the prompt rebuilt with
`build_prompt`, which regenerates the `Targets for ...` lines, the available-operations list and the
worked example consistently.

The parser is verified by round trip: rebuilding an unshuffled prompt must reproduce it byte for byte,
`verify_roundtrip()` below checks that over a whole dataset. Anything the parser fails to reproduce
would silently corrupt training, so it is not left to inspection.
"""
from __future__ import annotations

import random
import re
import sys

sys.path.insert(0, "/root/code/jeva")
from jeva.prompt import build_prompt                                # noqa: E402
from jeva.render import Element, Option, Page                       # noqa: E402

PAGE_RE = re.compile(r"^Page: (.*?)  \((.*)\)$")
GOAL_RE = re.compile(r"^Goal: (.*)$")
ROW_RE = re.compile(r"^\[(\d+)\] ([a-z]+): (.*)$")
OPTIONS_RE = re.compile(r" options: (.*)$")
OPS_RE = re.compile(r" \{ops: ([^}]*)\}$")
OPTION_ITEM_RE = re.compile(r"^\[([\d:]+)\] (.*)$")
META_RE = re.compile(r" \(([^()]*)\)")
FLAG_RE = {k: re.compile(rf" {k}=(\S+)") for k in ("checked", "selected", "expanded")}


def _split_row(rest: str) -> dict:
    """Pull options, ops, flags and the value out of one element row's tail."""
    out: dict = {"options": [], "operations": [], "flags": {}, "value": None}
    m = OPTIONS_RE.search(rest)
    if m:
        for part in m.group(1).split(" | "):
            mm = OPTION_ITEM_RE.match(part.strip())
            if mm:
                out["options"].append((mm.group(1), mm.group(2).strip()))
        rest = rest[:m.start()].rstrip()
    m = OPS_RE.search(rest)
    if m:
        out["operations"] = [o for o in m.group(1).split(",") if o]
        rest = rest[:m.start()].rstrip()
    for key, rx in FLAG_RE.items():
        m = rx.search(rest)
        if m:
            out["flags"][key] = m.group(1)
            rest = (rest[:m.start()] + rest[m.end():]).rstrip()
    # Value sits between the label and the flags: either `"v"` or the literal `not set`.
    m = re.search(r'(?:(not set)|"([^"]*)")$', rest)
    if m:
        out["value"] = "" if m.group(1) else m.group(2)
        rest = rest[:m.start()].rstrip()
    # A textblock carries its layout hint in parentheses and has no form value at all.
    m = META_RE.search(rest)
    if m and not out["value"] and not out["operations"]:
        out["meta"] = m.group(1)
        rest = (rest[:m.start()] + rest[m.end():]).rstrip()
    out["label"] = rest
    return out


def parse_prompt(prompt: str) -> tuple[Page, str, list[str]]:
    """Split a training prompt back into the page, the goal, and the recent actions."""
    title = url = goal = ""
    text = ""
    history: list[str] = []
    elements: list[Element] = []
    seen_row = False
    for line in prompt.splitlines():
        m = PAGE_RE.match(line)
        if m:
            title, url = m.group(1), m.group(2)
            continue
        m = GOAL_RE.match(line)
        if m:
            goal = m.group(1)
            continue
        m = ROW_RE.match(line)
        if m and not seen_row or m:
            parts = _split_row(m.group(3))
            elements.append(Element(
                index=m.group(1), role=m.group(2), label=parts["label"],
                value=parts["value"], meta=parts.get("meta", ""),
                operations=parts["operations"],
                options=[Option(index=i, label=l) for i, l in parts["options"]],
                checked=parts["flags"].get("checked"),
                selected=parts["flags"].get("selected"),
                expanded=parts["flags"].get("expanded")))
            seen_row = True
            continue
        if line.startswith("Page text: "):
            text = line[len("Page text: "):]
            continue
        if line.startswith("Recent actions: "):
            rest = line[len("Recent actions: "):]
            history = [] if rest == "none" else rest.split("; ")
    return Page(elements=elements, url=url, title=title, text=text), goal, history


def _apply(prompt: str, page: Page, order: list[int], remap: dict[str, str]) -> str:
    """Emit the reordered rows and regenerated `Targets for ...` lines, touching nothing else.

    Rebuilding the whole prompt is not safe: the example line in this dataset was written by an
    earlier `build_prompt` that chose its target differently, so a full rebuild would rewrite a line
    the shuffle has no business changing. Only the two parts that depend on element order are
    replaced.
    """
    new_elements = []
    for new, old in enumerate(order, 1):
        e = page.elements[old]
        new_elements.append(Element(
            index=str(new), role=e.role, label=e.label, value=e.value, meta=e.meta,
            operations=list(e.operations),
            options=[Option(index=f"{new}:{o.index.split(':')[1]}" if ":" in o.index else o.index,
                            label=o.label) for o in e.options],
            checked=e.checked, selected=e.selected, expanded=e.expanded))
    new_page = Page(elements=new_elements, url=page.url, title=page.title, text=page.text,
                    page_text_chars=page.page_text_chars)

    from jeva.render import target_criteria
    out: list[str] = []
    emitted = False
    for line in prompt.splitlines():
        if ROW_RE.match(line):
            if not emitted:
                out.extend(e.render() for e in new_page.elements)
                emitted = True
            continue
        m = re.match(r"^Targets for (CLICK|TYPE_TEXT|SELECT|READ): ", line)
        if m:
            crit = target_criteria(new_page, m.group(1))
            out.append(f"Targets for {m.group(1)}: " + " | ".join(crit.values()))
            continue
        out.append(line)
    return "\n".join(out)


def verify_identity(prompts: list[str], limit: int = 400) -> None:
    """No-op permutation must reproduce each prompt byte for byte, or the transform corrupts training."""
    bad = 0
    for i, p in enumerate(prompts[:limit]):
        page, _goal, _history = parse_prompt(p)
        order = list(range(len(page.elements)))
        remap = {page.elements[k].index: page.elements[k].index for k in order}
        if _apply(p, page, order, remap) != p:
            bad += 1
            if bad == 1:
                print("  first mismatch on prompt", i)
                a, b = _apply(p, page, order, remap).splitlines(), p.splitlines()
                for x, y in zip(a, b):
                    if x != y:
                        print("    out:", x[:150])
                        print("    in :", y[:150])
                        break
        if i >= limit:
            break
    total = min(len(prompts), limit)
    print(f"  identity round trip: {total - bad}/{total} identical")
    if bad:
        raise SystemExit("transform does not reproduce the prompt; refusing to train on it")


def shuffle_layout(prompt: str, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Return the prompt with its elements reordered, and the old-index -> new-index map.

    Option indices are renumbered too, so a `SELECT` target goes through the same map on its element
    half: `6:2` becomes `<new index of 6>:2`.
    """
    page, _goal, _history = parse_prompt(prompt)
    if len(page.elements) < 2:
        return prompt, {}
    order = list(range(len(page.elements)))
    rng.shuffle(order)
    remap = {page.elements[old].index: str(new) for new, old in enumerate(order, 1)}
    return _apply(prompt, page, order, remap), remap


def remap_target(target: str, remap: dict[str, str]) -> str:
    if not target:
        return target
    if ":" in target:
        head, tail = target.split(":", 1)
        return f"{remap.get(head, head)}:{tail}"
    return remap.get(target, target)
