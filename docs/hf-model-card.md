---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
base_model: openbmb/MiniCPM5-2B
tags:
  - decision-model
  - browser-agent
  - agent
  - tool-use
  - gguf
  - llama.cpp
  - minicpm
---

# jeva — MiniCPM5-2B-WebDecider

A 2B browser-agent decision model. Given a page observation and a goal, it returns **exactly one
action** as compact JSON — no prose, and no invented selectors.

```json
{"operation": "CLICK", "target": "2", "text": ""}
```

- **Base model:** [`openbmb/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B) (Apache-2.0)
- **Method:** LoRA SFT (r=16, α=32, 1 epoch), then **merged into the base weights** — this release is a full model, not an adapter
- **Training data:** 9,357 trajectories from driving a real Chrome with a deterministic solver — **zero human labels, no teacher model**
- **Project / training code:** <https://github.com/&lt;you&gt;/jeva>
- **License:** Apache-2.0

## What it does

Input: a rendered page state plus a goal. Output: one action over the **indices you supplied**.

| Operation | Meaning |
|---|---|
| `CLICK` | click an element, menu option, autocomplete suggestion |
| `TYPE_TEXT` | enter or replace text in a field (the caller supplies `text`) |
| `SELECT` | choose a dropdown value, targeted as `<element>:<option>` |
| `WAIT` | an action is already in progress |
| `DONE` | every requirement is visibly satisfied |
| `BLOCKED` | no supported operation can progress |

The model never emits CSS, coordinates, or JavaScript. Your executor resolves the index against
the same snapshot it sent.

## Evaluation

Task success rate on randomised suites. Each task is executed step by step in **real Chrome** and
verified independently: the submitted values are read back from the resulting page state, so a task
counts only when the browser really ended up in the right place. The model's own `DONE` is never
trusted.

| Decision model | Params | Site 1 | Site 2 | **Site 3 (never trained)** | Latency / decision |
|---|---|---|---|---|---|
| MiniCPM5-2B (base, zero-shot) | 2B | 8% | — | — | 237 ms |
| Bonsai-27B (zero-shot) | 27B | 70% (10 tasks) | — | — | 6800 ms |
| **jeva** | **2B** | **100%** | **100%** | **100%** | **276 ms** (Q4_K_M) |

Per task type on Site 1:

| Task type | Base | **jeva** |
|---|---|---|
| Blocked page (CAPTCHA / rate limit) | 0 / 13 | **100%** |
| Flight search (radio + 3 fields + submit) | 0 / 32 | **100%** |
| Hotel filters (2 selects + checkbox) | 0 / 28 | **100%** |
| Contact form (3 fields + consent) | 2 / 13 | **100%** |
| Autocomplete (type → pick option) | 6 / 14 | **100%** |

Sites differ in label vocabulary and layout; Site 3
(`Origin` / `Destination` / `When` / `Cabin class` / `Guests` / `Round-trip` / `One-way` / `Submit`)
was held out completely, so its 100% is generalisation rather than recall.

Format reliability over 100 tasks: **0** unparseable responses and **0** invalid targets, against
97 and 171 for the base model.

Quantisation is free here: F16 (5.04 GB), Q8_0 (2.68 GB) and Q4_K_M (1.56 GB) all score 100%.

> The Bonsai column is a 10-task zero-shot sample and is indicative only. jeva is *trained for this
> action space*; the baselines are not. The comparison shows what self-generated supervision buys a
> 2B model, not that a 2B beats a 27B at being a general model.

## Usage

### llama.cpp

```bash
llama-server -m MiniCPM5-2B-WebDecider-Q4_K_M.gguf \
  --alias jeva --port 8020 -ngl 99 -c 4096 --jinja
```

```python
import json, urllib.request

SYSTEM = """You are a browser agent. Choose exactly ONE next action.
Output compact JSON only, no prose:
{"operation":"<one of the operations>","target":"<an offered target index, or empty>","text":"<only for TYPE_TEXT, else empty>"}

Hard rules:
- Never repeat a step that is already satisfied. A field that already holds the required value needs no
  further typing; a radio/checkbox already in the requested state needs no further click.
- If a suggestion/option list is open, click the matching option instead of typing again.
- DONE only when visible evidence proves every requirement of the goal is met.
- BLOCKED only when a challenge or a missing control prevents progress."""

state = """Page: Flight Search  (https://example.com/flights)
Goal: Find one-way flights from Zurich to London on September 20, 2026, for one adult in Economy.

[1] radio: Round trip "round" checked=true {ops: CLICK}
[2] radio: One way "oneway" checked=false {ops: CLICK}
[3] textbox: Where from? "Zurich" {ops: TYPE_TEXT,CLICK}
[4] textbox: Where to? "London" {ops: TYPE_TEXT,CLICK}
[5] textbox: Departure not set {ops: TYPE_TEXT,CLICK}
[7] button: Search {ops: CLICK}

Recent actions: CLICK Round trip"""

user = state + """

Available operations:
  CLICK: Click an element, button, menu option, autocomplete suggestion, or calendar day.
  TYPE_TEXT: Enter or replace text in an editable field. The caller supplies the value.
  SELECT: Select an observed dropdown value.
  DONE: Every requirement is visibly satisfied.
  BLOCKED: No supported operation can progress.

Note: to enter or change text in a field you MUST use TYPE_TEXT (CLICK on a text field only opens it). To choose a dropdown value use SELECT. Clicking a radio/checkbox that already has the requested state does nothing.
Targets for CLICK: [1] Round trip (current: round) | [2] One way (current: oneway) | [3] Where from? (current: Zurich) | [7] Search
Targets for TYPE_TEXT: [3] Where from? (current: Zurich) | [4] Where to? (current: London) | [5] Departure

Policy rules: Advance the goal by exactly one operation.

Reply with JSON only, e.g. {"operation":"CLICK","target":"3","text":""}"""

body = {"model": "jeva", "max_tokens": 64, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}]}
req = urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions",
                            data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
# {"operation": "CLICK", "target": "2", "text": ""}
```

### transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("MiniCPM5-2B-WebDecider")
model = AutoModelForCausalLM.from_pretrained("MiniCPM5-2B-WebDecider", dtype=torch.float16).cuda()

text = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

> `enable_thinking=False` is **not optional.** MiniCPM5 is a reasoning model: with thinking on, the
> trace consumes the token budget and `content` comes back empty.

### Python package

The project ships a client that builds the prompt and renders the state for you:

```bash
pip install git+https://github.com/<you>/jeva
```

```python
from jeva import Jeva, Page, Element          # noqa: F401

jeva = Jeva("http://127.0.0.1:8020/v1", model="jeva")
action = jeva.decide(page, "Find one-way flights from Zurich to London on September 20, 2026.")
```

## State format

jeva reads **compact plain text**, one line per interactive element. Five rules, each measured to
change accuracy:

1. **Never render an empty string for a missing value** — write `not set`. An empty value is read as *present*.
2. **Always emit `{ops: ...}`** — the only structural signal separating a select-only dropdown from a typeable textbox.
3. For dropdowns, list only the **unselected** options, keyed `<element>:<option>`.
4. Render `checked=true/false` for radios and checkboxes.
5. **Compact plain text.** A JSON blob degrades accuracy substantially.

The system prompt, the operation descriptions and the state layout are **part of the model**;
reproduce them exactly (`jeva.prompt` / `jeva.render` in the project do).

## Training data

No human labelling and no teacher model. A deterministic solver drives real Chrome and records
`(state, action)`; a trajectory is kept only when the environment verifies the task succeeded.
The solver sees a privileged "already satisfied / not yet done" line that is **not** recorded, so
the student has to learn state tracking from the state alone.

2,200 tasks → 9,357 verified examples in ~11 minutes; LoRA training ~70 minutes on one V100.

## Limitations

- **It decides; it does not act.** Output is an index into the observation you supplied.
- **Narrow action space:** no `SCROLL`, no file uploads, no multi-step dropdown widgets.
- Trained on form-shaped flows (search / filter / form / autocomplete); off-distribution behaviour is not characterised.
- **Prompt-sensitive** — changing the system prompt or state layout degrades output.
- Not a general assistant, and it does not return calibrated probabilities.
- Evaluated on three synthetic sites in real Chrome, not on production websites; the 27B baseline rests on 10 tasks.

## License

Apache-2.0, same as the base. jeva is a fine-tuned derivative of
[`openbmb/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B); the name follows the
MiniCPM derivative convention.

```bibtex
@misc{minicpm5,
  title  = {MiniCPM5-2B},
  author = {OpenBMB},
  year   = {2026},
  url    = {https://modelscope.cn/models/OpenBMB/MiniCPM5-2B}
}
```
