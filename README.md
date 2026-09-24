# jeva

A 2B browser-agent decision model you can train and run yourself.

<p>
  <a href="#license"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="#models"><img alt="Weights: 2B · merged · GGUF" src="https://img.shields.io/badge/WEIGHTS-2B%20%C2%B7%20merged%20%C2%B7%20GGUF-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="evals/"><img alt="Eval suites: frozen" src="https://img.shields.io/badge/EVAL%20SUITES-frozen-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="docs/pipeline.md"><img alt="Data pipeline" src="https://img.shields.io/badge/DATA-zero%20human%20labels-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="docs/model-card.md"><img alt="Model card" src="https://img.shields.io/badge/MODEL%20CARD-docs-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

<p><a href="README.md"><b>English</b></a> · <a href="README.zh-CN.md">简体中文</a></p>

jeva turns a page observation plus a goal into **exactly one action** — a compact JSON object, no prose:

```json
{"operation": "CLICK", "target": "2", "text": ""}
```

It is a fine-tune of [`openbmb/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B),
trained on trajectories collected by driving a **real Chrome** with a deterministic solver. The
weights are **merged into the base** — there is no adapter to pair up, and nothing to install
beyond your usual inference stack.

## Highlights

- **One decision, one call, ~230 ms.** 2B parameters, 1.6 GB as a Q4_K_M GGUF, running under llama.cpp on a single V100.
- **Typed action space.** `CLICK` · `TYPE_TEXT` · `SELECT` · `WAIT` · `DONE` · `BLOCKED`, targeting indices from the observation you supplied.
- **No invented selectors.** The model returns an index; your executor resolves it against the same snapshot. It never emits CSS, coordinates, or JavaScript.
- **100% task success on the frozen suites**, including a site whose label vocabulary and layout never appear in training.
- **Zero human labels.** 10,686 training trajectories from 2,600 tasks, collected in ~30 minutes; training is one LoRA pass, ~75 minutes on one V100.
- **Everything included**: the collector, the training script, the merge/quantise pipeline, the three test sites, and the frozen eval results.

## Install

Neither route needs a clone, and neither installs a second dependency.

```bash
pip install jeva        # client + CLI, ~19 KB, zero dependencies
jeva download          # Q4_K_M weights (1.6 GB) into ~/.cache/jeva, with sha256
jeva demo              # start a server if it is not running, make one decision
jeva serve             # leave the server running on 127.0.0.1:8020
jeva check             # report what is missing (llama-server, weights, port)
jeva run <url> <goal>  # drive a real Chrome towards the goal, one step at a time
```

Only the client ships in the wheel — weights are fetched on demand, because a 1.6 GB GGUF has no
business inside a Python distribution. Point `JEVA_BASE_URL` (or `--url`) at your own mirror if you
prefer to download elsewhere.

Until the PyPI release is live, install straight from Git:

```bash
pip install "git+https://github.com/chemany/jeva"
```

## Quick start

By hand — grab a GGUF from [ModelScope](https://modelscope.cn/models/imjasonli/jeva) and serve it:

```bash
llama-server -m MiniCPM5-2B-WebDecider-Q4_K_M.gguf \
  --alias jeva --port 8020 -ngl 99 -c 4096 --jinja
```

Then ask it for a decision. jeva needs the observation in the format described under
[State format](#state-format); `jeva` renders it for you:

```python
from jeva import Jeva, Page, Element, Option

page = Page(
    title="Flight Search", url="https://example.com/flights",
    text="Search flights. Round trip / One way.",
    elements=[
        Element(index="1", role="radio", label="Round trip", value="round",
                checked="true", operations=["CLICK"]),
        Element(index="2", role="radio", label="One way", value="oneway",
                checked="false", operations=["CLICK"]),
        Element(index="3", role="textbox", label="Where from?", value="Zurich",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="4", role="textbox", label="Where to?", value="London",
                operations=["TYPE_TEXT", "CLICK"]),
        Element(index="7", role="button", label="Search", operations=["CLICK"]),
    ],
)

jeva = Jeva("http://127.0.0.1:8020/v1", model="jeva")
action = jeva.decide(page, "Find one-way flights from Zurich to London on September 20, 2026, "
                           "for one adult in Economy.")
print(action.as_dict())
# {'operation': 'CLICK', 'target': '2', 'text': ''}      ← the radio the goal needs
```

A radio that is already in the requested state is left alone; an empty text field is filled; a
challenge page returns `BLOCKED`. The four rules that decide *when not to act* are baked into the
model, not into your prompt.

Try it without installing anything else:

```bash
python examples/quickstart.py            # uses the built-in demo observation
```

## Driving a browser

`Jeva` decides; `jeva.Agent` runs the loop around it. One command drives a real Chrome:

```bash
jeva run "https://example.com/order" \
  "Order one large pizza with mushrooms and cheese, for delivery at 12:30, then place the order." \
  --screenshots ./steps --json
```

```python
from jeva import Agent

with Agent("https://example.com/order", "Order one large pizza ...") as agent:
    result = agent.run()
print(result.status, result.url, len(result.steps))
```

### Logged-in sites: sign in once, then stay headless

The default is a **headless browser with no permission prompts, no window and no `DISPLAY`**. Login
state comes from a persistent profile instead of borrowing your daily browser:

```bash
jeva login --url https://example.com/login          # one time: a window opens, you sign in
jeva run "https://example.com/orders" "Download last month's invoices" \
  --profile-dir ~/.local/share/jeva/chrome          # headless from here on
```

Chrome's "allow remote debugging" switch is deliberately *not* the mechanism: that prompt has to be
approved per connection, which is the opposite of unattended. A persistent profile needs no
approval, and it is a separate throwaway profile -- your own bookmarks, passwords and tabs are
never touched. Two caveats: let Chrome close normally (Chrome flushes cookies in batches, so a
killed process loses them, which `jeva login` handles), and the site must set a cookie with an
expiry -- a pure session cookie is not persisted by any browser.

The loop exists because a decision is only safe to execute if something checks it:

| Guard | What it prevents |
|---|---|
| the target index is resolved against the observation the decision came from | a decision acting on an element that appeared later |
| the decision is consumed once | a retry clicking twice |
| freshness is re-checked before acting, and before accepting `DONE` / `BLOCKED` | clicking into a page that moved, or "done" against a stale page |
| three actions that change nothing end the run | a confused decider spending the whole step budget |

`status` is the model's claim. The result carries the final URL, the page text and the steps, so
your code can verify -- a submitted form usually puts every value in the URL. **Do not treat
`status == "done"` as proof.**

Measured on the flight fixture: 7 steps in 4.2 s, screenshots included (one V100, Q4_K_M).

## Models

| Model | Base | Parameters | Format | Size | Suite accuracy | Latency / decision |
|---|---|---|---|---|---|---|
| **jeva** (this release) | MiniCPM5-2B | 2.5B (2.0B non-embed) | merged, HF | 5.0 GB | **100%** | 1500 ms (naive HF) |
| jeva · GGUF F16 | ” | ” | llama.cpp | 5.04 GB | 100% | ~250 ms |
| **jeva · GGUF Q8_0** | ” | ” | llama.cpp | 2.68 GB | **100%** | **283 ms** |
| **jeva · GGUF Q4_K_M** | ” | ” | llama.cpp | **1.56 GB** | **100%** | **229 ms** |
| MiniCPM5-2B (untrained) | — | 2.5B | llama.cpp | 1.56 GB | 10% (100 tasks) | 147 ms |
| Bonsai-27B (zero-shot) | Qwen3.8-27B | 27B | GGUF q4_0 | 14 GB | 92.5% (40 tasks) | 1769 ms |

The GGUF variants live under `gguf/` on
[ModelScope](https://modelscope.cn/models/imjasonli/jeva); the merged transformer weights sit at the
root of the same repository. `jeva download` fetches them from there.

Quantisation costs nothing here: all three GGUF variants score **100%** on the same suites.
The untrained base on the same prompt scores **10%** — it loops on already-checked radios, uses
`CLICK` where `TYPE_TEXT` is required, and invents element indices 18 times in 100 tasks.

> **Not an apples-to-apples comparison.** jeva is *trained for this action space*; Bonsai-27B and
> the untrained base are zero-shot on the same prompt. The point of the table is what 11 minutes of
> self-generated supervision buys a 2B model — not that a 2B beats a 27B at being a general model.

## Benchmarks

![Accuracy by task type for jeva, the untrained base, and Bonsai-27B](docs/comparison.png)

Task success rate on randomised suites. Each task is executed in **real Chrome** and verified
against the resulting page state — the submitted values are read back from the URL, so a task only
counts when the browser really ended up in the right state.

| Task type | MiniCPM5-2B (base) | Bonsai-27B (zero-shot) | **jeva** |
|---|---|---|---|
| Blocked page (CAPTCHA / rate limit) | 9 / 9 · 100% | 5 / 5 · 100% | **9 / 9 · 100%** |
| Flight search (radio + 3 fields + submit) | 0 / 21 · 0% | 7 / 7 · 100% | **21 / 21 · 100%** |
| Hotel filters (2 selects + checkbox) | 0 / 26 · 0% | 12 / 12 · 100% | **26 / 26 · 100%** |
| Contact form (3 fields + consent) | 0 / 6 · 0% | 0 / 1 · 0% | **6 / 6 · 100%** |
| Autocomplete (type → pick option) | 1 / 7 · 14% | 4 / 4 · 100% | **7 / 7 · 100%** |
| Order form (many requirements + clock + note) | 0 / 31 · 0% | 9 / 11 · 82% | **31 / 31 · 100%** |
| **Overall** | **10%** (100 tasks) | **92.5%** (40 tasks) | **100%** (100 tasks) |

The Bonsai column rests on 40 tasks and is indicative only; it failed the order form by stopping
without submitting, and failed the contact form by inventing a target.

### Subsets the suite averages away

A family with 31 tasks can lose three of them without the overall number moving. These subsets keep
only the goals whose page pre-fills nothing, so every requirement has to be handled by the decider
itself ([`evals/subset_eval.py`](evals/subset_eval.py), 20 tasks each):

| Subset | What the goal leaves to the model | jeva |
|---|---|---|
| `when` | a delivery time, written as `12:30` or `9:00 AM` | **20 / 20** |
| `note` | a free-text note phrased four different ways | **20 / 20** |
| `contact` | name, phone and email given as a bare list | **20 / 20** |
| `compound` | all three at once, plus toppings — an ordinary order form | **20 / 20** |

### Real-site spot check

jeva was trained only on the synthetic fixtures in `site/`. Pointed at a live third-party form it
had never seen — [`httpbin.org/forms/post`](https://httpbin.org/forms/post), a real page with three
text fields, a radio group, four checkboxes and a native time input — it fills every field the goal
names and submits. Four runs:

| | Core fields (name / phone / email / size) | Both toppings asked for | Time of day | Submitted + correct `DONE` |
|---|---|---|---|---|
| 4 runs | **4 / 4** | **4 / 4** | **4 / 4** | **4 / 4** |

Every run derives the right values from the goal (`Jason Zhang`, `555-0100`, `jason@example.com`,
`size: large`), enters `12:30`, ticks `Mushroom` **and** `Extra Cheese` — for a goal that says
"with mushrooms and cheese", so the label has to be matched by meaning, not by string — and stops
only after the browser really submitted.

Reproduce: `python evals/real_site_test.py` (needs network access).

This check is the reason the last three training rounds exist. It found that the decider could not
see `<input type=time>` at all, could not type into one once it could see it, and was never asked to
fill a field whose label does not appear in the goal. Each is now a task family in the collector and
a subset in `evals/subset_eval.py`.

### Held-out sites

Three sites with **different label vocabularies and layouts** were used. `site3` is the strict
holdout: it was never used for training, and its labels are entirely different
(`Origin` / `Destination` / `When` / `Cabin class` / `Guests` / `Round-trip` / `One-way` / `Submit`).

| Suite | Tasks | jeva |
|---|---|---|
| `site1` (in training distribution) | 100 | **100%** |
| `site2` (in training distribution, extra nav distractors) | 40 | **100%** |
| **`site3` (never trained, new labels)** | 40 | **100%** |
| live `httpbin.org/forms/post` (never trained, real markup) | 4 runs | see [above](#real-site-spot-check) |

### What moved the number

| Change | Effect |
|---|---|
| LoRA SFT on verified trajectories | 10% → **100%** (overall) |
| Multi-site collection (including distractor links that shift element indices) | site2 flight **19% → 100%** |
| Task family for goals that need several boxes at once | order form **0% → 100%** |
| Task family for a label the goal does not spell out | "cheese" now finds `Extra Cheese` |
| Task family for a clock field and for a free-text note | `when` / `note` / `contact` subsets **100%** |
| Removing the leftover index duplication in the prompt | no change (verified equivalent, 100% → 100%) |
| Merging the adapter + Q4_K_M quantisation | no change (100% → 100%), 5.0 GB → 1.6 GB |
| Serving path, same weights and prompt | llama.cpp **0.23 s** · vLLM 0.95 s · naive HF generate 8.4 s |

Training data scale and ablations are in [docs/pipeline.md](docs/pipeline.md); the eval JSONs are
frozen in [`evals/results/`](evals/results/).

**What is in this repo:** the runtime package, the collector and trainer, the three test sites, the
frozen eval results, and a 120-record [sample](evals/data/sample.jsonl) of the training set so you
can inspect the format. **What is not:** the full 10,686-record set (~72 MB) and the weights — the
former is regenerated by the collector in ~30 minutes, the latter is on ModelScope.
The ModelScope/Hugging Face facing model card is [docs/hf-model-card.md](docs/hf-model-card.md).

## State format

jeva reads **compact plain text**, one line per interactive element. Five rules, each measured:

```
Page: Flight Search  (https://example.com/flights)
Goal: Find one-way flights from Zurich to London on September 20, 2026, for one adult in Economy.

[1] radio: Round trip "round" checked=true {ops: CLICK}
[2] radio: One way "oneway" checked=false {ops: CLICK}
[3] textbox: Where from? "Zurich" {ops: TYPE_TEXT,CLICK}
[4] textbox: Where to? "London" {ops: TYPE_TEXT,CLICK}
[5] textbox: Departure not set {ops: TYPE_TEXT,CLICK}
[6] combobox: Cabin "Economy" {ops: SELECT} options: [6:1] Premium economy | [6:2] Business
[7] button: Search {ops: CLICK}

Page text: Search flights. Round trip / One way. Where from? Where to? Departure.
Recent actions: CLICK Round trip
```

1. **Never render an empty string for a missing value** — write `not set`. An empty value is read as *present*.
2. **Always emit `{ops: ...}`** — the only structural signal separating a select-only dropdown from a typeable textbox.
3. For dropdowns, list only the **unselected** options, keyed `<element>:<option>`.
4. Render `checked=true/false` for radios and checkboxes.
5. **Compact plain text.** A JSON blob degrades accuracy substantially.

`jeva.render.render_state()` produces exactly this from `Page`/`Element`/`Option` objects, and
`jeva.prompt.build_prompt()` wraps it with the operation descriptions and target lists the model
expects. Both are the same code used during training.

> **The system prompt is part of the model.** Changing the wording measurably degrades output —
> see the ablation in [docs/pipeline.md](docs/pipeline.md). If you need a different contract,
> fine-tune rather than reword.

## API

jeva is a plain OpenAI-compatible chat endpoint. There is nothing to install server-side.

```bash
curl -s localhost:8020/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "jeva",
  "max_tokens": 64,
  "temperature": 0.0,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [
    {"role": "system", "content": "<jeva/prompt.py SYSTEM, verbatim>"},
    {"role": "user",   "content": "<state>\n\nAvailable operations:\n…"}
  ]}'
```

`chat_template_kwargs.enable_thinking=false` is required: MiniCPM5 is a reasoning model, and with
thinking on the trace consumes the token budget and `content` returns empty.

### Python

```python
from jeva import Jeva, Page, Element

jeva = Jeva("http://127.0.0.1:8020/v1", model="jeva")
action = jeva.decide(page, goal, history=["CLICK Round trip"])
```

| Symbol | Purpose |
|---|---|
| `Jeva(base_url, model)` | client; `.decide(page, goal, history)` → `Action`, `.last_latency_ms` |
| `Page(elements, url, title, text)` | the observation |
| `Element(index, role, label, value, operations, options, checked)` | one control |
| `Action(operation, target, text)` | one decision; `.terminal` is true for `DONE`/`BLOCKED` |
| `jeva.resolve(page, action)` | look the target index up in the observation you sent |
| `jeva.render_state` / `jeva.build_prompt` | the exact training-time rendering |

## How it works

A goal-following loop is: observe → decide → execute → observe. jeva is only the *decide* step.

```
browser ──snapshot──▶ Page(elements, text) ──▶ jeva ──▶ Action(operation, target, text)
   ▲                                                        │
   └──────────────── execute the target ────────────────────┘
```

The model was trained to respect the rules a browser agent actually needs — the ones that are
easy to state and hard to get right:

- do not re-click a radio that is already in the requested state,
- do not re-type into a field that already holds the required value,
- use `TYPE_TEXT` (not `CLICK`) to enter text, `SELECT` for dropdowns,
- `DONE` only on visible evidence, `BLOCKED` when a challenge stops progress.

Everything else — element identity, selectors, freshness checks, retries — stays in the executor.

### Training data

No human labelling and no teacher model. A deterministic solver drives real Chrome and records
`(state, action)`; a trajectory is kept only when the environment verifies the task succeeded.

```
real Chrome ──snapshot──▶ DOM (elements + page text)
     ▲                          │
     │                          ▼
 execute ◀── deterministic solver (sees a privileged "already satisfied / not yet done" line)
                                │
                                ▼
              record (state, action)  ← the recording contains NO hint
                                │
              keep only if the environment verifies success
                                ▼
                     LoRA SFT → merge → GGUF
```

Yield: **2,600 tasks → 10,686 verified examples in ~30 minutes**. Only trajectories the
environment verifies are kept, so a task that fails simply contributes nothing. The solver sees a privileged
summary of what is already satisfied; the student does not, so it has to learn state tracking
from the state alone. Full detail and ablations: [docs/pipeline.md](docs/pipeline.md).

## Training your own

```bash
# 1. three test sites (different label vocabularies / layouts)
python -m http.server 8899 --directory site/site1 &
python -m http.server 8898 --directory site/site2 &
python -m http.server 8897 --directory site/site3 &

# 2. collect verified trajectories (~11 min for 9k examples)
SITES="http://127.0.0.1:8899,http://127.0.0.1:8898" python scripts/collect.py 2200 evals/data/sft.jsonl

# 3. LoRA SFT (1 epoch, ~70 min on one V100)
EPOCHS=1 BATCH=4 ACCUM=4 MAXLEN=1152 \
  SFT_DATA=evals/data/sft.jsonl OUT_DIR=runs/lora python scripts/train_lora.py

# 4. merge into the base, then quantise
python scripts/merge_lora.py
python scripts/convert_gguf.sh
```

The collector is site-agnostic: point it at your own pages and extend the alias table, and the
same pipeline produces training data for that site. See [docs/pipeline.md](docs/pipeline.md).

## Limits

- **The model decides; the loop acts.** `Jeva.decide()` returns an index into the observation *you*
  supplied and nothing else. `jeva.Agent` (or `jeva run`) supplies the loop, the executor and the
  guards -- but its `DONE` is still a claim, not proof.
  Your executor must resolve it, check the page is still fresh, and handle failure.
- **Narrow action space.** No `SCROLL`, no file uploads, no multi-step dropdown widgets. If your
  agent needs those, extend the action space and retrain.
- **Form-shaped tasks.** It was trained on search/filter/form/autocomplete flows. Treat `DONE` and
  `BLOCKED` as proposals that a verifier should confirm, especially off-distribution.
- **Prompt-sensitive.** The system prompt, the operation descriptions and the state layout are
  part of the model. Reproduce them exactly (`jeva.prompt` / `jeva.render` do).
- **Not a general assistant.** It is a decision head, not a chat model.
- **A value the page refuses is not detected.** Set a `<input type=time min="11:00">` to 09:00 and
  the browser blocks the form; the model keeps clicking submit instead of reporting `BLOCKED`. It
  cannot see `min`/`max` in the state, and nothing in training failed that way. Give constrained
  fields a value inside their range, or have your executor surface the validation message.
- **Termination is only as good as its training families.** It now stops correctly on pages with
  no interactive elements and on goals it cannot satisfy from the current page, but a *new* shape
  of dead end is not covered by construction.
- Evaluated on three synthetic sites plus one live third-party form (real Chrome, real DOM);
  this is not a production-website benchmark.

## License

**Apache-2.0**, same as the base model. jeva is a fine-tuned derivative of
[`openbmb/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B); the name follows the
MiniCPM derivative convention, and the original work is credited below.

```bibtex
@misc{minicpm5,
  title  = {MiniCPM5-2B},
  author = {OpenBMB},
  year   = {2026},
  url    = {https://modelscope.cn/models/OpenBMB/MiniCPM5-2B}
}
```
