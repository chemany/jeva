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

- **One decision, one call, ~250 ms.** 2B parameters, 1.6 GB as a Q4_K_M GGUF, running under llama.cpp on a single V100.
- **Typed action space.** `CLICK` · `TYPE_TEXT` · `SELECT` · `WAIT` · `DONE` · `BLOCKED`, targeting indices from the observation you supplied.
- **No invented selectors.** The model returns an index; your executor resolves it against the same snapshot. It never emits CSS, coordinates, or JavaScript.
- **100% task success on the frozen suites**, including a site whose label vocabulary and layout never appear in training.
- **Zero human labels.** 9,357 training trajectories from 2,200 tasks, collected in ~11 minutes; training is one LoRA pass, ~70 minutes on one V100.
- **Everything included**: the collector, the training script, the merge/quantise pipeline, the three test sites, and the frozen eval results.

## Quick start

Grab a GGUF from the [releases page](https://github.com/chemany/jeva/releases) (or `models/` if you built it yourself) and serve it:

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

## Models

| Model | Base | Parameters | Format | Size | Suite accuracy | Latency / decision |
|---|---|---|---|---|---|---|
| **jeva** (this release) | MiniCPM5-2B | 2.5B (2.0B non-embed) | merged, HF | 5.0 GB | **100%** | 1500 ms (naive HF) |
| jeva · GGUF F16 | ” | ” | llama.cpp | 5.04 GB | 100% | ~250 ms |
| **jeva · GGUF Q8_0** | ” | ” | llama.cpp | 2.68 GB | **100%** | **283 ms** |
| **jeva · GGUF Q4_K_M** | ” | ” | llama.cpp | **1.56 GB** | **100%** | **276 ms** |
| MiniCPM5-2B (untrained) | — | 2.5B | — | 5.0 GB | 8% | 237 ms |
| Bonsai-27B (zero-shot) | Qwen3.8-27B | 27B | GGUF | 14 GB | 70% (10 tasks) | 6800 ms |

All variants are attached to the [release](https://github.com/chemany/jeva/releases); the merged transformer weights are
mirrored on ModelScope and Hugging Face, linked from that page.

Quantisation costs nothing here: all three GGUF variants score **100%** on the same suites.
The untrained base on the same prompt scores **8%** — it loops on already-checked radios, uses
`CLICK` where `TYPE_TEXT` is required, and never emits `BLOCKED`.

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
| Blocked page (CAPTCHA / rate limit) | 0 / 13 · 0% | 1 / 1 · 100% | **100%** |
| Flight search (radio + 3 fields + submit) | 0 / 32 · 0% | 2 / 4 · 50% | **100%** |
| Hotel filters (2 selects + checkbox) | 0 / 28 · 0% | 2 / 3 · 67% | **100%** |
| Contact form (3 fields + consent) | 2 / 13 · 15% | 2 / 2 · 100% | **100%** |
| Autocomplete (type → pick option) | 6 / 14 · 43% | — | **100%** |
| **Overall** | **8%** (100 tasks) | **70%** (10 tasks) | **100%** (100 tasks) |

The Bonsai column rests on 10 tasks and is indicative only. It failed the way the untrained base
does: repeatedly clicking `Round trip` when the goal asked for one way.

### Held-out sites

Three sites with **different label vocabularies and layouts** were used. `site3` is the strict
holdout: it was never used for training, and its labels are entirely different
(`Origin` / `Destination` / `When` / `Cabin class` / `Guests` / `Round-trip` / `One-way` / `Submit`).

| Suite | Tasks | jeva |
|---|---|---|
| `site1` (in training distribution) | 40 | **100%** |
| `site2` (in training distribution, extra nav distractors) | 100 | **100%** |
| **`site3` (never trained, new labels)** | 40 | **100%** |

### What moved the number

| Change | Effect |
|---|---|
| LoRA SFT on 9,357 verified trajectories | 8% → **100%** (overall) |
| Multi-site collection (including distractor links that shift element indices) | site2 flight **19% → 100%** |
| Removing the leftover index duplication in the prompt | no change (verified equivalent, 100% → 100%) |
| Merging the adapter + Q4_K_M quantisation | no change (100% → 100%), 5.0 GB → 1.6 GB |
| Serving path, same weights and prompt | llama.cpp **0.25 s** · vLLM 0.95 s · naive HF generate 8.4 s |

Training data scale and ablations are in [docs/pipeline.md](docs/pipeline.md); the eval JSONs are
frozen in [`evals/results/`](evals/results/).

**What is in this repo:** the runtime package, the collector and trainer, the three test sites, the
frozen eval results, and a 120-record [sample](evals/data/sample.jsonl) of the training set so you
can inspect the format. **What is not:** the full 9,357-record set (~64 MB) and the weights — the
former is regenerated by the collector in ~11 minutes, the latter is attached to the releases.
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

Yield: **2,200 tasks → 9,357 verified examples in ~11 minutes**. Only trajectories the
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

- **It decides; it does not act.** The output is an index into the observation *you* supplied.
  Your executor must resolve it, check the page is still fresh, and handle failure.
- **Narrow action space.** No `SCROLL`, no file uploads, no multi-step dropdown widgets. If your
  agent needs those, extend the action space and retrain.
- **Form-shaped tasks.** It was trained on search/filter/form/autocomplete flows. Treat `DONE` and
  `BLOCKED` as proposals that a verifier should confirm, especially off-distribution.
- **Prompt-sensitive.** The system prompt, the operation descriptions and the state layout are
  part of the model. Reproduce them exactly (`jeva.prompt` / `jeva.render` do).
- **Not a general assistant.** It is a decision head, not a chat model.
- Evaluated on three synthetic sites (real Chrome, real DOM), not on production websites.

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
