# Data and training pipeline

jeva's supervision is generated, not labelled. This document is the full recipe, the ablations
that fixed real failures, and the traps we hit — so you can reproduce it or point it at your own
site.

## 1. Why not just prompt a bigger model

The task is *decide the next browser action*. A general model can do this zero-shot, and we
measured it: Bonsai-27B manages roughly 60% on the same suites, with the failures concentrated in
the same place — it re-clicks controls that are already in the requested state. Offline
single-step accuracy is not the bottleneck either; a 421M classifier scored 84% on hand-written
decision states while still failing in the loop.

## 2. The trick: a deterministic solver is the teacher

We already had a rule-based policy that can decide the next action for a form-shaped task
(read the goal, compare against the current DOM, act on the first unmet requirement). Driving
real Chrome with that policy produces **real observations with guaranteed-correct actions**:

```
real Chrome ──snapshot──▶ DOM (elements + page text)
     ▲                          │
     │                          ▼
 execute ◀── solver (goal spec vs DOM → next unmet requirement)
                                │
                                ▼
              record (state, action)  ← no hint in the recording
                                │
              keep the trajectory only if the environment verifies success
                                ▼
                        LoRA SFT → merge → GGUF
```

No LLM is in the loop, and nothing is hand-labelled. Two details matter:

**Privileged information.** The solver sees a line like
`Already satisfied: trip type, origin. Not yet done: destination, departure.`
The student never sees it — only `(state, action)` is recorded. The student therefore has to learn
state tracking itself, which is exactly the capability the zero-shot baselines lack. (Feeding that
line to a 27B at inference raised it from 3/5 to 5/5 on our hand-written states, so it is a real
capability, not a formatting quirk.)

**The environment is the verifier.** A trajectory is kept only if the task really succeeded —
the submitted values are read back from the resulting URL. Failed trajectories are discarded, so
no wrong action ever becomes supervision.

## 3. Yield and cost

| Stage | Scale | Wall clock |
|---|---|---|
| Collect (real Chrome, replay) | 2,600 tasks / **10,686 examples** (~87% of generated tasks pass verification) | **~30 min** |
| LoRA SFT, 1 epoch, 1× V100 | 10,686 examples | ~75 min |
| Merge + GGUF + quantise | — | seconds |
| **Human labelling** | — | **none** |

Loss reaches ~6e-6 by step 400: the task is narrow and one epoch is enough. More epochs did
nothing.

## 4. Ablations

| Change | Effect |
|---|---|
| **Base → LoRA SFT** | 10% → 100% overall |
| **Single-site → multi-site collection** | site2 flight **19% → 100%** |
| **Adding the multi-requirement goal family** | order form **0% → 100%** |
| **Adding the clock-field and free-text-note families** | `when` / `note` / `contact` **100%** |
| Clean vs duplicated index in the prompt | equivalent (100% both ways) |
| Merge + Q4_K_M quantisation | equivalent (100%), 5.0 GB → 1.6 GB |
| Serving path (same weights/prompt) | llama.cpp 0.23 s · vLLM 0.95 s · naive HF 8.4 s |

### Task families are found by breaking the model on a real page

Three rounds of this project came from one live third-party form
([`httpbin.org/forms/post`](https://github.com/chemany/jeva#real-site-spot-check)) rather than from
any synthetic suite. Each failure was a *goal shape* the collector never produced, not a capacity
limit:

| What the live page exposed | What was missing |
|---|---|
| The goal named a delivery time and the run looped on an unrelated field | the observation had no role for `<input type=time>`, so the control was invisible; once visible, `Input.insertText` is silently ignored by date/time inputs |
| The goal asked for "mushrooms **and cheese**" and only `Mushroom` got ticked | no family where the goal's word differs from the control's label |
| The goal asked for several checkboxes and the run submitted after one | no family with more than one requirement of the same kind |
| The goal listed contact details as "Name X, phone Y, email Z" and they were skipped | no family where several different field types are required at once, in that phrasing |

Each became a generator branch, a page, and a subset in `evals/subset_eval.py`, because a
forty-task random suite hides a gap that costs one task per family.

### The multi-site result is the interesting one

v1 was trained on `site1` only. It scored 100% on `site3`, whose *labels* are completely
different (`Origin` / `Destination` / `Round-trip`), so label vocabulary was not the problem.
It scored **19%** on `site2`, whose only differences were **three extra navigation links** that
shifted every element index by three.

So: **randomise layout and distractors, not just synonyms.** The model had learned a positional
pattern, and the layout shift broke it. v2, trained on both layouts, scores 100% on all three
sites.

## 5. Traps we hit (all fixed)

| Trap | Symptom | Fix |
|---|---|---|
| `pkill -f "llama-server"` | killed the production LLM, not just the test server | always scope by port: `pkill -f "llama-server.*8020"` |
| One-hot with a single candidate | the agent's validator rejected the response (`|Σp − 1| > 0.02`) | a single candidate must get probability **1.0** |
| `warmup_ratio` | gone in transformers 5.x | use `warmup_steps` |
| `MAXLEN=2048` while prompts are 552–1066 tokens | 55% of compute spent on padding | set the cap from the measured distribution (1152) |
| Solver matched radio labels by hard-coded strings | could not recognise `Return` / `Single`, flight success capped at 73% | match through the alias table |
| Test site submitted to `/site2/...` while served at the root | 404, form silently did nothing | root-relative action URLs |
| Disabling gradient checkpointing to go faster | OOM on a 16 GB V100 | keep it on |
| Prompt duplicated the option index (`[1] [1] label`) | cosmetic; verified equivalent after fixing | strip the prefix in the criteria flattening |

## 6. Extending it to your own site

The collector is site-agnostic. To add a site:

1. Put the pages under `site/`, and make the *initial state* settable from URL parameters — that
   is what lets you mint thousands of task instances from one page.
2. Add an entry to the task generators (goal, spec) and the alias table (how each field might be
   labelled).
3. Extend the solver with the checks for the new task type; the pattern is
   `(field, label, predicate, action)` per requirement.
4. Extend `verify()` so the environment can adjudicate success.
5. Collect, train, and evaluate on a *third*, unseen site — that number is the only one that means
   anything.

Keep at least one site out of training. In our runs the in-distribution number was 100% long
before generalisation was real.
