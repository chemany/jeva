# Running jev-ultrafast with jeva

[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) is the
reference agent this project learned its architecture from: a browser layer that reads the DOM into
an indexed element table, a loop with staleness and repeat guards, and a decision model that picks
one operation and one target.

Its decision model is TypeSafe's hosted System One API plus a second language model for field values.
Both are replaced here by one local 2B model. **`agent.py`, `browser.py` and `snapshot.js` are used
unmodified** — only `choose()` and `field_text()` are swapped.

```bash
# one-time: a jev-ultrafast clone
git clone https://github.com/browser-use/jev-ultrafast /tmp/jev-ultrafast
export JEVA_JEV_REPO=/tmp/jev-ultrafast          # default /tmp/jev-ultrafast

# needs a running jeva server (pm2 jeva-serve on 127.0.0.1:8020)
python integrations/jev-ultrafast/run.py \
  "Find one-way flights from Zurich to London on September 20, 2026, for two adults in Business." \
  "http://127.0.0.1:8899/flights.html"
```

Measured: **4.4 s** for the flight fixture (7 operations, results page reached), **5.2 s** for the
live httpbin form.

## browser-harness without the approval prompt

jev drives the user's own Chrome through `browser-harness`, which asks for remote debugging to be
allowed by hand — once per connection, which makes unattended use impossible. The same library also
honours `BU_CDP_WS`, so `run.py` launches a throwaway headless Chrome and points the harness at it.
No prompt, no borrowed session.

## What is deliberately not identical

**`probabilities` is a one-hot, not a distribution.** jeva emits one action; it has no probability
head. jev's own validator requires the distribution to sum to 1.0, so the chosen action is reported
as `1.0`. Nothing downstream reads it as calibrated and it should not be treated as such — a
measured check found correct answers carry a median top-1 of 48% while wrong ones carry 80%, so a
confidence signal here is worse than useless.

**The prompt is jeva's own format, not TypeSafe's questions object.** jeva was fine-tuned on a
compact indexed table (`[3] textbox: Where from? "Zurich" {ops: TYPE_TEXT,CLICK}`). Handed TypeSafe's
nested `{type, criteria, instructions}` shape it tends to echo the schema back: measured 9% on a task
where the flat form scored 82%. `build_prompt` renders jev's questions back into that table.

**The second language model is gone.** jev calls an LLM to produce the value for `TYPE_TEXT`; jeva
returns the value in the same decision, so `field_text` just hands it back.

## Limits

* **No `READ`.** jev's action space is `CLICK / TYPE_TEXT / SELECT / SCROLL / WAIT / DONE / BLOCKED`.
  Asking it to *report* what a page says is not an operation it has, so this path drives pages rather
  than answering questions about them. Adding that is a change to jev's own question set, not to the
  model.
* **Not benchmarked.** The two runs above are smoke tests. The guarded, verified numbers for jeva's
  decisions live in `evals/` and are measured through this repo's own harness, not through jev's.
* jev's `SCROLL` operation exists but jeva was never trained to emit it, so a target below the fold
  is out of reach on this path too.
