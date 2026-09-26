# Content-selection training

The pipeline that taught jeva to answer "which block on this page is the first news article",
so the answer is the model's judgement rather than a per-site rule about font sizes and columns.

Scripts live here; **data does not** -- corpora, adapters and checkpoints stay in
`/root/code/jeva-content/` (`JEVA_DATA=...` overrides). `collect3`/`collect4` are history, kept
because the difference between them and `collect8` is the point of the section below.


Teaching jeva to answer "which block on this page is the first news article", so the answer is the
model's judgement instead of a per-site rule about font sizes and column positions.

## Method

* Deterministic half: enumerate every visible text block in reading order with its layout facts
  (`host`, size, column, row). No filtering -- which block is content is the thing being taught.
* Teacher: the local 27B, **offline only**. At run time jeva answers alone.
* Labels: the teacher marks which blocks are article links.
* Training: LoRA on top of the shipped model, mixed with action examples as rehearsal.

## A methodology bug worth remembering

The first four collection rounds reloaded each homepage N times, on the assumption that a news
homepage rotates its content. It does not -- it is static for hours. Measured afterwards:

    raw7: 150 collected pages from 17 URLs contained 17 distinct contents
    "56-page holdout"                 actually 8 distinct samples

Every accuracy figure from those rounds was computed on a handful of samples, and the differences
between them (86% vs 43%) were differences between *three and six sites*, not between 24 and 56
pages. `collect8.py` walks each site's own section links instead: 123 pages produced 118 distinct
contents.

## Results on 30 genuinely distinct held-out pages, 10 sites never trained on

| model | hit |
|---|---|
| v7  | 18/30 = 60% |
| v9  | **28/30 = 93%** |
| v10 | 25/30 = 83% |

v10 is a regression and is not shipped: trained on a corpus that was ~90% duplicates, it memorised
the training set (99%) and generalised worse.

## The prompt-shape bug this round found

`build_prompt`'s last line -- the example -- was hardcoded to `CLICK` even on pages whose only
operation is `READ`. Found by reading the model's logits rather than its answer: on a failed page the
model's *first* choice was correct (block 2, 39.9%) while the text it produced was not. Fixing the
example took the content task from 71% to 86% on the then-current holdout, with no weight change.

Two things measured and ruled out:

* **confidence cannot drive a retry** -- correct answers carry a median top-1 probability of 48.2%,
  wrong ones 79.7%; the model is more confident when wrong.
* **self-verification is too phrasing-sensitive to gate on** -- asked directly, "is block N an
  article?" is right 7/7 on a failing page, but as a loop it stops on the wrong block, and rewording
  flips real articles to "not an article".

## Files

| file | what |
|---|---|
| `collect*.py` | collection rounds; `collect8.py` is the only one whose corpus should be used |
| `build_dataset.py` | pages -> training examples, plus prefix and dropout augmentation |
| `mix.py` | content + action rehearsal into one training file |
| `eval_content.py` | accuracy on a holdout; run it against jeva and the teacher for reference points |
| `raw8.jsonl` | the correctly collected corpus (118 distinct contents) |

## Homepage benchmark (30 homepages, verified by hand)

`home_bench.jsonl` + `bench/*.png`: one homepage per site, the model's answer, and my verdict after
looking at each screenshot. Built because every earlier number was measured on *section pages*
(which is what crawling a site's nav links produces) while the task people actually ask for is a
homepage.

| | |
|---|---|
| correct | 17 / 30 = **57%** |
| wrong | 7 |
| picked a top ticker instead of the lead headline | 2 |
| homepage has no single lead item (carousel / topic banner) | 4 |
| **correct among pages with a clear answer** | **17 / 26 = 65%** |

For contrast: **93%** on 30 section pages from 10 unseen sites (`raw8.jsonl`). Homepages are a
harder, different distribution -- they add site logos, utility links (English / 设为首页), column
names, keyword rows, and carousels, none of which appear on a section page.

Failure modes, all observed:

* the site's own name (`东方财富`)
* a utility link (`English`)
* a column name (`光华锐评`)
* a later list item instead of the first (`第③条` where `第①条` was the lead)
* a top scrolling ticker where the lead headline was meant (`新浪财经`, `观察者网`)

The last one is a definition, not a mistake: a ticker headline *is* a news headline. Which is also
why four homepages cannot be scored at all -- an image carousel or a topic banner has no single
"first news item". The task needs the definition pinned before the numbers mean much.

**Do not train against a metric that is not this benchmark.** v10 was optimised on a holdout that
turned out to be 8 distinct samples, and regressed.
