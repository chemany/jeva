# jeva

一个 2B 的浏览器智能体决策模型，你可以自己训练、自己部署。

<p>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="#模型"><img alt="Weights: 2B · merged · GGUF" src="https://img.shields.io/badge/WEIGHTS-2B%20%C2%B7%20merged%20%C2%B7%20GGUF-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="evals/"><img alt="Eval suites: frozen" src="https://img.shields.io/badge/EVAL%20SUITES-frozen-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="docs/pipeline.md"><img alt="Data pipeline" src="https://img.shields.io/badge/DATA-zero%20human%20labels-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="docs/model-card.md"><img alt="Model card" src="https://img.shields.io/badge/MODEL%20CARD-docs-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

<p><b>简体中文</b> · <a href="README.md">English</a></p>

jeva 接收「一次页面观察 + 一个目标」，输出**恰好一个动作**——紧凑 JSON，不生成任何多余文本：

```json
{"operation": "CLICK", "target": "2", "text": ""}
```

它是对 [`openbmb/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B) 的微调，
训练数据由**确定性求解器驱动真实 Chrome** 采集而来。权重已**合并进基座**——没有需要配对的
适配器，除了你日常的推理栈之外不需要额外安装任何东西。

## 亮点

- **一次决策、一次调用，约 250 毫秒。** 2B 参数，Q4_K_M 量化后仅 1.6 GB，单张 V100 跑 llama.cpp 即可。
- **带类型的动作空间。** `CLICK` · `TYPE_TEXT` · `SELECT` · `WAIT` · `DONE` · `BLOCKED`，目标是**你提供的那份观察**里的元素序号。
- **不会自己编选择器。** 模型只返回序号，由执行器在**同一份快照**上解析。它永远不输出 CSS、坐标或 JavaScript。
- **在冻结测试集上 100% 完成任务**，其中包括一个标签体系与布局从未出现在训练中的站点。
- **零人工标注。** 2,600 个任务产出 10,686 条训练轨迹，采集约 30 分钟；训练是一轮 LoRA，单张 V100 约 75 分钟。
- **代码齐全**：采集器、训练脚本、合并/量化流程、三个测试站点、以及冻结的评测结果。

## 安装

两种方式都不需要克隆仓库，也都不会多装一个依赖。

```bash
pip install jeva        # 客户端 + CLI，约 19 KB，零依赖
jeva download          # 拉 Q4_K_M 权重（1.6 GB）到 ~/.cache/jeva，附 sha256
jeva demo              # 服务没起就帮你起，然后做一次真实决策
jeva serve             # 把服务留在 127.0.0.1:8020
jeva check             # 报告缺什么（llama-server / 权重 / 端口）
```

wheel 里**只装客户端**，权重按需下载——1.6 GB 的 GGUF 不应该塞进 Python 发行包里。
想从自己的镜像下，把 `JEVA_BASE_URL`（或 `--url`）指过去即可。

PyPI 正式发布之前，可以直接从 Git 安装：

```bash
pip install "git+https://github.com/chemany/jeva"
```

## 快速开始

手动起服务——从 [release](https://github.com/chemany/jeva/releases) 取一份 GGUF 然后：

```bash
llama-server -m MiniCPM5-2B-WebDecider-Q4_K_M.gguf \
  --alias jeva --port 8020 -ngl 99 -c 4096 --jinja
```

接着向它要一个决策。jeva 需要特定格式的观察（见 [state 格式](#state-格式)），
`jeva` 包会帮你渲染：

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
# {'operation': 'CLICK', 'target': '2', 'text': ''}      ← 目标要求的那个 radio
```

已经处于目标状态的单选框不会去重复点击；空的文本框会被填写；遇到验证码页面返回 `BLOCKED`。
**「什么时候不该动手」这四条规则是训进模型里的，不是靠你的提示词。**

无需额外安装即可试一下：

```bash
python examples/quickstart.py            # 用内置的示例观察
```

## 模型

| 模型 | 基座 | 参数 | 形态 | 体积 | 测试集准确率 | 单次决策延迟 |
|---|---|---|---|---|---|---|
| **jeva**（本次发布） | MiniCPM5-2B | 2.5B（非嵌入 2.0B） | 已合并，HF | 5.0 GB | **100%** | 1500 ms（朴素 HF） |
| jeva · GGUF F16 | ” | ” | llama.cpp | 5.04 GB | 100% | ~230 ms |
| **jeva · GGUF Q8_0** | ” | ” | llama.cpp | 2.68 GB | **100%** | **283 ms** |
| **jeva · GGUF Q4_K_M** | ” | ” | llama.cpp | **1.56 GB** | **100%** | **229 ms** |
| MiniCPM5-2B（未训练） | — | 2.5B | llama.cpp | 1.56 GB | 10%（100 题） | 147 ms |
| Bonsai-27B（零样本） | Qwen3.8-27B | 27B | GGUF q4_0 | 14 GB | 92.5%（40 题） | 1769 ms |

所有形态都在 [ModelScope](https://modelscope.cn/models/imjasonli/jeva) 上：GGUF 放在 `gguf/` 下，
合并后的 transformer 权重放在同一仓库根目录。`jeva download` 即从那里取回。

量化在这里**没有代价**：三档 GGUF 在同一批测试上都是 **100%**。
同一提示词下，未训练的基座只有 **10%**——它会反复点击已勾选的单选框、该用 `TYPE_TEXT` 的地方用
`CLICK`，100 个任务里有 18 次编造了不存在的元素序号。

> **这不是公平对比。** jeva 是**针对该动作空间训练过的**；Bonsai-27B 和未训练基座是同一提示词下的
> 零样本。这张表要说明的是「半小时自生成监督能给一个 2B 模型换来什么」，而不是「2B 打赢了 27B」。

## 基准

![jeva、未训练基座与 Bonsai-27B 的逐任务类型准确率](docs/comparison.png)

在随机任务集上的**任务成功率**。每个任务都在**真实 Chrome** 里逐步执行，并**独立校验**最终页面状态
——提交的值会从结果 URL 里读回来，所以只有浏览器真的停在正确状态时才算成功。

| 任务类型 | MiniCPM5-2B（基座） | Bonsai-27B（零样本） | **jeva** |
|---|---|---|---|
| 拦截页（CAPTCHA / 限流） | 9 / 9 · 100% | 5 / 5 · 100% | **9 / 9 · 100%** |
| 机票搜索（radio + 3 字段 + 提交） | 0 / 21 · 0% | 7 / 7 · 100% | **21 / 21 · 100%** |
| 酒店筛选（2 个 select + 复选框） | 0 / 26 · 0% | 12 / 12 · 100% | **26 / 26 · 100%** |
| 联系表单（3 字段 + 同意条款） | 0 / 6 · 0% | 0 / 1 · 0% | **6 / 6 · 100%** |
| 自动补全（输入 → 点选项） | 1 / 7 · 14% | 4 / 4 · 100% | **7 / 7 · 100%** |
| 订单表单（多要求 + 时间 + 备注） | 0 / 31 · 0% | 9 / 11 · 82% | **31 / 31 · 100%** |
| **总体** | **10%**（100 题） | **92.5%**（40 题） | **100%**（100 题） |

Bonsai 那一列只有 40 题，仅供参考。它做订单表单时**没提交就停下**，做联系表单时编造了目标。

### 会被平均值掩盖的子集

一个 31 题的家族丢掉 3 题，总体数字可以纹丝不动。下面这些子集只保留「页面什么都不预填」的目标，
因此每一项要求都必须由决策模型自己动手（[`evals/subset_eval.py`](evals/subset_eval.py)，各 20 题）：

| 子集 | 目标把什么留给模型 | jeva |
|---|---|---|
| `when` | 送达时间，写成 `12:30` 或 `9:00 AM` | **20 / 20** |
| `note` | 自由文本备注，四种不同措辞 | **20 / 20** |
| `contact` | 姓名/电话/邮箱，用裸列表给出 | **20 / 20** |
| `compound` | 以上全部 + 配料，即普通订单表单 | **20 / 20** |

### 真实站点抽查

jeva 只在 `site/` 里的合成页面训过。把它指向一个**从未见过的真实第三方表单**——
[`httpbin.org/forms/post`](https://httpbin.org/forms/post)（三个文本框、一组单选、四个复选框，
外加一个浏览器原生时间输入框），它把目标点名的每个字段都填好并提交。跑了四次：

| | 核心字段（姓名/电话/邮箱/尺寸） | 目标要求的两个配料 | 送达时间 | 已提交且正确判 `DONE` |
|---|---|---|---|---|
| 四次 | **4 / 4** | **4 / 4** | **4 / 4** | **4 / 4** |

每一次都从目标里正确推导出值（`Jason Zhang`、`555-0100`、`jason@example.com`、`size: large`），
填入 `12:30`，**同时**勾中 `Mushroom` 与 `Extra Cheese`——而目标的写法是「with mushrooms and
cheese」，也就是标签必须**按语义**匹配、而非按字面——并且只在浏览器真的提交之后才停下。

复现：`python evals/real_site_test.py`（需要网络）。

后面三轮训练都是这项检查逼出来的：它先暴露出决策模型**根本看不见** `<input type=time>`，
接着暴露出看得见也**写不进去**，最后暴露出它从没被要求填过「标签不在目标里」的字段。
现在每一种都成了采集器里的一个任务家族，也是 `evals/subset_eval.py` 里的一个子集。

### 留出站点

用了三个**标签体系与布局都不同**的站点。`site3` 是严格留出：从未用于训练，标签也完全不同
（`Origin` / `Destination` / `When` / `Cabin class` / `Guests` / `Round-trip` / `One-way` / `Submit`）。

| 测试集 | 任务数 | jeva |
|---|---|---|
| `site1`（训练分布内） | 100 | **100%** |
| `site2`（训练分布内，带额外导航干扰） | 40 | **100%** |
| **`site3`（从未训练，标签全新）** | 40 | **100%** |
| 线上 `httpbin.org/forms/post`（从未训练，真实标记） | 4 次 | 见[上文](#真实站点抽查) |

### 是什么把数字推上去的

| 改动 | 效果 |
|---|---|
| 用已校验轨迹做 LoRA SFT | 总体 **10% → 100%** |
| 多站点采集（含会改变元素序号的干扰链接） | site2 机票 **19% → 100%** |
| 「一个目标要勾多个框」的任务家族 | 订单表单 **0% → 100%** |
| 「目标没写出标签原文」的任务家族 | 目标说 cheese 也能找到 `Extra Cheese` |
| 时间字段与自由文本备注的任务家族 | `when` / `note` / `contact` 子集 **100%** |
| 去掉提示词里遗留的序号重复 | 无变化（实测等价，100% → 100%） |
| 合并权重 + Q4_K_M 量化 | 无变化（100% → 100%），5.0 GB → 1.6 GB |
| 服务路径（同权重、同提示词） | llama.cpp **0.23 s** · vLLM 0.95 s · 朴素 HF generate 8.4 s |

训练数据规模与消融见 [docs/pipeline.md](docs/pipeline.md)；评测 JSON 冻结在
[`evals/results/`](evals/results/)。

**仓库里有什么：** 运行时包、采集器与训练脚本、三个测试站点、冻结的评测结果，
以及一份 120 条的[训练集样本](evals/data/sample.jsonl)供你核对格式。
**仓库里没有什么：** 完整的 10,686 条训练集（约 72 MB）与权重文件——前者用采集器
约 30 分钟即可再生，后者在 [ModelScope](https://modelscope.cn/models/imjasonli/jeva) 上。模型卡在
[docs/hf-model-card.md](docs/hf-model-card.md)。

## state 格式

jeva 读的是**紧凑纯文本**，每个可交互元素一行。五条规则，每条都实测过：

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

1. **「无值」绝不能渲染成空字符串** —— 写 `not set`。空值会被读成**有值**。
2. **必须带 `{ops: ...}`** —— 这是区分「只能选的下拉框」与「可以打字的文本框」的**唯一结构性信号**。
3. 下拉框只列**未选中**的选项，键为 `<元素>:<选项>`。
4. 单选框与复选框要渲染 `checked=true/false`。
5. **用紧凑纯文本。** 换成 JSON blob 会显著掉点。

`jeva.render.render_state()` 从 `Page`/`Element`/`Option` 对象产出这段文本，
`jeva.prompt.build_prompt()` 再包上模型期望的操作说明与候选目标清单。
**这两处就是训练时用的同一份代码。**

> **系统提示词是模型的一部分。** 改措辞会显著掉点——消融见 [docs/pipeline.md](docs/pipeline.md)。
> 如果你需要不同的契约，请微调，而不是改写提示词。

## 接口

jeva 就是一个普通的 OpenAI 兼容 chat 端点，服务端不需要安装任何东西。

```bash
curl -s localhost:8020/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "jeva",
  "max_tokens": 64,
  "temperature": 0.0,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [
    {"role": "system", "content": "<jeva/prompt.py 里的 SYSTEM，逐字照抄>"},
    {"role": "user",   "content": "<state>\n\nAvailable operations:\n…"}
  ]}'
```

`chat_template_kwargs.enable_thinking=false` **是必需的**：MiniCPM5 是推理模型，
开着思考时推理轨迹会吃光 token 预算，`content` 返回空。

### Python

```python
from jeva import Jeva, Page, Element

jeva = Jeva("http://127.0.0.1:8020/v1", model="jeva")
action = jeva.decide(page, goal, history=["CLICK Round trip"])
```

| 符号 | 用途 |
|---|---|
| `Jeva(base_url, model)` | 客户端；`.decide(page, goal, history)` → `Action`，`.last_latency_ms` |
| `Page(elements, url, title, text)` | 一次观察 |
| `Element(index, role, label, value, operations, options, checked)` | 一个控件 |
| `Action(operation, target, text)` | 一个决策；`DONE`/`BLOCKED` 时 `.terminal` 为真 |
| `jeva.resolve(page, action)` | 把目标序号解析回你发出的那份观察 |
| `jeva.render_state` / `jeva.build_prompt` | 与训练时完全一致的渲染 |

## 工作原理

一个跟随目标的循环是：观察 → 决策 → 执行 → 再观察。jeva 只负责**决策**这一步。

```
browser ──snapshot──▶ Page(elements, text) ──▶ jeva ──▶ Action(operation, target, text)
   ▲                                                        │
   └──────────────── 执行这个目标 ──────────────────────────┘
```

模型被训练去遵守浏览器智能体真正需要的那几条规则——**容易说清、难做对**的那些：

- 已经处于目标状态的单选框，不要重复点击；
- 已经持有正确值的字段，不要重复输入；
- 输入文本要用 `TYPE_TEXT`（不是 `CLICK`），选下拉值要用 `SELECT`；
- 只有在可见证据充分时才 `DONE`，被验证码挡住才 `BLOCKED`。

其余一切——元素身份、选择器、页面新鲜度检查、重试——都留在执行器里。

### 训练数据

没有人工标注，也没有教师模型。确定性求解器驱动真实 Chrome，记录 `(state, action)`；
只有环境校验任务成功时，那条轨迹才会被保留。

```
真实 Chrome ──快照──▶ DOM（元素 + 页面文本）
     ▲                        │
     │                        ▼
   执行 ◀── 确定性求解器（能看到特权的「已满足 / 未完成」一行）
                              │
                              ▼
              记录 (state, action)  ← 这份记录里**不含**提示
                              │
              仅当环境校验成功才保留
                              ▼
                    LoRA SFT → 合并 → GGUF
```

产出：**2,600 个任务 → 10,686 条已校验样本，约 30 分钟**。求解器看得到「哪些已满足」的特权摘要，
学生看不到，所以它必须自己从状态里学会**状态追踪**。细节与消融见
[docs/pipeline.md](docs/pipeline.md)。

## 训练你自己的

```bash
# 1) 三个测试站（标签体系与布局各不相同）
python -m http.server 8899 --directory site/site1 &
python -m http.server 8898 --directory site/site2 &
python -m http.server 8897 --directory site/site3 &

# 2) 采集已校验轨迹（1 万样本约 30 分钟）
SITES="http://127.0.0.1:8899,http://127.0.0.1:8898" python evals/collect.py 2200 evals/data/sft.jsonl

# 3) LoRA SFT（1 轮，单张 V100 约 70 分钟）
EPOCHS=1 BATCH=4 ACCUM=4 MAXLEN=1152 \
  SFT_DATA=evals/data/sft.jsonl OUT_DIR=runs/lora python scripts/train_lora.py

# 4) 合并进基座，再量化
python scripts/merge_lora.py
python scripts/convert_gguf.sh
```

采集器与站点无关：把它指向你自己的页面、扩一下别名表，同一条管线就能为你那个站点产出训练数据。
见 [docs/pipeline.md](docs/pipeline.md)。

## 局限

- **它只做决策，不做执行。** 输出是**你提供的那份观察**里的序号，执行器必须自己解析、检查页面是否
  仍然新鲜、并处理失败。
- **动作空间很窄。** 没有 `SCROLL`、没有文件上传、没有多步下拉控件。你的智能体若需要这些，
  请扩展动作空间并重训。
- **面向表单形态的任务。** 它在搜索/筛选/表单/自动补全这类流程上训练；`DONE` 与 `BLOCKED`
  请当作**建议**，由你自己的验证器确认，尤其在分布之外。
- **对提示词敏感。** 系统提示词、操作说明、state 布局都是模型的一部分。请逐字复用
  （`jeva.prompt` / `jeva.render` 就是）。
- **不是通用助手。** 它是一个决策头，不是聊天模型。
- **页面拒收的值它检测不出来。** 把 `<input type=time min="11:00">` 填成 09:00，浏览器会拦住
  表单提交，而模型会一直点提交、不会报 `BLOCKED`。它看不到 state 里的 `min`/`max`，训练里也
  没有这种失败样本。要么给受限字段范围内的值，要么让执行器把校验信息暴露出来。
- **终止能力只覆盖训练过的形态。** 现在「页面没有可交互元素」和「当前页面无法满足目标」都能
  正确停下，但**新形态**的死路不在覆盖范围内。
- 评测在三个合成站点 + 一个线上第三方表单（真实 Chrome、真实 DOM）上完成，**不是生产网站基准**。

## 许可

**Apache-2.0**，与基座一致。jeva 是
[`openbmb/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B) 的微调衍生模型；
模型名遵循 MiniCPM 的衍生命名惯例，原始工作在下方署名。

```bibtex
@misc{minicpm5,
  title  = {MiniCPM5-2B},
  author = {OpenBMB},
  year   = {2026},
  url    = {https://modelscope.cn/models/OpenBMB/MiniCPM5-2B}
}
```
