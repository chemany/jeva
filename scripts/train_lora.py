#!/usr/bin/env python3
"""MiniCPM5-2B 的 LoRA SFT：用确定性求解器采集的真实轨迹，教它做浏览器决策。

数据来自 collect.py：每条是 (prompt_plain, 动作)。
prompt_plain 由 llm_backend.build_prompt 生成 —— **与推理时逐字一致**，避免训练/部署漂移。
目标输出是紧凑 JSON，如 {"operation":"CLICK","target":"3","text":""}。

V100 不支持 bf16，全程 fp16 + 梯度检查点。
"""
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

BASE = os.environ.get("BASE_MODEL", "/root/code/models/MiniCPM5-2B-hf")
DATA = os.environ.get("SFT_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals", "data", "sft.jsonl"))
OUT = os.environ.get("OUT_DIR", "runs/lora")
EPOCHS = float(os.environ.get("EPOCHS", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
BS = int(os.environ.get("BATCH", "2"))
ACCUM = int(os.environ.get("ACCUM", "8"))
LR = float(os.environ.get("LR", "1e-4"))

# 复用与推理完全相同的 system prompt，避免两边不一致
import importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jeva.prompt import SYSTEM as SYS_PROMPT


def build_dataset(tok):
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    prompt_texts, answers = [], []
    for r in rows:
        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": r["prompt"]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ans = json.dumps({"operation": r["operation"],
                          "target": "" if r["target"] is None else str(r["target"]),
                          "text": r.get("text") or ""}, ensure_ascii=False)
        prompt_texts.append(p)
        answers.append(ans + (tok.eos_token or ""))
    print(f"样本 {len(rows)} 条")
    return prompt_texts, answers


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts, answers = build_dataset(tok)

    def encode(batch):
        full = [p + a for p, a in zip(batch["prompt"], batch["answer"])]
        enc = tok(full, truncation=True, max_length=MAXLEN, padding=False)
        plens = [len(tok(p, truncation=True, max_length=MAXLEN)["input_ids"]) for p in batch["prompt"]]
        labels = []
        for ids, pl in zip(enc["input_ids"], plens):
            lab = list(ids)
            for i in range(min(pl, len(lab))):        # 只在答案部分算 loss
                lab[i] = -100
            labels.append(lab)
        enc["labels"] = labels
        return enc

    ds = Dataset.from_dict({"prompt": prompts, "answer": answers}).map(
        encode, batched=True, remove_columns=["prompt", "answer"])

    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                                trust_remote_code=True)
    model.config.use_cache = False
    GC = os.environ.get("GRAD_CKPT", "1") == "1"
    if GC:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT + "_ckpt", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BS, gradient_accumulation_steps=ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=int(os.environ.get("WARMUP","80")),
        logging_steps=20, save_strategy="epoch", save_total_limit=2,
        fp16=True, optim="adamw_torch", report_to=[], dataloader_num_workers=2,
        gradient_checkpointing=GC, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"LoRA 已保存 -> {OUT}")


if __name__ == "__main__":
    main()
