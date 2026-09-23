#!/usr/bin/env python3
"""把 LoRA 适配器合并进基座权重，产出可直接分发的完整模型。

开源场景下合并版比 LoRA 更友好：使用者不用装 peft、不用配对基座版本，
下载即用（transformers / vLLM / llama.cpp 都直接支持）。

在 CPU 上做，避免和 GPU 上的服务抢显存。基座 Apache-2.0，合并后仍为 Apache-2.0。
"""
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ.get("BASE_MODEL", "/root/code/models/MiniCPM5-2B-hf")
ADAPTER = os.environ.get("ADAPTER_DIR", "runs/lora")
OUT = os.environ.get("OUT_DIR", "/root/code/models/MiniCPM5-2B-WebDecider")


def main():
    print(f"载入基座 {BASE}（CPU, fp16）", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                                trust_remote_code=True, device_map="cpu")
    print(f"挂载 LoRA {ADAPTER}", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("合并权重 …", flush=True)
    merged = model.merge_and_unload()
    merged.eval()

    os.makedirs(OUT, exist_ok=True)
    print(f"保存到 {OUT}", flush=True)
    merged.save_pretrained(OUT, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.save_pretrained(OUT)

    # 模版与生成配置必须一起带走，否则推理端行为会变
    for f in ("chat_template.jinja", "generation_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, f))

    cfg = json.load(open(os.path.join(OUT, "config.json")))
    cfg["_name_or_path"] = OUT
    cfg["architectures"] = ["LlamaForCausalLM"]
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    print("完成", flush=True)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"  {f:<40s} {os.path.getsize(p)/1e6:8.1f} MB" if os.path.isfile(p) else f"  {f}/")


if __name__ == "__main__":
    main()
