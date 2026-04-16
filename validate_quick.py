#!/usr/bin/env python3
"""Quick validation: test best checkpoint (epoch 9) for each speaker."""

import os
import sys
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

BASE_DIR = "/root/workspace/Qwen3-TTS/finetuning"
OUT_DIR = os.path.join(BASE_DIR, "validation_outputs")
DEVICE = "cuda:0"
TEXT = "今天天气真不错，我们一起去公园散步吧。"

SPEAKERS = ["MT", "安吉拉", "汉克狗", "狗狗本"]
EPOCH = 9

os.makedirs(OUT_DIR, exist_ok=True)

for speaker in SPEAKERS:
    ckpt = os.path.join(BASE_DIR, f"output_{speaker}", f"checkpoint-epoch-{EPOCH}")
    if not os.path.isdir(ckpt):
        print(f"[SKIP] {ckpt} not found", flush=True)
        continue

    print(f"[LOAD] {speaker} epoch-{EPOCH}...", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        ckpt, device_map=DEVICE, dtype=torch.bfloat16,
    )

    out_path = os.path.join(OUT_DIR, f"{speaker}_epoch{EPOCH}.wav")
    print(f"[GEN]  {out_path}", flush=True)
    try:
        wavs, sr = tts.generate_custom_voice(text=TEXT, speaker=speaker)
        sf.write(out_path, wavs[0], sr)
        dur = len(wavs[0]) / sr
        print(f"[OK]   {dur:.2f}s @ {sr}Hz", flush=True)
    except Exception as e:
        print(f"[ERR]  {e}", flush=True)

    del tts
    torch.cuda.empty_cache()
    print("", flush=True)

print("[DONE]", flush=True)
