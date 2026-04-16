#!/usr/bin/env python3
"""Generate 10 sample sentences per speaker using the best finetuned checkpoint."""

import os
import sys
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))

from qwen_tts import Qwen3TTSModel

BASE_DIR = "/root/workspace/Qwen3-TTS/finetuning"
OUT_DIR = os.path.join(BASE_DIR, "validation_outputs")
DEVICE = "cuda:0"
BEST_EPOCH = 9

SENTENCES = [
    "今天天气真不错，我们一起去公园散步吧。",
    "你好，请问这个东西多少钱？",
    "我觉得这部电影特别好看，你看过吗？",
    "明天早上八点我们在学校门口集合。",
    "这道菜的味道好极了，你一定要尝尝。",
    "小心点，路上有点滑，别摔倒了。",
    "谢谢你的帮助，真的非常感谢。",
    "我昨天晚上做了一个很奇怪的梦。",
    "快要过年了，你打算怎么庆祝呢？",
    "不好意思，我来晚了，路上堵车了。",
]

SPEAKERS = ["MT", "安吉拉", "汉克狗", "狗狗本"]
MAX_NEW_TOKENS = 512


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for speaker in SPEAKERS:
        ckpt = os.path.join(BASE_DIR, f"output_{speaker}",
                            f"checkpoint-epoch-{BEST_EPOCH}")
        if not os.path.isdir(ckpt):
            print(f"[SKIP] {speaker}: checkpoint not found at {ckpt}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[LOAD] {speaker} (epoch {BEST_EPOCH})", flush=True)

        tts = Qwen3TTSModel.from_pretrained(
            ckpt,
            device_map=DEVICE,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        speaker_dir = os.path.join(OUT_DIR, speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        for i, text in enumerate(SENTENCES, 1):
            out_path = os.path.join(speaker_dir, f"{i:02d}.wav")
            print(f"  [{i:02d}/10] \"{text}\"", flush=True)
            try:
                wavs, sr = tts.generate_custom_voice(
                    text=text,
                    speaker=speaker.lower(),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                sf.write(out_path, wavs[0], sr)
                dur = len(wavs[0]) / sr
                print(f"         -> {out_path} ({dur:.2f}s @ {sr}Hz)", flush=True)
            except Exception as e:
                print(f"         ERROR: {e}", flush=True)

        del tts
        torch.cuda.empty_cache()
        print(f"[DONE] {speaker}\n", flush=True)

    print(f"\n[ALL DONE] Outputs at: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
