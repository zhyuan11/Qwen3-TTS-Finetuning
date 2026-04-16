#!/usr/bin/env python3
"""A/B comparison: base model voice-clone vs finetuned custom_voice for 安吉拉."""

import os
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "validation_outputs")
DEVICE = "cuda:0"

BASE_MODEL_PATH = os.environ.get("QWEN3_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
FT_CKPT = os.path.join(BASE_DIR, "output_安吉拉", "checkpoint-epoch-9")
REF_AUDIO = os.path.join(BASE_DIR, "datasets_24k/train/安吉拉/冷静/安吉拉_冷静_120.wav")

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

MAX_NEW_TOKENS = 512


def generate_with_base_model():
    """Generate all 10 sentences with voice clone using the base model."""
    out_subdir = os.path.join(OUT_DIR, "安吉拉_base_clone")
    os.makedirs(out_subdir, exist_ok=True)

    print("=" * 60, flush=True)
    print("[LOAD] Base model for voice clone", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        BASE_MODEL_PATH,
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("[BUILD] Creating voice clone prompt (x_vector_only_mode=True) ...", flush=True)
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        x_vector_only_mode=True,
    )
    print("[BUILD] Done.", flush=True)

    for i, text in enumerate(SENTENCES, 1):
        out_path = os.path.join(out_subdir, f"{i:02d}.wav")
        print(f"  [{i:02d}/10] \"{text}\"", flush=True)
        t0 = time.time()
        try:
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Chinese",
                voice_clone_prompt=prompt_items,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            dur = len(wavs[0]) / sr
            elapsed = time.time() - t0
            sf.write(out_path, wavs[0], sr)
            print(f"         -> {out_path} ({dur:.2f}s audio, {elapsed:.1f}s wall)", flush=True)
        except Exception as e:
            print(f"         ERROR: {e}", flush=True)

    del tts
    torch.cuda.empty_cache()
    print("[DONE] Base model voice clone\n", flush=True)


def generate_with_finetuned():
    """Generate all 10 sentences with the finetuned custom_voice model."""
    out_subdir = os.path.join(OUT_DIR, "安吉拉_finetuned")
    os.makedirs(out_subdir, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"[LOAD] Finetuned model: {FT_CKPT}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        FT_CKPT,
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    for i, text in enumerate(SENTENCES, 1):
        out_path = os.path.join(out_subdir, f"{i:02d}.wav")
        print(f"  [{i:02d}/10] \"{text}\"", flush=True)
        t0 = time.time()
        try:
            wavs, sr = tts.generate_custom_voice(
                text=text,
                speaker="安吉拉",
                max_new_tokens=MAX_NEW_TOKENS,
            )
            dur = len(wavs[0]) / sr
            elapsed = time.time() - t0
            sf.write(out_path, wavs[0], sr)
            print(f"         -> {out_path} ({dur:.2f}s audio, {elapsed:.1f}s wall)", flush=True)
        except Exception as e:
            print(f"         ERROR: {e}", flush=True)

    del tts
    torch.cuda.empty_cache()
    print("[DONE] Finetuned model custom voice\n", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    generate_with_base_model()
    generate_with_finetuned()
    print("[ALL DONE] Compare outputs at:", flush=True)
    print(f"  Base clone:  {OUT_DIR}/安吉拉_base_clone/", flush=True)
    print(f"  Finetuned:   {OUT_DIR}/安吉拉_finetuned/", flush=True)


if __name__ == "__main__":
    main()
