#!/usr/bin/env python3
"""Run inference on finetuned Qwen3-TTS checkpoints for validation.

Tests a subset of checkpoints (epochs 0, 4, 9) for each speaker and saves
output WAVs to an output directory for manual listening evaluation.
"""

import argparse
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

TEST_TEXTS = {
    "zh": "今天天气真不错，我们一起去公园散步吧。",
    "en": "The quick brown fox jumps over the lazy dog.",
}

SPEAKERS = ["MT", "安吉拉", "汉克狗", "狗狗本"]
TEST_EPOCHS = [0, 4, 9]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="/root/workspace/Qwen3-TTS/finetuning")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str,
                        default="/root/workspace/Qwen3-TTS/finetuning/validation_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for speaker in SPEAKERS:
        for epoch in TEST_EPOCHS:
            ckpt_path = os.path.join(args.base_dir, f"output_{speaker}",
                                     f"checkpoint-epoch-{epoch}")
            if not os.path.isdir(ckpt_path):
                print(f"[SKIP] {ckpt_path} not found")
                continue

            print(f"\n[INFO] Loading {speaker} epoch-{epoch} from {ckpt_path}")
            tts = Qwen3TTSModel.from_pretrained(
                ckpt_path,
                device_map=args.device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

            for lang, text in TEST_TEXTS.items():
                out_path = os.path.join(
                    args.output_dir,
                    f"{speaker}_epoch{epoch}_{lang}.wav"
                )
                print(f"  Generating: {out_path}")
                try:
                    wavs, sr = tts.generate_custom_voice(
                        text=text,
                        speaker=speaker,
                    )
                    sf.write(out_path, wavs[0], sr)
                    print(f"  OK: {len(wavs[0])/sr:.2f}s @ {sr}Hz")
                except Exception as e:
                    print(f"  ERROR: {e}")

            del tts
            torch.cuda.empty_cache()

    print(f"\n[DONE] Validation outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
