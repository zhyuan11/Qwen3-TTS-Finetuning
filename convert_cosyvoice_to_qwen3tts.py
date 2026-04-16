#!/usr/bin/env python3
"""Convert a CosyVoice-style dataset (wav + .normalized.txt pairs) into
Qwen3-TTS finetuning JSONL format.

Input layout (CosyVoice):
    {src_dir}/{split}/{speaker}/{emotion}/{utt_id}.wav
    {src_dir}/{split}/{speaker}/{emotion}/{utt_id}.normalized.txt

    Text files contain one line:
        用{角色}的{情绪}的语气说<|endofprompt|>{transcript}

Output:
    - Resampled 24 kHz mono WAVs under {output_dir}/datasets_24k/...
    - One reference audio per speaker under {output_dir}/ref_audio/{speaker}_ref.wav
    - Per-speaker JSONL files: {output_dir}/{split}_{speaker}.jsonl
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROMPT_SEP = "<|endofprompt|>"
MIN_DURATION = 0.5
MAX_DURATION = 30.0
REF_MIN_DURATION = 3.0
REF_MAX_DURATION = 10.0
TARGET_SR = 24000


def parse_transcript(txt_path: str) -> str:
    """Extract the raw transcript after <|endofprompt|>."""
    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.read().strip()
    if PROMPT_SEP in line:
        return line.split(PROMPT_SEP, 1)[1]
    return line


def resample_and_save(src_path: str, dst_path: str) -> float:
    """Load audio, convert to 24 kHz mono, save, and return duration in seconds."""
    audio, _ = librosa.load(src_path, sr=TARGET_SR, mono=True)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, audio, TARGET_SR)
    return len(audio) / TARGET_SR


def select_ref_audio(entries: list[dict]) -> dict | None:
    """Pick the best reference utterance: longest duration between REF_MIN and REF_MAX seconds."""
    candidates = [
        e for e in entries
        if REF_MIN_DURATION <= e["duration"] <= REF_MAX_DURATION
    ]
    if not candidates:
        candidates = sorted(entries, key=lambda e: abs(e["duration"] - 5.0))
        candidates = candidates[:1]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["duration"])


def main():
    parser = argparse.ArgumentParser(description="Convert CosyVoice dataset to Qwen3-TTS JSONL format")
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Root of the CosyVoice dataset (contains train/ and dev/)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for resampled audio and JSONL files")
    args = parser.parse_args()

    src_dir = args.src_dir
    output_dir = args.output_dir
    datasets_24k_dir = os.path.join(output_dir, "datasets_24k")
    ref_audio_dir = os.path.join(output_dir, "ref_audio")
    os.makedirs(ref_audio_dir, exist_ok=True)

    all_entries: dict[str, dict[str, list[dict]]] = {}  # speaker -> split -> [entry]

    for split in ["train", "dev"]:
        split_dir = os.path.join(src_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory not found: {split_dir}")
            continue

        wav_files = sorted(glob(os.path.join(split_dir, "*", "*", "*.wav")))
        print(f"[INFO] Found {len(wav_files)} wav files in {split}")

        skipped = 0
        for wav_path in wav_files:
            parts = Path(wav_path).relative_to(split_dir).parts
            if len(parts) != 3:
                print(f"[WARN] Unexpected path depth: {wav_path}")
                continue
            speaker, emotion, filename = parts
            utt_id = filename.replace(".wav", "")

            txt_path = wav_path.replace(".wav", ".normalized.txt")
            if not os.path.exists(txt_path):
                print(f"[WARN] Missing transcript: {txt_path}")
                continue

            transcript = parse_transcript(txt_path)
            if not transcript.strip():
                print(f"[WARN] Empty transcript: {txt_path}")
                continue

            dst_path = os.path.join(datasets_24k_dir, split, speaker, emotion, filename)
            duration = resample_and_save(wav_path, dst_path)

            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                print(f"[SKIP] {utt_id}: duration {duration:.2f}s out of range [{MIN_DURATION}, {MAX_DURATION}]")
                continue

            entry = {
                "audio": os.path.abspath(dst_path),
                "text": transcript,
                "speaker": speaker,
                "split": split,
                "duration": duration,
                "utt_id": utt_id,
            }

            all_entries.setdefault(speaker, {}).setdefault(split, []).append(entry)

        print(f"[INFO] {split}: skipped {skipped} utterances due to duration filter")

    speakers = sorted(all_entries.keys())
    print(f"\n[INFO] Speakers found: {speakers}")

    for speaker in speakers:
        train_entries = all_entries.get(speaker, {}).get("train", [])
        all_speaker_entries = train_entries + all_entries.get(speaker, {}).get("dev", [])

        ref_entry = select_ref_audio(all_speaker_entries)
        if ref_entry is None:
            print(f"[ERROR] No suitable reference audio for speaker {speaker}, skipping")
            continue

        ref_dst = os.path.join(ref_audio_dir, f"{speaker}_ref.wav")
        audio, _ = librosa.load(ref_entry["audio"], sr=TARGET_SR, mono=True)
        sf.write(ref_dst, audio, TARGET_SR)
        ref_audio_abs = os.path.abspath(ref_dst)
        print(f"[INFO] Speaker '{speaker}' ref audio: {ref_entry['utt_id']} ({ref_entry['duration']:.2f}s) -> {ref_dst}")

        for split in ["train", "dev"]:
            entries = all_entries.get(speaker, {}).get(split, [])
            if not entries:
                continue

            jsonl_path = os.path.join(output_dir, f"{split}_{speaker}.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for e in entries:
                    line = {
                        "audio": e["audio"],
                        "text": e["text"],
                        "ref_audio": ref_audio_abs,
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"[INFO] Wrote {len(entries)} entries to {jsonl_path}")

    print("\n[DONE] Conversion complete.")
    print(f"  Resampled audio: {datasets_24k_dir}")
    print(f"  Reference audio:  {ref_audio_dir}")
    print(f"  JSONL files:      {output_dir}/<split>_<speaker>.jsonl")


if __name__ == "__main__":
    main()
