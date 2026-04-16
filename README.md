# Qwen3-TTS Finetuning (Fixed)

A **bug-fixed** fork of the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) single-speaker finetuning pipeline. The official finetuning scripts contain critical bugs that cause finetuned models to produce **noise, gibberish, or unnaturally fast speech** — even though training loss decreases normally. This repository applies community-validated fixes so that finetuning actually works.

## What's Fixed

The official `sft_12hz.py` has three bugs. All are fixed in this repo.

### 1. Missing `text_projection` → noise output

The training script uses raw text embeddings, but inference applies a 2-layer MLP (`text_projection`) on top. This train/inference mismatch causes the finetuned model to produce pure noise.

```diff
- input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
+ input_text_embedding = model.talker.text_projection(
+     model.talker.model.text_embedding(input_text_ids)
+ ) * text_embedding_mask
```

> Ref: [QwenLM/Qwen3-TTS#39](https://github.com/QwenLM/Qwen3-TTS/issues/39)

### 2. Double label shifting → fast/garbled speech

The script manually shifts inputs (`[:, :-1]`) and labels (`[:, 1:]`), but HuggingFace's `ForCausalLMLoss` also shifts internally. This double shift makes the model predict token `t+2` from `t` instead of `t+1`, causing progressively faster speech.

```diff
  outputs = model.talker(
-     inputs_embeds=input_embeddings[:, :-1, :],
-     attention_mask=attention_mask[:, :-1],
-     labels=codec_0_labels[:, 1:],
+     inputs_embeds=input_embeddings,
+     attention_mask=attention_mask,
+     labels=codec_0_labels,
      output_hidden_states=True
  )
  hidden_states = outputs.hidden_states[0][-1]
- talker_hidden_states = hidden_states[codec_mask[:, :-1]]
- talker_codec_ids = codec_ids[codec_mask]
+ target_codec_mask = codec_mask[:, 1:]
+ talker_hidden_states = hidden_states[:, :-1][target_codec_mask]
+ talker_codec_ids = codec_ids[:, 1:][target_codec_mask]
```

> Ref: [QwenLM/Qwen3-TTS#179](https://github.com/QwenLM/Qwen3-TTS/issues/179), fix by [@fumyou13](https://github.com/QwenLM/Qwen3-TTS/issues/179#issuecomment-3870059313)

### 3. Sub-talker double shift (library patch required)

The sub-talker's `forward_finetune` has the same issue: its logits are already aligned with labels, but `ForCausalLMLoss` shifts them again. Fix by replacing with direct `F.cross_entropy` in the installed `qwen_tts` library.

> A ready-to-apply patch is provided in `patches/modeling_qwen3_tts.patch`.

### 4. Speaker name case sensitivity

The library looks up speakers via `speaker.lower()`, but the original script saved names in their original case. Fixed by storing lowercase names in the checkpoint config.

## Before vs After

| | Official script | This repo (fixed) |
|---|---|---|
| Initial loss | ~13–22 | ~2.5–3.0 |
| Converged loss | ~2 (misleading) | ~2.0–2.2 |
| Output quality | Noise / gibberish | Clear speech |
| EOS token | Not emitted (runs to max_new_tokens) | Properly emitted |
| Speech speed | Progressively faster each epoch | Stable, natural speed |

## Quick Start

### Prerequisites

```bash
pip install qwen-tts
# Apply the sub-talker patch (required):
QWEN_TTS_DIR=$(python -c "import qwen_tts; import os; print(os.path.dirname(qwen_tts.__file__))")
patch -p2 -d "$QWEN_TTS_DIR" < patches/modeling_qwen3_tts.patch
```

### 1. Prepare training data

Create a JSONL file (`train_raw.jsonl`):

```jsonl
{"audio": "path/to/utterance.wav", "text": "transcript text", "ref_audio": "path/to/ref.wav"}
```

- Audio: **24kHz, mono, 16-bit PCM WAV**
- Use the **same `ref_audio`** for all samples (a clear 3–6s clip of the target speaker)

### 2. Extract audio codes

```bash
python prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl train_raw.jsonl \
    --output_jsonl train_with_codes.jsonl \
    --batch_size 4
```

### 3. Finetune

```bash
python sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path output \
    --train_jsonl train_with_codes.jsonl \
    --batch_size 2 \
    --lr 2e-6 \
    --num_epochs 10 \
    --speaker_name my_speaker
```

### 4. Generate speech

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-9",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="Hello, this is a test.",
    speaker="my_speaker",
    max_new_tokens=512,
)
sf.write("output.wav", wavs[0], sr)
```

## Recommended Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `--lr` | `2e-6` | Official `2e-5` is too high for small batches |
| `--batch_size` | `2` | For 24GB VRAM; effective batch = batch_size × grad_accum (4) = 8 |
| `--num_epochs` | `5–10` | Loss converges by epoch 3–5 |

## Files

| File | Description |
|---|---|
| `sft_12hz.py` | Fixed SFT training script |
| `dataset.py` | Training dataset class (from official repo) |
| `prepare_data.py` | Audio code extraction (added `--batch_size` arg) |
| `convert_cosyvoice_to_qwen3tts.py` | CosyVoice → Qwen3-TTS data converter |
| `generate_samples.py` | Batch inference for validation |
| `generate_ab_compare.py` | A/B test: base voice-clone vs finetuned custom-voice |
| `plot_loss.py` | Training loss curve plotter |
| `run_all_sft.sh` | Multi-speaker sequential training script |
| `patches/` | Library patch for sub-talker fix |

## Acknowledgements

Bug fixes are based on community findings from the Qwen3-TTS GitHub issues:
- [#39](https://github.com/QwenLM/Qwen3-TTS/issues/39) — `text_projection` mismatch ([@KdaiP](https://github.com/KdaiP))
- [#179](https://github.com/QwenLM/Qwen3-TTS/issues/179) — Double label shifting ([@humblenginr](https://github.com/humblenginr), [@fumyou13](https://github.com/fumyou13))
- [#72](https://github.com/QwenLM/Qwen3-TTS/issues/72) — Voice similarity analysis ([@EthanLifeGreat](https://github.com/EthanLifeGreat))

## License

The finetuning scripts follow the original [Apache 2.0 License](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) from Qwen3-TTS.
