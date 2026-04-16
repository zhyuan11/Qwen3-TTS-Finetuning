#!/usr/bin/env python3
"""Plot training loss curves for all speakers."""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(CJK_FONT_PATH)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_SCRIPT_DIR, "logs")
OUT_PATH = os.path.join(LOG_DIR, "training_loss_curves.png")

SPEAKERS = ["MT", "安吉拉", "汉克狗", "狗狗本"]
PATTERN = re.compile(r"Epoch (\d+) \| Step (\d+) \| Loss: ([\d.]+)")


def parse_loss_file(path):
    epochs, steps, losses = [], [], []
    with open(path) as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                steps.append(int(m.group(2)))
                losses.append(float(m.group(3)))
    return epochs, steps, losses


def compute_epoch_avg(epochs, losses):
    epoch_losses = {}
    for e, l in zip(epochs, losses):
        epoch_losses.setdefault(e, []).append(l)
    ep = sorted(epoch_losses.keys())
    avg = [np.mean(epoch_losses[e]) for e in ep]
    return ep, avg


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Qwen3-TTS Finetuning Loss Curves", fontsize=16, fontweight="bold")

for ax, speaker in zip(axes.flat, SPEAKERS):
    loss_file = os.path.join(LOG_DIR, f"loss_{speaker}.txt")
    if not os.path.exists(loss_file):
        ax.set_title(f"{speaker} (no data)")
        continue

    epochs, steps, losses = parse_loss_file(loss_file)
    if not losses:
        ax.set_title(f"{speaker} (no data)")
        continue

    global_steps = []
    max_step_per_epoch = {}
    for e, s in zip(epochs, steps):
        max_step_per_epoch[e] = max(max_step_per_epoch.get(e, 0), s)

    steps_per_epoch = {}
    for e in sorted(max_step_per_epoch.keys()):
        steps_per_epoch[e] = max_step_per_epoch[e] + 10

    offset = 0
    prev_epoch = -1
    for e, s in zip(epochs, steps):
        if e != prev_epoch:
            if prev_epoch >= 0:
                offset += steps_per_epoch.get(prev_epoch, 0)
            prev_epoch = e
        global_steps.append(offset + s)

    ax.plot(global_steps, losses, alpha=0.3, linewidth=0.8, color="steelblue", label="Per-step")

    ep, avg = compute_epoch_avg(epochs, losses)
    epoch_centers = []
    cum = 0
    for e in ep:
        epoch_centers.append(cum + steps_per_epoch.get(e, 0) / 2)
        cum += steps_per_epoch.get(e, 0)
    ax.plot(epoch_centers, avg, "o-", color="darkorange", linewidth=2, markersize=5, label="Epoch avg")

    ax.set_title(f"{speaker}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for e, c, a in zip(ep, epoch_centers, avg):
        if e % 2 == 0 or e == ep[-1]:
            ax.annotate(f"E{e}: {a:.1f}", (c, a), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
