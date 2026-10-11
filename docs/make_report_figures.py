"""Draws the charts in docs/REPORT.md from the measured numbers below (sources in the comments).

    python docs/make_report_figures.py      # needs matplotlib; writes docs/images/fig_*.png

Every number here was measured, not estimated: see REPORT.md for how, and the research repo's RESULTS.md (§9, §13)
for the runs they come from.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"
NAVY, ORANGE, GREY, GREEN, RED = "#1c3252", "#f28c28", "#9aa4b2", "#3aa655", "#d64545"

# Dev shots called right when the tracker sees every Nth saved frame (replay_eval.py --stride N, every offset k).
# Before / after the held-ball fix; gym 1 = sessions 1-4 (162 shots), gym 2 = sessions 5, 9, 10, 11 (102 shots).
FRAMERATE = {
    "gym1": {"before": {60: [153], 30: [150, 146], 20: [150, 144, 143]},
             "after": {60: [153], 30: [150, 147], 20: [151, 144, 142]}, "n": 162},
    "gym2": {"before": {60: [97], 30: [97, 96], 20: [89, 92, 89]},
             "after": {60: [97], 30: [97, 97], 20: [97, 97, 95]}, "n": 102},
}
# The app's Check button: 60 PyTorch boxes on 24 dev frames.
PARITY = {  # variant: (mean IoU, min IoU, max |conf difference|, max corner difference px, extra boxes)
    "fp32\nexact resize": (1.000, 0.999, 0.000, 0.0, 0),
    "fp16 (live)\nexact resize": (0.988, 0.959, 0.013, 0.7, 0),
    "fp16\nApple vImage": (0.982, 0.940, 0.066, 3.2, 1),
}
# Per-frame time on an iPhone 11 (HUD readouts): two runs.
TIMING = {  # label: (prepare ms, model ms, decode ms, detector fps)
    "0.5x, nominal heat\n(Oct 11)": (1.6, 30.6, 0.2, 32.0),
    "0.5x, warm phone\n(Oct 10)": (1.9, 34.9, 0.3, 27.9),
}
# First-look test in the second gym (sessions 6-8, 90 shots, offline Python, frozen c78a239): rows = label.
CONFUSION = [[43, 9, 1],  # made: called made, called missed, not found
             [1, 34, 2]]  # missed


def framerate():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (gym, d) in zip(axes, FRAMERATE.items()):
        fps = [60, 30, 20]
        x = range(len(fps))
        for i, (when, color) in enumerate((("before", GREY), ("after", ORANGE))):
            vals = [d[when][f] for f in fps]
            mean = [100 * sum(v) / len(v) / d["n"] for v in vals]
            lo = [m - 100 * min(v) / d["n"] for m, v in zip(mean, vals)]
            hi = [100 * max(v) / d["n"] - m for m, v in zip(mean, vals)]
            ax.bar([p + (i - 0.5) * 0.38 for p in x], mean, 0.38, color=color, yerr=[lo, hi], capsize=4,
                   label=f"{when} the fix")
        ax.set_xticks(list(x), [f"{f} fps" + (" (every frame)" if f == 60 else f" (every {60 // f})") for f in fps])
        ax.set_title(f"{'First' if gym == 'gym1' else 'Second'} gym, dev ({d['n']} shots)")
        ax.set_ylim(80, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("shots called right (%)")
    axes[1].legend(loc="lower left")
    fig.suptitle("Accuracy at the phone's frame rates (bars: mean over frame offsets, whiskers: min-max)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_framerate.png", dpi=130)


def parity():
    names = list(PARITY)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (k, title, fmt) in zip(axes, ((0, "Mean IoU with PyTorch's box (1 = identical)", "{:.3f}"),
                                          (2, "Largest confidence difference", "{:.3f}"),
                                          (3, "Largest box-corner difference (px)", "{:.1f}"))):
        vals = [PARITY[n][k] for n in names]
        bars = ax.bar(names, vals, color=[NAVY, ORANGE, GREY])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom")
        ax.set_title(title, fontsize=10)
        if k == 0:
            ax.set_ylim(0.9, 1.01)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("The phone's detector vs PyTorch on 24 dev frames (60 boxes)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_parity.png", dpi=130)


def timing():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    labels = list(TIMING)
    left = [0.0] * len(labels)
    for i, (part, color) in enumerate((("letterbox (CPU)", GREY), ("Core ML model (Neural Engine)", ORANGE),
                                        ("decode + NMS (CPU)", NAVY))):
        vals = [TIMING[l][i] for l in labels]
        ax.barh(labels, vals, left=left, color=color, label=part)
        left = [a + b for a, b in zip(left, vals)]
    for y, l in enumerate(labels):
        ax.text(left[y] + 0.5, y, f"{TIMING[l][3]:.0f} detector fps", va="center")
    ax.set_xlabel("milliseconds per frame (stages overlap: the next frame is letterboxed while the model runs)")
    ax.set_xlim(0, 48)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_timing.png", dpi=130)


def confusion():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.imshow(CONFUSION, cmap="Oranges")
    for r, row in enumerate(CONFUSION):
        for c, v in enumerate(row):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14, color="white" if v > 20 else "black")
    ax.set_xticks([0, 1, 2], ["called made", "called missed", "not found"])
    ax.set_yticks([0, 1], ["made (53)", "missed (37)"])
    ax.set_ylabel("hand label")
    ax.set_title("First-look test, second gym (90 shots)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_test_confusion.png", dpi=130)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    framerate()
    parity()
    timing()
    confusion()
    print(f"wrote {OUT}/fig_*.png")
