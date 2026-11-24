"""Paired held-out evaluation; creates a per-episode audit trail and plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sb3_contrib import CrossQ

from experiment import ROOT, TEST_SEEDS, LATERAL_TEST_SEEDS, digest, episode, make_env, save_csv, save_json, summarize

INK, BLUE, ORANGE = "#182736", "#106f7b", "#ad501e"


def charts(out, rows, training):
    out = Path(out)
    with (training / "training_episodes.csv").open() as f:
        log = list(csv.DictReader(f))
    with (training / "training_progress.csv").open() as f:
        progress = list(csv.DictReader(f))
    x = np.array([int(r["step"]) / 1000 for r in log])
    y = np.array([float(r["reward"]) for r in log])
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK, "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.scatter(x, y, s=5, alpha=.35, color="#7f95a7", label="Training episode")
    window = min(30, len(y))
    if window:
        ax.plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode="valid"), color=BLUE,
                lw=2.4, label=f"Trailing {window} episodes")
    ax.plot([int(r["step"])/1000 for r in progress], [float(r["validation_mean"]) for r in progress],
            color=ORANGE, marker="o", markersize=3, label="Deterministic validation mean (10 seeds)")
    ax.set(xlabel="Environment steps (thousands)", ylabel="Undiscounted episode return")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(out / "training_curve.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 4.6), layout="constrained")
    groups = [[r["reward"] for r in rows if r["policy"] == p] for p in ("random", "crossq")]
    for left, right in zip(*groups):
        ax.plot([0, 1], [left, right], color="#8a98a4", alpha=.24, lw=.8, zorder=1)
    for j, (values, color) in enumerate(zip(groups, (ORANGE, BLUE))):
        ax.scatter(np.full(len(values), j), values, color=color, alpha=.64, s=13, zorder=2)
        ax.plot([j-.12, j+.12], [np.mean(values)]*2, color=INK, lw=3, zorder=3)
    ax.set(xlim=(-.36, 1.36), xticks=[0, 1], xticklabels=["Uniform random", "Trained CrossQ"],
           ylabel="Undiscounted episode return", title="Two policies on the same initial conditions")
    ax.text(.5, .02, "Each line pairs one seed. Black bars show means.", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def evaluate(model_file, out, training, seeds=None, *, lateral_start=False):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    model = CrossQ.load(model_file, device="cpu")
    env = make_env(lateral_start=lateral_start)
    if seeds is None:
        seeds = LATERAL_TEST_SEEDS if lateral_start else TEST_SEEDS
    rows = []
    for s in seeds:
        for policy in (None, model):
            rows.append(episode(env, s, policy))
    env.close()
    summary = {p: summarize([r for r in rows if r["policy"] == p]) for p in ("random", "crossq")}
    differences = np.array([b["reward"]-a["reward"] for a, b in zip(rows[::2], rows[1::2])])
    # Resample whole seed pairs; this describes seed uncertainty for this one trained policy.
    boot = np.random.default_rng(771).choice(differences, size=(10_000, len(seeds))).mean(axis=1)
    record = dict(model_sha256=digest(model_file), model_steps=model.num_timesteps, seeds=list(seeds),
                  random_action_seed_offset=100_000, reward_sd_ddof=1, policies=summary,
                  mean_paired_gain=float(differences.mean()), positive_pairs=int((differences>0).sum()),
                  paired_bootstrap_95_ci=np.quantile(boot, [.025,.975]).tolist())
    if lateral_start:
        record["lateral_start"] = True
        by_side = {}
        for side in ("left", "right"):
            side_results = {}
            for policy in summary:
                flights = [r for r in rows if r["policy"] == policy and r["start_side"] == side]
                if len(flights) >= 2:
                    side_results[policy] = summarize(flights)
            by_side[side] = side_results
        record["policies_by_start_side"] = by_side
    save_csv(out / "evaluation_results.csv", rows)
    save_json(out / "evaluation_results.json", record)
    charts(out, rows, training)
    print(json.dumps(record["policies"], indent=2))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/best_model.zip")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--training", type=Path, default=ROOT / "results")
    p.add_argument("--lateral-start", action="store_true")
    a = p.parse_args()
    evaluate(a.model, a.out, a.training, lateral_start=a.lateral_start)
