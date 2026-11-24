"""Train official CrossQ. Each completed block commits a resumable checkpoint."""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import torch
from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from experiment import training_config, StepCost, ROOT, TRAIN_SEED, VALIDATION_SEEDS, digest, episode, make_env, save_csv, save_json, summarize


class EpisodeLog(BaseCallback):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                e = info["episode"]
                self.rows.append(dict(step=self.num_timesteps, reward=e["r"], length=e["l"]))
        return True


def train(run, budget, interval, resume, profile):
    config = training_config(profile)
    run.mkdir(parents=True, exist_ok=True)
    pointer = run / "latest.json"
    if pointer.exists() != resume:
        raise ValueError("Choose an unused run directory for a new run, or --resume for an existing checkpoint.")
    if min(budget, interval) <= 0 or budget % config["train_freq"] or interval % config["train_freq"]:
        raise ValueError("Use positive budgets and intervals divisible by train_freq.")
    torch.set_num_threads(1)
    env = Monitor(make_env())
    if profile == "step-cost":
        env = StepCost(env)  # Monitor sits inside: training CSV still records original returns.
    validation = make_env()
    rows, history, validation_rows = [], [], []
    previous = None
    if resume:
        previous = run / json.loads(pointer.read_text())["directory"]
        state = json.loads((previous / "state.json").read_text())
        if state.get("profile", profile) != profile:
            raise ValueError("Resume using the original training profile.")
        rows, history, validation_rows = state["episodes"], state["history"], state["validation"]
        model = CrossQ.load(previous / "model.zip", env=env, device="cpu")
        if model.train_freq.frequency != config["train_freq"]:
            raise ValueError("Resume with the same --profile as the saved run.")
        model.load_replay_buffer(previous / "replay.pkl")
        # Fresh physics episode on restart. Preserve weights, optimizers, replay and counts.
        env.reset(seed=TRAIN_SEED + model.num_timesteps)
        model.get_env().seed(TRAIN_SEED + model.num_timesteps)
    else:
        model = CrossQ("MlpPolicy", env, seed=TRAIN_SEED, device="cpu", **config)
    if budget < model.num_timesteps:
        raise ValueError("Budget is the final cumulative step count.")
    callback = EpisodeLog(rows)
    while model.num_timesteps < budget:
        n = min(interval, budget - model.num_timesteps)
        start = time.perf_counter()
        model.learn(total_timesteps=n, reset_num_timesteps=False, callback=callback, log_interval=None)
        elapsed = time.perf_counter() - start
        step = model.num_timesteps
        block = run / f"step_{step:07d}"
        block.mkdir(exist_ok=True)
        evaluation = [episode(validation, s, model) for s in VALIDATION_SEEDS]
        validation_rows.extend(dict(step=step, **r) for r in evaluation)
        stats = summarize(evaluation)
        history.append(dict(step=step, updates=model._n_updates, seconds_learning=elapsed,
                            steps_per_second=n/elapsed, validation_mean=stats["mean_reward"],
                            validation_landings=stats["counts"]["landed"]))
        best = max(history, key=lambda h: h["validation_mean"])
        model.save(block / "model.zip")
        model.save_replay_buffer(block / "replay.pkl")
        shutil.copyfile(block / "model.zip" if best["step"] == step else previous / "best.zip", block / "best.zip")
        save_json(block / "state.json", dict(profile=profile, episodes=rows, history=history, validation=validation_rows))
        save_json(pointer, dict(directory=block.name))
        print(f"{step:7d} steps | {n/elapsed:.0f} steps/s | validation {stats['mean_reward']:.1f} | landed {stats['counts']['landed']}/{len(evaluation)}", flush=True)
        if previous and previous != block:
            # Keep lightweight historical models; only the latest needs a replay buffer.
            for name in ("replay.pkl", "best.zip", "state.json"):
                (previous / name).unlink(missing_ok=True)
        previous = block
    best = max(history, key=lambda h: h["validation_mean"])
    export = run / "export"
    for folder in ("results", "models"):
        (export / folder).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(previous / "best.zip", export / "models/best_model.zip")
    results = export / "results"
    save_csv(results / "training_episodes.csv", rows)
    save_csv(results / "training_progress.csv", history)
    save_csv(results / "validation_episodes.csv", validation_rows)
    save_json(results / "training_summary.json", dict(steps=model.num_timesteps, updates=model._n_updates,
              training_seed=TRAIN_SEED, validation_seeds=VALIDATION_SEEDS, config=config, profile=profile, training_step_cost=1.0 if profile == "step-cost" else 0.0, interval=interval,
              seconds_learning=sum(h["seconds_learning"] for h in history), episodes=len(rows),
              selected_step=best["step"], selected_validation_mean=best["validation_mean"],
              model_sha256=digest(export / "models/best_model.zip"), device="cpu", torch_threads=1,
              platform=platform.platform(), architecture=platform.machine(), python=platform.python_version()))
    env.close()
    validation.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "runs/seed19")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--interval", type=int, default=10_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--profile", choices=("standard", "sparse", "step-cost"), default="standard")
    args = p.parse_args()
    train(args.run, args.steps, args.interval, args.resume, args.profile)
