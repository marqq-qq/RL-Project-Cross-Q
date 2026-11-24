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


REPLAY_WARMUP = 5_000
FINE_TUNE_RATE = 0.0003


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


def train(run, budget, interval, resume, profile, *, lateral_start=False, initial_model=None):
    config = training_config(profile)
    run.mkdir(parents=True, exist_ok=True)
    pointer = run / "latest.json"
    if pointer.exists() != resume:
        raise ValueError("Choose an unused run directory for a new run, or --resume for an existing checkpoint.")
    if min(budget, interval) <= 0 or budget % config["train_freq"] or interval % config["train_freq"]:
        raise ValueError("Use positive budgets and intervals divisible by train_freq.")
    if resume and initial_model is not None:
        raise ValueError("--initial-model starts a new run. Use --resume on its own to continue.")
    torch.set_num_threads(1)
    env = Monitor(make_env(lateral_start=lateral_start))
    if profile == "step-cost":
        env = StepCost(env)  # Monitor sits inside: training CSV still records original returns.
    validation = make_env(lateral_start=lateral_start)
    rows, history, validation_rows = [], [], []
    previous = None
    warmup_remaining = 0
    source = None
    if resume:
        previous = run / json.loads(pointer.read_text())["directory"]
        state = json.loads((previous / "state.json").read_text())
        if state.get("profile", profile) != profile:
            raise ValueError("Resume using the original training profile.")
        if state.get("lateral_start", False) != lateral_start:
            raise ValueError("Resume with the original lateral-start setting.")
        source = state.get("initial_model")
        if source is not None:
            config["learning_rate"] = FINE_TUNE_RATE
        rows, history, validation_rows = state["episodes"], state["history"], state["validation"]
        model = CrossQ.load(previous / "model.zip", env=env, device="cpu")
        if model.train_freq.frequency != config["train_freq"]:
            raise ValueError("Resume with the same --profile as the saved run.")
        model.load_replay_buffer(previous / "replay.pkl")
        # Fresh physics episode on restart. Preserve weights, optimizers, replay and counts.
        env.reset(seed=TRAIN_SEED + model.num_timesteps)
        model.get_env().seed(TRAIN_SEED + model.num_timesteps)
    elif initial_model is not None:
        config["learning_rate"] = FINE_TUNE_RATE
        model = CrossQ.load(initial_model, env=env, device="cpu", learning_rate=FINE_TUNE_RATE)
        if model.train_freq.frequency != config["train_freq"] or profile == "step-cost":
            raise ValueError("Use the source model's update rate and unmodified rewards.")
        if model.num_timesteps < model.learning_starts:
            raise ValueError("The source model must have completed its original warm-up.")
        if REPLAY_WARMUP % config["train_freq"] or min(interval, budget - model.num_timesteps) <= REPLAY_WARMUP:
            raise ValueError("The first block must include replay warm-up and gradient updates.")
        warmup_remaining = REPLAY_WARMUP
        source = dict(sha256=digest(initial_model), steps=model.num_timesteps, replay_warmup=REPLAY_WARMUP)
    else:
        model = CrossQ("MlpPolicy", env, seed=TRAIN_SEED, device="cpu", **config)
    if budget < model.num_timesteps:
        raise ValueError("Budget is the final cumulative step count.")
    callback = EpisodeLog(rows)
    while model.num_timesteps < budget:
        block_steps = min(interval, budget - model.num_timesteps)
        n = block_steps
        start = time.perf_counter()
        if warmup_remaining:
            # Fill the fresh replay buffer using the loaded policy before updating its weights.
            print(f"Collecting {warmup_remaining:,} transitions with the loaded policy", flush=True)
            gradient_steps = model.gradient_steps
            model.gradient_steps = 0
            try:
                model.learn(total_timesteps=warmup_remaining, reset_num_timesteps=False,
                            callback=callback, log_interval=None)
            finally:
                model.gradient_steps = gradient_steps
            n -= warmup_remaining
            warmup_remaining = 0
        model.learn(total_timesteps=n, reset_num_timesteps=False, callback=callback, log_interval=None)
        elapsed = time.perf_counter() - start
        step = model.num_timesteps
        block = run / f"step_{step:07d}"
        block.mkdir(exist_ok=True)
        evaluation = [episode(validation, s, model) for s in VALIDATION_SEEDS]
        validation_rows.extend(dict(step=step, **r) for r in evaluation)
        stats = summarize(evaluation)
        history.append(dict(step=step, updates=model._n_updates, seconds_learning=elapsed,
                            steps_per_second=block_steps/elapsed, validation_mean=stats["mean_reward"],
                            validation_landings=stats["counts"]["landed"]))
        best = max(history, key=lambda h: h["validation_mean"])
        model.save(block / "model.zip")
        model.save_replay_buffer(block / "replay.pkl")
        shutil.copyfile(block / "model.zip" if best["step"] == step else previous / "best.zip", block / "best.zip")
        save_json(block / "state.json", dict(profile=profile, lateral_start=lateral_start, initial_model=source,
                                             episodes=rows, history=history, validation=validation_rows))
        save_json(pointer, dict(directory=block.name))
        print(f"{step:7d} steps | {block_steps/elapsed:.0f} steps/s | validation {stats['mean_reward']:.1f} | landed {stats['counts']['landed']}/{len(evaluation)}", flush=True)
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
              lateral_start=lateral_start, initial_model=source,
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
    p.add_argument("--lateral-start", action="store_true", help="Use left/right starts within 45° of horizontal.")
    p.add_argument("--initial-model", type=Path, help="Initialize from model weights and optimizer state with fresh replay.")
    args = p.parse_args()
    train(args.run, args.steps, args.interval, args.resume, args.profile,
          lateral_start=args.lateral_start, initial_model=args.initial_model)
