"""Shared experiment protocol. All seeds are fixed before checkpoint selection."""
import csv
import hashlib
import json
from pathlib import Path
from copy import deepcopy

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_SEED = 19
VALIDATION_SEEDS = tuple(range(4100, 4110))
TEST_SEEDS = tuple(range(8200, 8250))
DEMO_SEED = 12000
CONFIG = dict(learning_rate=0.001, learning_starts=10_000, buffer_size=200_000,
              batch_size=128, gamma=0.99, train_freq=4, gradient_steps=1,
              policy_delay=3, ent_coef="auto", policy_kwargs=dict(net_arch=[128, 128]))


def training_config(profile):
    """Return the fixed optimizer settings for one of the three recorded runs."""
    if profile not in {"sparse", "standard", "step-cost"}:
        raise ValueError(f"Unknown training profile: {profile}")
    return dict(deepcopy(CONFIG), train_freq=1 if profile == "standard" else 4)


class StepCost(gym.RewardWrapper):
    """Charge one extra point per interaction during training only."""
    def reward(self, reward):
        return float(reward) - 1.0


def make_env(render_mode=None):
    env = gym.make("LunarLander-v3", continuous=True, render_mode=render_mode)
    assert env.observation_space.shape == (8,)
    assert isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape == (2,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    return env


def outcome(env, obs, reward, terminated, truncated):
    """Use the pinned implementation's terminal branch order (sleep wins ties)."""
    body = env.unwrapped
    if terminated:
        if not body.lander.awake:
            assert reward == 100
            return "landed"
        assert reward == -100
        if body.game_over:
            return "crash"
        if abs(float(obs[0])) >= 1 - 1e-6:
            return "out_of_bounds"
        raise RuntimeError("Unrecognized terminal condition")
    if truncated:
        return "time_limit"
    return "flying"


def episode(env, seed, model=None, frame_callback=None):
    obs, _ = env.reset(seed=int(seed))
    env.action_space.seed(int(seed) + 100_000)
    total, steps, status = 0.0, 0, "flying"
    if frame_callback:
        frame_callback(env, total, steps, status)
    while status == "flying":
        action = env.action_space.sample() if model is None else model.predict(obs, deterministic=True)[0]
        obs, reward, term, trunc, _ = env.step(action)
        total += float(reward)
        steps += 1
        status = outcome(env, obs, reward, term, trunc)
        if frame_callback:
            frame_callback(env, total, steps, status)
    return dict(policy="random" if model is None else "crossq", seed=int(seed), reward=total,
                length=steps, outcome=status, final_reward=float(reward),
                terminated=bool(term), truncated=bool(trunc))


def summarize(rows):
    rewards = np.array([r["reward"] for r in rows])
    lengths = np.array([r["length"] for r in rows])
    counts = {k: sum(r["outcome"] == k for r in rows) for k in ("landed", "crash", "out_of_bounds", "time_limit")}
    return dict(n=len(rows), mean_reward=float(rewards.mean()), sd_reward=float(rewards.std(ddof=1)),
                median_reward=float(np.median(rewards)), mean_length=float(lengths.mean()),
                median_length=float(np.median(lengths)), sd_length=float(lengths.std(ddof=1)),
                counts=counts, landing_rate=counts["landed"]/len(rows), crash_rate=counts["crash"]/len(rows),
                fraction_return_ge_200=float(np.mean(rewards >= 200)))


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
