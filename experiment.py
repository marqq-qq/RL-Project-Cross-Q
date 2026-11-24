"""Shared experiment protocol. All seeds are fixed before checkpoint selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
from copy import deepcopy

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_SEED = 19
VALIDATION_SEEDS = tuple(range(4100, 4110))
TEST_SEEDS = tuple(range(8200, 8250))
LATERAL_TEST_SEEDS = tuple(range(9200, 9250))
LATERAL_HALF_ANGLE = 45.0
DEMO_SEED = 12000
CONFIG = dict(learning_rate=0.001, learning_starts=10_000, buffer_size=200_000,
              batch_size=128, gamma=0.99, train_freq=4, gradient_steps=1,
              policy_delay=3, ent_coef="auto", policy_kwargs=dict(net_arch=[128, 128]))


def training_config(profile):
    """Return the fixed optimizer settings for one of the three recorded runs."""
    if profile not in {"sparse", "standard", "step-cost"}:
        raise ValueError(f"Unknown training profile: {profile}")
    return dict(deepcopy(CONFIG), train_freq=1 if profile == "standard" else 4)


class LateralStart(gym.Wrapper):
    """Rotate the post-reset velocity into one of the two lateral sectors."""

    def reset(self, *, seed=None, options=None):
        from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W

        obs, info = self.env.reset(seed=seed, options=options)
        lander_env = self.env.unwrapped
        lander = lander_env.lander
        vx, vy = lander.linearVelocity
        speed = math.hypot(vx, vy)
        side = int(self.np_random.choice((-1, 1)))
        offset = math.radians(self.np_random.uniform(-LATERAL_HALF_ANGLE, LATERAL_HALF_ANGLE))
        dx = side * speed * math.cos(offset) - vx
        dy = speed * math.sin(offset) - vy

        # Move the whole assembly together so the joints retain their relative velocities.
        for body in (lander, *lander_env.legs):
            body_vx, body_vy = body.linearVelocity
            body.linearVelocity = (body_vx + dx, body_vy + dy)

        # Match Gymnasium 1.2.2's observation scaling and initial reward shaping.
        x_scale = VIEWPORT_W / SCALE / 2 / FPS
        y_scale = VIEWPORT_H / SCALE / 2 / FPS
        actual_vx, actual_vy = lander.linearVelocity
        obs[2:4] = (actual_vx * x_scale, actual_vy * y_scale)
        lander_env.prev_shaping += 100 * (
            math.hypot(vx * x_scale, vy * y_scale)
            - math.hypot(actual_vx * x_scale, actual_vy * y_scale)
        )
        info = dict(info, start_side="left" if side < 0 else "right",
                    start_angle_deg=math.degrees(math.atan2(actual_vy, actual_vx)))
        return obs, info


class StepCost(gym.RewardWrapper):
    """Charge one extra point per interaction during training only."""
    def reward(self, reward):
        return float(reward) - 1.0


def make_env(render_mode=None, *, lateral_start=False):
    env = gym.make("LunarLander-v3", continuous=True, render_mode=render_mode)
    assert env.observation_space.shape == (8,)
    assert isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape == (2,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    return LateralStart(env) if lateral_start else env


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
    obs, reset_info = env.reset(seed=int(seed))
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
    result = dict(policy="random" if model is None else "crossq", seed=int(seed), reward=total,
                  length=steps, outcome=status, final_reward=float(reward),
                  terminated=bool(term), truncated=bool(trunc))
    if "start_side" in reset_info:
        result.update(start_side=reset_info["start_side"], start_angle_deg=reset_info["start_angle_deg"])
    return result


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
