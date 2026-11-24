"""Exercise physics, continuous control, real gradient steps and model roundtrip."""
import math
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sb3_contrib import CrossQ
from experiment import make_env


def check_lateral_starts():
    standard = make_env()
    lateral = make_env(lateral_start=True)
    sides = set()
    for seed in range(64):
        original_obs, _ = standard.reset(seed=seed)
        obs, info = lateral.reset(seed=seed)
        repeated_obs, repeated_info = lateral.reset(seed=seed)
        np.testing.assert_array_equal(obs, repeated_obs)
        assert repeated_info == info
        assert lateral.observation_space.contains(obs)
        np.testing.assert_array_equal(obs[[0, 1, 4, 5, 6, 7]], original_obs[[0, 1, 4, 5, 6, 7]])

        original_vx, original_vy = standard.unwrapped.lander.linearVelocity
        vx, vy = lateral.unwrapped.lander.linearVelocity
        assert math.isclose(math.hypot(vx, vy), math.hypot(original_vx, original_vy), abs_tol=1e-6)
        assert abs(vy) <= abs(vx) + 1e-6
        assert info["start_side"] == ("left" if vx < 0 else "right")
        sides.add(info["start_side"])
        shaping = standard.unwrapped.prev_shaping + 100 * (
            math.hypot(*original_obs[2:4]) - math.hypot(*obs[2:4])
        )
        np.testing.assert_allclose(lateral.unwrapped.prev_shaping, shaping, atol=2e-5, rtol=0)
    assert sides == {"left", "right"}
    standard.close()
    lateral.close()
    print("Passed: seeded lateral starts within 45°, preserved speed and consistent initial shaping.")


def main():
    check_lateral_starts()
    torch.set_num_threads(1)
    env = make_env()
    obs, _ = env.reset(seed=13)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, _ = env.step(np.array([0.5, -0.6], dtype=np.float32))
    assert obs.shape == (8,) and np.isfinite(r) and not (term or trunc)
    m = CrossQ("MlpPolicy", env, policy_kwargs=dict(net_arch=[32, 32]), buffer_size=500,
               batch_size=32, learning_starts=32, seed=13, device="cpu")
    before = next(m.critic.parameters()).detach().clone()
    m.learn(160)
    assert m._n_updates == 128 and not torch.equal(before, next(m.critic.parameters()))
    a = m.predict(obs, deterministic=True)[0]
    assert env.action_space.contains(a)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.zip"
        m.save(p)
        np.testing.assert_array_equal(a, CrossQ.load(p).predict(obs, deterministic=True)[0])
    print("Passed: reset, Box(2) step, finite reward, 128 CrossQ updates, identical action after reload.")
    env.close()


if __name__ == "__main__":
    main()
