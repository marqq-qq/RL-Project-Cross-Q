"""Exercise physics, continuous control, real gradient steps and model roundtrip."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sb3_contrib import CrossQ
from experiment import make_env


def main():
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
