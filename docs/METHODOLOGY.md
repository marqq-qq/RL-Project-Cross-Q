# Training Methodology and Evaluation

## 1. Overview

This project applies the CrossQ reinforcement learning algorithm to the continuous-control LunarLander-v3 environment.

The objective is to train an agent that can control a lunar lander, manage its engines, stabilize its movement, and achieve successful landings.

The project uses three separate training configurations. Their checkpoints are evaluated on fixed validation seeds, and the selected policy is compared against a uniform random-action baseline on previously unseen test seeds.

## 2. Environment

We use Gymnasium's `LunarLander-v3` environment with continuous actions enabled.

The agent receives an observation consisting of eight values:

- Horizontal position and vertical position.
- Horizontal and vertical velocity.
- Angular position and angular velocity.
- Contact indicators for the two landing legs.

The agent produces two continuous actions, both ranging from -1 to +1:

- Main engine control.
- Lateral engine control.

The environment provides rewards based on the lander's position, velocity, angle, and leg contact. Engine usage also incurs a penalty.

An episode can finish in several ways:

- **Landed:** The lander body enters the sleeping state, receiving a terminal reward of +100.
- **Crash:** The hull contacts the ground, resulting in a terminal reward of -100.
- **Out of bounds:** The lander leaves the horizontal boundaries, resulting in a terminal reward of -100.
- **Time limit:** The episode reaches the 1,000-step limit.

## 3. CrossQ Algorithm

CrossQ is an off-policy actor-critic reinforcement learning algorithm based on the Soft Actor-Critic (SAC) framework.

It uses:
- An actor network that generates continuous actions.
- Two critic networks that estimate state-action values.
- A replay buffer to reuse previously collected transitions.
- An entropy objective to encourage exploration.
- Automatic entropy coefficient adjustment.

A key feature of CrossQ is its use of Batch Renormalization (BatchRenorm) across jointly processed current and successor state-action batches. The implementation uses a momentum of 0.01 and epsilon of 0.001.

Unlike conventional SAC implementations, CrossQ does not maintain separate slowly updated target critic networks. Instead, it uses the online critics and joint batch normalization to construct its critic targets.

The algorithm therefore combines off-policy learning with a different approach to stabilizing critic updates.

## 4. Training Configuration

The project uses the `CrossQ` implementation from SB3-Contrib.

The main training parameters are:

| Parameter | Value |
|---|---:|
| Learning rate | 0.001 |
| Learning starts | 10,000 steps |
| Replay buffer size | 200,000 |
| Batch size | 128 |
| Discount factor (gamma) | 0.99 |
| Gradient steps | 1 |
| Policy delay | 3 |
| Entropy coefficient | Automatic |
| Network architecture | [128, 128] |
| Training seed | 19 |

The actor and each critic use multilayer perceptrons with two hidden layers of 128 units.

Training uses a CPU device and one PyTorch thread.

## 5. Training Experiments

Three configurations were trained using the same environment and initialization seed.

Each configuration was trained for 200,000 environment steps, resulting in 600,000 total training steps across all three runs.

| Profile | Training frequency | Additional reward cost |
|---|---:|---:|
| Sparse | 4 | 0 |
| Standard | 1 | 0 |
| Step-cost | 4 | 1 per interaction |

The standard configuration performs an update after every environment step, while the other two configurations perform updates every four environment steps.

The step-cost configuration additionally subtracts one point from each training interaction's reward.

Validation and test evaluations use the original environment rewards, without this additional training cost.

### Training results

The three configurations were compared using their best validation checkpoints.

| Profile | Best checkpoint | Mean validation return |
|---|---:|---:|
| Sparse | 120,000 steps | 2.02 |
| Standard | 180,000 steps | 280.91 |
| Step-cost | 200,000 steps | 259.20 |

The standard configuration achieved the highest validation return and was selected for the final evaluation.

## 6. Checkpoint Selection

Checkpoint selection was performed using ten fixed validation seeds:

`4100–4109`

Every 10,000 training steps, the current model was evaluated on these seeds.

The checkpoint with the highest mean original environment return was selected.

The selected checkpoint is:

- Training profile: Standard
- Training step: 180,000
- Mean validation return: 280.91
- Model file: `models/best_model.zip`

This selection procedure was applied across all three training configurations.

## 7. Evaluation Protocol

The selected checkpoint was evaluated on 50 held-out test seeds:

`8200–8249`

The same test seeds were used for both policies to provide matched initial conditions.

Two policies were compared:

1. **CrossQ policy:** Uses deterministic actions predicted by the trained actor.
2. **Random policy:** Samples uniformly from the continuous action space.

The random action generator uses a seed offset of 100,000 to keep its action sampling reproducible.

For every episode, the evaluation records:

- Total episode reward.
- Episode length.
- Final outcome.
- Termination and truncation status.

The evaluation reports mean and median returns, episode lengths, landing rates, and outcome counts.

## 8. Evaluation Results

The selected CrossQ checkpoint achieved the following results:

| Metric | CrossQ | Random |
|---|---:|---:|
| Episodes | 50 | 50 |
| Successful landings | 48 | 0 |
| Landing rate | 96% | 0% |
| Mean reward | 274.59 | -218.12 |
| Median reward | 280.13 | -217.06 |
| Mean episode length | 264.08 | 104.16 |

The CrossQ policy achieved 48 successful landings out of 50 episodes. The remaining two episodes ended due to the time limit.

The random policy produced 30 crashes and 20 out-of-bounds episodes.

The mean paired reward improvement of CrossQ over the random baseline was 492.72 points.

A paired bootstrap analysis produced a 95% confidence interval of [459.22, 527.03].

## 9. Reproducibility

The experiment uses fixed training, validation, and test seeds to make the evaluation procedure reproducible.

The repository stores:
- Training configuration and checkpoint-selection information in `results/experiments.json`.
- Selected training run details in `results/training_summary.json`.
- Evaluation statistics in `results/evaluation_results.json`.
- Per-episode evaluation records in `results/evaluation_results.csv`.

The selected model's SHA-256 hash is also recorded to verify that the evaluated checkpoint matches the saved training results.

The notebook can be executed using **Restart Kernel and Run All** to reproduce the demonstration calculations and replay selected evaluation episodes.

## 10. Limitations

Several limitations should be considered when interpreting the results:

- All three training configurations use the same initialization seed (19). Therefore, the experiments do not measure variability across independent training runs.
- The evaluation compares CrossQ against uniform random actions, not against another trained reinforcement learning algorithm such as SAC or TD3.
- The reported results describe one selected checkpoint under the default environment configuration.
- A successful landing is identified using the simulator's lander sleeping state. This does not independently verify that the lander is positioned within the center of the landing pad.

These limitations should be considered when drawing conclusions about the algorithm's performance and generalization.

## 11. Summary

The experiments demonstrate that CrossQ can learn an effective continuous-control policy for the LunarLander-v3 environment.

Among the three tested training configurations, the standard profile achieved the highest validation performance. Its 180,000-step checkpoint reached a 96% landing rate on 50 held-out test seeds, substantially outperforming the uniform random-action baseline.

The results demonstrate the effectiveness of the trained policy in the evaluated environment, while further experiments with independent training seeds and additional algorithm baselines would be needed for broader conclusions.