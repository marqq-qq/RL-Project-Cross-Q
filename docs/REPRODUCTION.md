# Reproducing the Experiments

This document describes how to set up the project environment, run the notebook, inspect the recorded training experiments, and reproduce the saved policy's evaluation.

## 1. Requirements

The project uses Python 3.12.

The exact dependency versions are listed in `requirements.txt`:

- Gymnasium 1.2.2
- Box2D
- PyTorch 2.6.0
- Stable-Baselines3 2.7.1
- SB3-Contrib 2.7.1
- NumPy 1.26.4
- Matplotlib 3.11.2
- Pandas 3.0.6
- Jupyter and IPykernel

Box2D may require a C++ compiler on systems without a compatible wheel.

## 2. Environment setup

Run the commands from the repository root.

### Windows (PowerShell)

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install swig==4.5.1
python -m pip install -r requirements.txt
```

## 3. Run the notebook

Install the project Jupyter kernel:

```bash
python -m ipykernel install --sys-prefix --name crossq-team19 --display-name "CrossQ Team 19"
```

Launch Jupyter Lab:

```bash
python -m jupyter lab CrossQ_LunarLander_Final.ipynb
```

In Jupyter:

1. Select the `CrossQ Team 19` kernel.
2. Choose **Restart Kernel and Run All**.
3. Allow the notebook to execute its demonstrations and evaluation analysis.

The notebook uses the saved experiment artifacts in `results/`. It replays six short evaluation episodes and compares the first three matched test-seed pairs against the archived CSV results.

Small numerical differences between platforms may change individual trajectories.

## 4. Experiment configuration

The shared experiment protocol is defined in `experiment.py`.

### Environment

- Environment: `LunarLander-v3`
- Action mode: continuous
- Observation dimensions: 8
- Action dimensions: 2
- Action range: [-1, 1]
- Training seed: 19
- Device: CPU
- PyTorch threads: 1

### Seeds

| Purpose | Seeds | Count |
|---|---|---:|
| Training | 19 | 1 |
| Validation | 4100–4109 | 10 |
| Test | 8200–8249 | 50 |
| Lateral-start test | 9200–9249 | 50 |

The standard evaluation uses the default LunarLander environment. The lateral-start seeds are defined separately in `experiment.py`.

### Training profiles

Three profiles were recorded. Each run used 200,000 environment steps.

| Profile | Train frequency | Step cost | Best checkpoint |
|---|---:|---:|---:|
| Sparse | 4 | 0 | 120,000 |
| Standard | 1 | 0 | 180,000 |
| Step-cost | 4 | 1 | 200,000 |

The step-cost profile subtracts one additional reward point per interaction during training. Validation and test scores use the original Gymnasium rewards.

The three runs used the same environment and initialization seed. They are not independent training-seed replications.

## 5. Checkpoint selection

Checkpoints were evaluated every 10,000 training steps using the ten fixed validation seeds.

The selection criterion was the highest mean original return across all recorded runs and checkpoints.

Tie-breaking rules:

1. Within a run, an earlier checkpoint wins a tie.
2. Across runs, profile order is `sparse`, `standard`, then `step-cost`.

The selected checkpoint is:

- Profile: `standard`
- Training steps: 180,000
- Validation mean return: 280.91
- File: `models/best_model.zip`

Its SHA-256 hash is:

```text
04c045d5629bfcf359a753ae4c087a684f27d0782d8e36cb006538c3540b0035
```

The notebook verifies this hash before evaluating the saved model.

## 6. Evaluation protocol

The selected CrossQ policy is evaluated against a uniform random-action baseline.

Both policies use the same 50 test seeds, so each pair starts from the same seeded environment initialization.

- CrossQ uses deterministic policy predictions.
- The random baseline samples uniformly from the environment's action space.
- The random action space is seeded using the environment seed plus 100,000.
- Each episode runs until landing, crashing, leaving the horizontal boundary, or reaching the time limit.

The evaluation implementation is shared in `experiment.py`, particularly the `episode()`, `outcome()`, and `summarize()` functions.

## 7. Recorded results

The archived evaluation contains 50 episodes per policy.

| Metric | CrossQ | Random |
|---|---:|---:|
| Mean return | 274.59 | -218.12 |
| Median return | 280.13 | -217.06 |
| Mean episode length | 264.08 | 104.16 |
| Successful landings | 48/50 | 0/50 |
| Crashes | 0 | 30 |
| Out of bounds | 0 | 20 |
| Time limits | 2 | 0 |

The mean paired return gain was 492.72 points. All 50 paired comparisons favored CrossQ.

The paired bootstrap 95% confidence interval was [459.22, 527.03].

A successful landing is identified using the simulator's lander-body sleep flag.

## 8. Experiment artifacts

The following files contain the recorded experiment evidence:

| File | Contents |
|---|---|
| `results/experiments.json` | Training profiles and checkpoint-selection results |
| `results/training_summary.json` | Selected run configuration, seed, and checkpoint metadata |
| `results/evaluation_results.json` | Aggregate evaluation metrics |
| `results/evaluation_results.csv` | Per-episode evaluation results |
| `results/training_curve.png` | Training and validation curves |
| `results/comparison.png` | CrossQ and random-policy comparison |
| `models/best_model.zip` | Selected 180,000-step checkpoint |
| `models/checkpoint_100000.zip` | Earlier 100,000-step checkpoint |

`experiment.py` contains shared experiment utilities and configuration. The recorded results and notebook provide the experiment protocol and analysis; running the notebook does not retrain all three 200,000-step profiles.

## 9. Reproducibility limitations

Exact trajectory-level reproduction can depend on the operating system, hardware, numerical libraries, and simulator behavior.

The recorded result describes one selected policy trained with one initialization seed. The three training profiles do not establish variability across independent training seeds.

Independent training-seed replications and a matched-budget comparison against another algorithm would be needed for stronger claims about training variability and the contribution of CrossQ itself.