# CrossQ for continuous lunar landing

**Team 19 · Reinforcement Learning final project**

The selected policy landed on **48 of 50** held-out seeds. Its mean return was **274.59**, compared with **−218.12** for uniform random actions. Both policies flew the same initial conditions.

[Local flight console](#local-flight-console) · [Live notebook](CrossQ_LunarLander_Final.ipynb) · [Defense video](videos/demo_final.mp4) · [Editable slides](Final_Presentation.pptx) · [Team defense guide](DEFENSE_GUIDE.md) · [Experiment report](REPORT.md)

For the **30–35 minute group defense**, start with the five-slide `Final_Presentation.pptx` or PDF. After slide 5, play `videos/demo_final.mp4` (56.16 seconds, full HD), then present `CrossQ_LunarLander_Final.ipynb`. Its five live sections cover the simulator, CrossQ, a numerical PyTorch update, training code and held-out evaluation. Open `CrossQ_LunarLander_Final.html` for the offline rendered copy. `DEFENSE_GUIDE.md` assigns the five presenters, budgets 12 minutes for the notebook and includes likely questions. The slide notes also appear in `presentation_script.md`.

## Setup

Use Python 3.12. The reference run used CPython 3.12.7 on Apple Silicon. Box2D 2.3.5 builds a native extension on systems without a compatible wheel. Install a C++ compiler appropriate for your OS (Xcode command-line tools on macOS, build-essential on Debian/Ubuntu, or Visual C++ Build Tools on Windows). Clone the repository and run the setup commands from its root directory.

```bash
git clone https://github.com/marqq-qq/RL-Project-Cross-Q.git
cd RL-Project-Cross-Q
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install swig==4.5.1
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name crossq-team19 --display-name "CrossQ · Team 19"
python tests/smoke.py
```

On Windows use `py -3.12 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. Keep the environment active during installation so the Box2D build can find SWIG. The video writer uses the FFmpeg binary supplied by imageio-ffmpeg.

## Local flight console

Complete [Setup](#setup), then run these commands from the repository root.

```bash
source .venv/bin/activate
python -m demo_ui.server
```

Open [127.0.0.1:8765](http://127.0.0.1:8765/). The console defaults to the 180,000-step
checkpoint. Select the 100,000-step checkpoint to try the earlier policy. Manual mode sets
the initial force vector, lander tilt, and environment seed. Random mode uses Gymnasium's
seeded random launch force. The console displays observations, actions, and flight telemetry.
Model results shows the recorded held-out evaluation for the selected checkpoint when
available. Expand **Model details** to see the checkpoint filename and SHA-256.
Use `--port 8766` if port 8765 is busy.

## Open the live notebook

From the repository root, activate the environment and launch the notebook with the following command. The notebook uses the **CrossQ · Team 19** kernel registered in Setup. If reusing an existing environment, run the kernel-registration command once before opening Jupyter. **Restart Kernel and Run All** reproduces the walkthrough and six held-out flights. The saved HTML opens offline in a browser. To regenerate that reading copy after saving the executed notebook, run `python presentation/export_notebook.py`.

```bash
python -m jupyter lab CrossQ_LunarLander_Final.ipynb
```

## Use the submitted model

```bash
python evaluate.py --out reproduced/results
python record_demo_final.py --out reproduced/demo_final.mp4
python tests/verify_final_demo.py
python -m jupyter nbconvert --to notebook --execute CrossQ_LunarLander_Final.ipynb --output CrossQ_reexecuted.ipynb --ExecutePreprocessor.timeout=180
python tests/verify_submission.py
```

The default input model is `models/best_model.zip`. Evaluation uses `results/` for the training history and writes a JSON summary, a per-episode CSV and two plots. A demo writes its own episode manifest beside the MP4. The verification script replays all 100 held-out episodes, checks the archived numbers and model identity, validates notebook outputs, reads both decks and decodes every video frame. On other CPU, Box2D or PyTorch builds, small numerical differences can eventually change a trajectory. The strict replay audit is intended for the reference platform.

## Defense video

Play **`videos/demo_final.mp4`** locally. The 1920 × 1080 H.264 film runs at 25 fps, with silent, real-time playback and the complete Gymnasium viewport. The sidebar shows the actual step, cumulative return and engine power. Nikita’s timed speaking cues are in `videos/demo_narration.md`. The earlier 49.88-second recording and its original manifest remain at `videos/demo.mp4` and `videos/demo.json`.

Run `python record_demo_final.py` to regenerate the defense film and `videos/demo_final.json` from the two archived policies. For a separate copy use the `--out` command above. The manifest records SHA-256 hashes, per-flight outcomes, scene boundaries and the source of the end-card statistics. `python tests/verify_final_demo.py` checks the MP4 against those files and fresh simulator frames.

## Retrain from scratch

```bash
python train.py --profile standard --run runs/reproduction --steps 200000 --interval 10000
python evaluate.py --model runs/reproduction/export/models/best_model.zip --training runs/reproduction/export/results --out runs/reproduction/test
python record_demo.py --model runs/reproduction/export/models/best_model.zip --intermediate runs/reproduction/step_0100000/model.zip --evaluation runs/reproduction/test/evaluation_results.json --out runs/reproduction/demo.mp4
```

The submission scripts resolve project inputs relative to their own files. Training runs headless with the official `sb3_contrib.CrossQ`. Settings are in `experiment.py`. The single environment is always created with `gym.make("LunarLander-v3", continuous=True)`; assertions check the two-dimensional Box action space. The export directory contains the selected model, training episode returns and lengths, checkpoint throughput, validation episodes and run summary. The run directory retains intermediate models and the latest replay buffer. The ZIP includes the selected model and the 100k intermediate model used for the film.

To continue an interrupted run, repeat the training command with `--resume`. For example, to extend the same run to a cumulative 240,000 steps:

```bash
python train.py --profile standard --run runs/reproduction --resume --steps 240000 --interval 10000
```

A completed block writes its model, optimizers, replay data, validation history and logs before publishing `latest.json`. Restarting begins a new seeded physics episode. The Box2D world and exact random generator positions are not checkpointed, so a restarted trajectory can diverge from an uninterrupted run. The bundled model can fly immediately. Continuing *the archived run* with its exact replay history requires the local training run directory, which is excluded from the submission ZIP for size.

To reproduce model selection across all three completed runs, use the training commands below in addition to the standard run above. The selector only reads training exports and validation histories.

```bash
python train.py --profile sparse --run runs/pilot
python train.py --profile step-cost --run runs/step_cost
python select_model.py --pilot runs/pilot --standard runs/reproduction --step-cost runs/step_cost --out runs/selected
python evaluate.py --model runs/selected/models/best_model.zip --training runs/selected/results --out runs/selected/test
```

## Experimental protocol

- **Training:** three fresh, 200,000-step runs from seed 19. Each starts with 10,000 uniform actions and uses one PyTorch CPU thread. Networks, buffer, batch size, learning rate and environment settings are shared. The `sparse` pilot uses one critic update per four steps. The `standard` run uses one per step. The `step-cost` run uses the pilot update rate and subtracts 1 from each training reward. An inner Monitor records original episode returns for all training curves.
- **Selection:** every 10,000 steps, evaluate deterministic actions on seeds 4100–4109 with original rewards. Select the maximum mean over all runs and checkpoints. Earlier checkpoints win exact ties within a run. `results/experiments.json` records every run’s best score, selected step and log location. To reproduce the other configurations, substitute `--profile sparse --run runs/pilot` or `--profile step-cost --run runs/step_cost` in the training command.
- **Test:** use seeds 8200–8249 once the checkpoint is fixed. Both policies receive the same reset seed per pair. The uniform random policy seeds its Box sampler with the environment seed plus 100,000. Sample standard deviations use N−1. The paired interval resamples 50 seed pairs 10,000 times with RNG seed 771.
- **Video:** two uniform random flights (12000–12001), a genuine 100k standard-run checkpoint (12000), and three flights of the selected 180k policy (12000–12002). All are complete Gymnasium episodes. The 50 Hz simulation is sampled at 25 fps. Short terminal labels and a held-out results card give the viewer time to read. The manifest records frame boundaries and model, evaluation and media digests.
- **Landing definition:** classify a sleeping Box2D body as landed. Otherwise distinguish hull contact, horizontal exit and time-limit-only truncation. A simultaneous true termination takes precedence over truncation. Outcome, terminal reward, flags, length and full episode return are logged.

## Project files

| File | Use |
|---|---|
| `train.py`, `experiment.py` | Train, select and checkpoint. Centralize seeds, environment creation and outcome logic. |
| `evaluate.py`, `record_demo_final.py`, `record_demo.py` | Run matched-seed comparisons, render the defense film, or reproduce the backup recording. |
| `results/training_episodes.csv`, `training_progress.csv`, `validation_episodes.csv` | Inspect every completed training episode, chunk timing and checkpoint score. |
| `results/evaluation_results.csv`, `evaluation_results.json` | Inspect the paired test episodes and aggregated metrics. |
| `results/training_curve.png`, `comparison.png` | Inspect raw training returns, their smoother, validation means and paired test returns. |
| `models/best_model.zip`, `models/checkpoint_100000.zip`, `videos/demo_final.json` | Load the selected and intermediate policies. Match each filmed flight to its seed, weights and frame range. |
| `presentation_script.md`, `presentation/build.py` | Rehearse or regenerate the editable talk from archived data. |
| `requirements.txt`, `results/environment.txt` | Recreate core versions or inspect the complete reference package inventory. |

To regenerate the deck, run `python record_demo.py --stills presentation/stills` and then `python presentation/build.py`. Export that PPTX to PDF in PowerPoint or with LibreOffice (`soffice --headless --convert-to pdf --outdir . Final_Presentation.pptx`).

## Sources

- Bhatt et al., *CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity*, ICLR 2024. [Conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html)
- Haarnoja et al., *Soft Actor-Critic*, ICML 2018. [Proceedings](https://proceedings.mlr.press/v80/haarnoja18b.html)
- SB3-Contrib algorithm documentation and the implementation used. [Documentation](https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html) · [CrossQ 2.7.1 source](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/crossq/crossq.py)
- Farama Foundation, environment and reward rules. [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) · [Gymnasium 1.2.2 source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.2/gymnasium/envs/box2d/lunar_lander.py)
- RL Baselines3 Zoo, continuous LunarLander CrossQ starting configuration. [Hyperparameter file](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/crossq.yml)
