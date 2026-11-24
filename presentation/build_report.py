"""Regenerate the report from the archived run and video manifests."""
import csv
from pathlib import Path
import json
import statistics

root=Path(__file__).resolve().parents[1]
result=root/"results"
tr=json.loads((result/"training_summary.json").read_text())
ev=json.loads((result/"evaluation_results.json").read_text())
demo=json.loads((root/"videos/demo_final.json").read_text())
with (result/"training_progress.csv").open() as f: progress=list(csv.DictReader(f))
with (result/"training_episodes.csv").open() as f: episodes=list(csv.DictReader(f))
pilot=json.loads((result/"pilot/training_summary.json").read_text())
experiments=json.loads((result/"experiments.json").read_text())
winner=next(x for x in experiments["runs"] if x["profile"]==experiments["selected_profile"])
runs_table=["| Profile | Training reward | Critic updates per step | Best validation mean | Step | Landings at that step | Learning time |","|---|---|---:|---:|---:|---:|---:|"]
for run in experiments["runs"]:
    desc="original − 1 per step" if run["step_cost"] else "original"
    runs_table.append(f"| {run['profile']} | {desc} | {1/run['train_freq']:.2f} | {run['validation_mean']:.2f} | {run['best_step']:,} | {run['best_landed']}/10 | {run['training_seconds']:.1f} s |")
design_table="\n".join(runs_table)
rd,cr=ev["policies"]["random"],ev["policies"]["crossq"]
lo,hi=ev['paired_bootstrap_95_ci']
comp=["| Measure | Uniform random | CrossQ |","|---|---:|---:|"]
for label,key in [('Mean return','mean_reward'),('Sample SD of return','sd_reward'),('Median return','median_reward'),('Mean episode length','mean_length'),('Sample SD of length','sd_length'),('Median episode length','median_length')]:
    comp.append(f"| {label} | {rd[key]:.2f} | {cr[key]:.2f} |")
for label,key in [('Landed (sleep)','landed'),('Crashed (hull contact)','crash'),('Horizontal exit','out_of_bounds'),('Time-limit only','time_limit')]:
    comp.append(f"| {label} | {rd['counts'][key]} / {rd['n']} ({rd['counts'][key]/rd['n']:.0%}) | {cr['counts'][key]} / {cr['n']} ({cr['counts'][key]/cr['n']:.0%}) |")
comp.append(f"| Return ≥ 200 | {rd['fraction_return_ge_200']:.0%} | {cr['fraction_return_ge_200']:.0%} |")
table='\n'.join(comp)
clips='\n'.join(f"- Seed {r['seed']}, {r['stage']} — {r['outcome']}, {r['length']} steps, return {r['reward']:+.2f}." for r in demo['episodes'])
text=f"""# Continuous LunarLander with CrossQ

**Team 19 · Final experimental report**

The validation-selected CrossQ actor achieved **{cr['mean_reward']:.2f} mean return** and **{cr['counts']['landed']}/{cr['n']} simulator-classified landings** on held-out initial conditions. Uniform random actions averaged {rd['mean_reward']:.2f} and landed {rd['counts']['landed']}/{rd['n']} times.

## Training

Three fresh runs used the same initialization seed and 200,000 online interactions each, for **{experiments['total_training_steps']:,} training steps in total**. The sparse-update pilot completed {pilot['updates']:,} critic updates in {pilot['seconds_learning']:.1f} learning seconds. None of its 200 validation episodes landed. Two additional validation probes at step 160,000 showed hovering through all 1,000 steps without leg contact. We tested two remedies independently against this pilot. One restored the default update rate. The other charged one point per training step to discourage indefinite hovering. Every validation and test episode uses unmodified Gymnasium rewards. We selected the highest original-reward validation mean across all 60 checkpoint evaluations.

{design_table}

The selected profile was **{winner['profile']}**. For the step-cost run the replay buffer stores $r_{{train}}=r_{{env}}-1$ on every transition, including the terminal transition. Shorter flights become more attractive under the training objective. This shaping can also favor an early crash. Outcome counts therefore matter. The Monitor inside the training wrapper logs original returns. The physical simulator, actor, terminal rules and evaluation remain identical across profiles. The sum of the three in-learner runtimes was {sum(x['training_seconds'] for x in experiments['runs'])/60:.2f} CPU-run minutes. Some jobs overlapped in wall time.

We trained official SB3-Contrib CrossQ 2.7.1 from random initialization in Gymnasium 1.2.2 using `LunarLander-v3` with `continuous=True`. The actor receives eight state components and outputs two bounded controls. The registered continuous alias matched the explicit constructor for a seeded reset and 25 consecutive actions.

| Setting | Measured run or chosen value |
|---|---|
| Budget | {tr['steps']:,} environment steps, {tr['episodes']:,} completed training episodes |
| Hardware | Apple M1 Pro, 8 CPU cores, 16 GB RAM, CPU training with one PyTorch thread |
| Runtime | {tr['seconds_learning']:.1f} s ({tr['seconds_learning']/60:.2f} min) inside learning calls, {tr['steps']/tr['seconds_learning']:.1f} environment steps/s overall |
| Networks | Two 128-unit ReLU hidden layers in actor and each of two critics |
| Replay | Capacity 200,000, minibatch 128, 10,000 uniform warm-up steps |
| Optimization | Adam, actor/critic learning rate 0.001, betas (0.5, 0.999), γ = 0.99 |
| Updates | One critic optimizer step per environment step after warm-up, {tr['updates']:,} critic updates total. Actor and entropy temperature every third critic update |
| Entropy | Automatic temperature, initial α = 1, target entropy −2 |
| Normalization | SB3-Contrib BatchRenorm, momentum 0.01, ε = 0.001, renorm warm-up 100,000 training-mode calls per layer |
| Seeds | Training 19, validation 4100–4109, test 8200–8249, demonstration 12000–12002 |

The Zoo specifies a 200,000-step budget, 10,000-step warm-up and a 1,000,000-entry buffer. A capacity of 200,000 retains every interaction in this run. We reduced the actor/critic widths and batch size for the small state vector and CPU budget. The selected run used a post-warm-up update-to-data ratio of {1/winner['train_freq']:.2f}. The correction phase of the BatchRenorm critic begins after 100,000 training-mode calls. The selected run completed {tr['updates']:,} critic updates. During its initial phase the layers use batch normalization in training mode and running statistics for inference. Timing covers interaction and optimization inside `learn`; checkpoint serialization and validation are excluded. [Zoo configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/crossq.yml) · [Pinned CrossQ implementation](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/v2.7.1/sb3_contrib/crossq)

## Selection and learning

The ten-seed deterministic validation mean at the end of random warm-up was **{float(progress[0]['validation_mean']):.2f}**. The best checkpoint scored **{tr['selected_validation_mean']:.2f}** at **{tr['selected_step']:,} steps**. The final checkpoint scored {float(progress[-1]['validation_mean']):.2f}. We fixed the highest-validation checkpoint before running the held-out comparison. Repeated selection on ten validation seeds can favor an unusually high validation mean. Over the first and last 30 completed exploratory training episodes, mean return was {statistics.mean(float(r['reward']) for r in episodes[:30]):.2f} and {statistics.mean(float(r['reward']) for r in episodes[-30:]):.2f}, respectively.

![Episode returns, rolling mean and validation means](results/training_curve.png)

## Held-out comparison

Each policy ran once on each of the same 50 seeds. Episode return is the undiscounted sum of rewards. The deterministic actor and uniform-action policy were evaluated without rendering.

{table}

The trained actor gained **{ev['mean_paired_gain']:+.2f} points per seed** on average, improving on {ev['positive_pairs']} of {cr['n']} paired episodes. A 95% percentile bootstrap interval over seed pairs was [{lo:.2f}, {hi:.2f}]. The plotted points and the JSON summary can be recomputed from `results/evaluation_results.csv`.

![Return on each matched test seed](results/comparison.png)

**Outcome interpretation.** The simulator replaces the final reward with +100 when the body falls asleep. An awake lander receives −100 for hull collision or horizontal exit. We distinguish those causes using the environment state. The TimeLimit wrapper can also truncate an episode. A landed body can be outside the marked pad. An individual score above 200 does not identify a terminal cause. [Gymnasium source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.2/gymnasium/envs/box2d/lunar_lander.py)

## Recorded demonstration

The {demo['duration_seconds']:.2f}-second widescreen defense video (`videos/demo_final.mp4`) shows six complete `rgb_array` episodes. Two use uniform random actions, one uses the archived 100,000-step standard-run checkpoint, and three use the selected weights. The demonstration uses consecutive seeds disjoint from validation and testing. The end card summarizes the separate held-out test.

{clips}

## Scope and verification

There is one initialization seed shared by the three configurations and one default-gravity, wind-disabled environment. Uncertainty over test seeds does not estimate training-seed variance. A comparison against random actions cannot establish an advantage over SAC or isolate a benefit from normalization. Multiple training seeds and a matched-budget SAC ablation would address those questions.

The selected model's SHA-256 fingerprint is `{ev['model_sha256']}`. The training, evaluation and video manifests store the same digest. The submission includes the episode CSVs, a re-executable notebook, an offline presentation with notes and a standalone MP4. Run `python tests/verify_submission.py` to audit the bundle.
"""
(root/"REPORT.md").write_text(text)
