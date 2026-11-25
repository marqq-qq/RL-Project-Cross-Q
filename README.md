# CrossQ for continuous lunar landing

**Team 19 · Reinforcement Learning final project**

- **Flight Lab** runs locally and lets you control the initial launch conditions.
- [Live notebook](CrossQ_LunarLander_Final.ipynb) walks through the simulator, a CrossQ update, checkpoint selection, and evaluation.
- [Demo video](videos/demo_final.mp4) shows six flights with random actions and two checkpoints from the standard training run.

## Setup

Use Python 3.12 and run the commands from the repository root. Box2D needs a C++ compiler on systems without a compatible wheel. Use Visual C++ Build Tools on Windows, Xcode Command Line Tools on macOS, or build-essential on Debian/Ubuntu.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

With the environment active, install SWIG before building Box2D.

```bash
python -m pip install --upgrade pip
python -m pip install swig==4.5.1
python -m pip install -r requirements.txt
```

## Run Flight Lab

```bash
python -m demo_ui.server
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). Manual mode lets you set the launch force, direction, rotation, and seed. Random mode uses Gymnasium's seeded launch. Both modes display observations, actions, and flight telemetry from the 200,000-step fine-tuned policy.

## Run the notebook

```bash
python -m ipykernel install --sys-prefix --name crossq-team19 --display-name "CrossQ Team 19"
python -m jupyter lab CrossQ_LunarLander_Final.ipynb
```

Select the **CrossQ Team 19** kernel and choose **Restart Kernel and Run All**. The notebook replays six flights and displays archived results for the full evaluation. Small numerical differences between platforms can change individual trajectories.

## Results

We chose the 180,000-step standard checkpoint by its mean return on ten validation seeds before testing. On 50 matched held-out seeds it landed **48/50** times, with a mean return of **274.59**. Uniform random actions landed **0/50** times, with a mean return of **−218.12**. A landing means the simulator put the lander body to sleep. [Per-flight results](results/evaluation_results.csv) · [Summary](results/evaluation_results.json)

Flight Lab uses a separate checkpoint fine-tuned on lateral starts. On 50 lateral test starts it landed **49/50** times, compared with **50/50** for the original checkpoint. Median episode length fell from 239 to 208.5 steps. [Lateral comparison](results/lateral45/comparison.json)
