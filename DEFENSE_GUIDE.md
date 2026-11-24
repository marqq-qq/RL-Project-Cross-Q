# Team 19 · Live defense guide

**32-minute rehearsal target.** Allow 4:30 for five slides, 0:56 for the demo, 12:00 for one notebook, and 14:34 for questions. The same prepared material leaves 12:34–17:34 for questions in a 30–35 minute session. The roles below are proposed speaking assignments. They describe the defense, not authorship of individual files.

## Before the audience arrives

Extract the whole folder and activate the environment in `README.md`. Open `Final_Presentation.pptx` or its PDF, cue `videos/demo_final.mp4` at the beginning, and open `CrossQ_LunarLander_Final.ipynb` from the project directory. Confirm the active kernel is **CrossQ · Team 19**, then use **Restart Kernel and Run All** once before presenting. Leave outputs visible and enlarge the interface until the back row can read the tables. The optional training cell defines a function. Running the notebook never calls it. If a kernel or projector connection fails, `CrossQ_LunarLander_Final.html` contains the same executed walkthrough for offline viewing.

## Running order

| Elapsed | Lead | Screen and cue |
|---|---|---|
| 0:00–1:30 | Danil Khasanshin | Slides 1–2. Ask what the actor needs to know before cutting thrust. Point to the velocity and contact features. |
| 1:30–2:25 | Martin Klimov | Slide 3. Trace one replay action and one freshly sampled successor action into the joint critic batch. |
| 2:25–3:30 | Arseniy Laptev | Slide 4. Follow the validation curve to 180k. Explain how a checkpoint won before testing. |
| 3:30–4:30 | Nikita Selezenev | Slide 5. Read the physical outcomes alongside the returns. Launch the MP4. |
| 4:30–5:26 | Nikita Selezenev | Full video, 56.16 seconds. Use the timed lines in `videos/demo_narration.md`. Switch to the top of the notebook. |
| 5:26–7:11 | Danil Khasanshin | Notebook §1, **Read the lander**. Show the real step and how physical termination differs from a time limit. |
| 7:11–9:41 | Martin Klimov | §2, **Inside CrossQ**. Trace the diagram, explain the two losses, run the normalization example. |
| 9:41–11:56 | Malik Nurullin | §3, **One numerical update**. Run the target cell, predict the sign of the gradient, then run the update cell. |
| 11:56–14:41 | Arseniy Laptev | §4, **From replay to a checkpoint**. Show settings, loop excerpt, run table and the measured learning curve. |
| 14:41–17:26 | Nikita Selezenev | §5, **Fly the saved actor and measure the outcome**. Show six live replay rows, then the full test distribution and limitations. |
| 17:26–32:00 | All five | Questions. Route environment questions to Danil, algorithm questions to Martin, arithmetic to Malik, training to Arseniy, evaluation to Nikita. |

The timed slide text is in `presentation_script.md` and the PowerPoint speaker notes. All live code and explanations are in the notebook.

## Notebook talk tracks

**Danil · Read the lander.** “Try the command (0.6, −0.75). The main engine fires at 80% because a positive command starts above half power. The left booster fires at 75%. Compare the two eight-number vectors. An ordinary reward reflects the change in motion, position, tilt and leg contact, minus engine costs. At the end we check whether the body slept, collided, left horizontally, or reached the time limit.” Hand off to Martin with “How does a single transition teach the pilot?”

**Martin · Inside CrossQ.** “The replay action was taken by older behavior. At the successor state we sample a new action from the current actor. Both inputs pass through each online critic together. Watch the four numbers in the normalization cell. Separate scales erase the gap between the halves. Joint normalization preserves it. Split the predictions, use the smaller next Q and stop gradients through the target. The actor favors high predicted value while maintaining entropy.” Hand off to Malik with “What does one of these target errors actually do?”

**Malik · One numerical update.** “Our current scalar estimate is 2. The smaller next Q is 3, and entropy contributes +0.1. With reward 1, discount 0.9 and no termination, the target becomes 3.79. Which direction should Q move? The TD error is +1.79, but the derivative of half-squared error is −1.79. Subtracting that gradient at learning rate 0.1 gives 2.179. The notebook shows the loss falling against the same target.” If rerunning, execute **both** numerical cells in order. The first recreates the parameter and computation graph. Hand off to Arseniy for the real minibatch update schedule.

**Arseniy · From replay to a checkpoint.** “Read the settings from the archived standard run. After 10,000 warm-up steps we draw batches of 128, update the critics every interaction and update the actor and temperature every third critic update. Each run began fresh with seed 19. Across three runs we evaluated 60 checkpoints on ten validation seeds. Trace the orange line and find the winning checkpoint. The 200k checkpoint was available, but 180k scored higher.” Hand off to Nikita with “How did the frozen policy do on new starts?”

**Nikita · Evaluation and demo.** “The model digest links these weights to the evaluation record. The first three paired seeds reproduce six rows. The aggregate comes from all 50 seeds per policy. The selected actor landed 48 times and hit the time limit twice. Random actions crashed 30 times and left horizontally 20 times. Follow one line on the paired plot. Its endpoints share terrain and initial force. These results characterize one learned policy under the default environment.” End by inviting questions.

## Demo cues

- **0:00–0:10.20 · Opening and random flights.** Uniform actions on seeds 12000 and 12001 establish a baseline. A seeded action sampler makes each example repeatable.
- **0:10.20–0:31.48 · Intermediate flight.** Seed 12000, genuine 100k checkpoint from the standard run. Watch the step counter reach 1,000 while the lander stays airborne. Its ten archived validation returns were also independently reproduced.
- **0:31.48–0:56.16 · Selected flights and end card.** The 180k checkpoint flies three consecutive demo seeds. The first two revisit the random baseline's initial conditions. At 0:49.16 the final card reports the separate 50-seed test. These six filmed episodes illustrate behavior.

## Likely TA questions

| Question | Answer |
|---|---|
| Why use a continuous environment? | The actor outputs two real commands in a Box. One controls main thrust and one chooses and scales a side booster. Each has a dead zone. |
| Does a score of 200 prove a landing on the pad? | The score sums shaped and terminal rewards. The landing label comes from Box2D putting the body to sleep. A sleeping lander can be outside the marked pad. |
| Which endings suppress bootstrapping? | Genuine MDP terminations. For a time-limit-only ending, the replay buffer retains a bootstrap term using the final observation. |
| Why two critics and a minimum? | The smaller next prediction reduces sensitivity to an optimistic estimate. Both critics fit the same detached soft target. |
| What distinguishes CrossQ from SAC here? | CrossQ computes both halves with online critics and joint normalization. SAC typically bootstraps with slowly averaged target critics. Both use a stochastic actor and an entropy-regularized objective. |
| Why a fixed target if there are no target networks? | Detaching a numerical target stops gradients through that branch for one update. A separate target network would be another set of weights maintained across updates. |
| What does entropy contribute in the example? | Minus α times log density gives −0.2 × (−0.5) = +0.1. The example uses a chosen temperature; the real run learns it. |
| Why is 1.60205 the loss? | One critic’s contribution is half of 1.79 squared. The full implementation sums both critic contributions and averages over a minibatch. |
| Did the training-only step cost produce the submitted policy? | The standard run won using original rewards. The step-cost run was a separate experiment that subtracted one point for each interaction during training. All validation and testing used original rewards. |
| Why did you choose 180k, and could selection be optimistic? | It had the highest mean original return, 280.91, on the ten validation seeds. Selecting from 60 checkpoints can inflate that estimate. We then evaluated the frozen choice on 50 held-out seeds. |
| Why did the trained actor have longer episodes than random? | Most random flights ended in a crash or exit. A controlled approach can take longer. Two timeouts also increased the trained mean length. |
| What uncertainty does the reported interval capture? | The 95% paired bootstrap resamples matched initial conditions for this saved policy. It does not estimate variation between independently trained seeds. |
| Do these runs show that CrossQ outperforms SAC? | The measured comparison is with uniform random actions. A performance comparison against SAC would require a matched-budget SAC baseline and independent training seeds. |
