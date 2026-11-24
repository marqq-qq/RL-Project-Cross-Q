# CrossQ for Continuous Lunar Landing

Team 19 · Speaker script · 4 minutes 30 seconds of introductory slides. Play the separate demo after slide 5, then open the live notebook.

## 1 · Danil Khasanshin · The question (0:40)

Look at the lander above the two flags. It can earn points by moving closer, but it still needs to slow its descent, keep both feet underneath it and eventually cut thrust. A policy that learns only the first part can hover until time runs out. Our question was whether CrossQ could learn the whole sequence using continuous engine commands. We will introduce the experiment here. After the flights, each of us will walk through one part of the same notebook.

## 2 · Danil Khasanshin · Inputs and engines (0:50)

The picture lets us watch the approach. The actor receives eight numbers. Horizontal velocity tells it how fast it is drifting, angle tells it which way thrust will point, and the last two numbers tell it when each leg touches the terrain. The two controls have dead zones. For example, a main command of 0.6 requests 80 percent power. A lateral command of minus 0.75 fires the left booster at 75 percent. Approaching upright and slowly improves the shaping score, while engine use costs reward. Martin will explain how the agent learns which commands are worth applying.

- https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.2/gymnasium/envs/box2d/lunar_lander.py

## 3 · Martin Klimov · The joint critic pass (0:55)

SAC gives us a stochastic actor and two critics that estimate the soft return of each action. Consider one replay sample. The action actually taken belongs on the current side of the value update. On the successor side, the actor proposes a fresh action. Those two inputs can have different feature distributions. CrossQ puts both sets of rows through each critic together, so normalization measures both on one scale. We split the outputs and hold the target fixed for the gradient step. Our live example will show the difference between separate and joint statistics. Malik will then use PyTorch to take one critic step we can check by hand.

- https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html
- https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/crossq/crossq.py

## 4 · Arseniy Laptev · Training and selection (1:05)

Each run began afresh with a ten-thousand-step random warm-up. The sparse pilot made one critic update every four interactions and often hovered. The standard run increased that to one update per interaction. A third run kept the sparse update rate and charged one extra point per training step. All three used the original rewards in validation. Follow the line for the standard run. Its validation return stayed near zero for a long stretch before rising sharply. We evaluated every ten thousand steps on the same ten validation seeds. The highest mean across sixty checkpoints was 280.91 at 180,000 steps of the standard run. Only then did we compare the frozen actor with uniform random actions on fifty new seeds.

- https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/crossq.yml

## 5 · Nikita Selezenev · Test flights and handoff (1:00)

Every row uses the same fifty initial conditions for both policies. The selected actor averaged 274.59 points, compared with -218.12 for uniform random actions. Look at the outcomes as well as the mean. CrossQ landed on 48 seeds and timed out twice. The random policy crashed thirty times and left the horizontal boundary twenty times. We use the simulator's sleeping-body condition to identify a landing. The video now shows separate example seeds. Look for the difference between the intermediate actor hovering and the selected actor settling. After the end card, Danil will open the notebook and show exactly what the pilot observes.
