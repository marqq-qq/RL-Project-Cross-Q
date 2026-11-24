"""Render the widescreen defense film from the project's frozen Gymnasium policies."""
import argparse
import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import gymnasium as gym
import imageio_ffmpeg
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import CrossQ

from experiment import DEMO_SEED, ROOT, digest, episode, make_env, save_json, summarize

WIDTH, HEIGHT, FPS = 1920, 1080, 25
VIEW = (44, 82, 1440, 960)  # x, y, width, height; full 600 × 400 simulator viewport
INTRO, END = 75, 175
HOLDS = {"random": 30, "intermediate": 32, "selected": 35}
NAVY, LINE, TRACK = "#101e2d", "#435365", "#2a3b4c"
WHITE, MUTED, TEAL, RED, GREEN = "#f5f8fb", "#b7c9d4", "#80dfdc", "#ff9b92", "#8ce0ad"
STAGES = {"random": ("RANDOM", "BASELINE"), "intermediate": ("INTERMEDIATE", "CROSSQ"),
          "selected": ("TRAINED", "CROSSQ")}
OUTCOMES = {"flying": ("In flight", TEAL), "crash": ("Crash", RED),
            "out_of_bounds": ("Out of bounds", RED), "time_limit": ("Time limit", RED),
            "landed": ("Landed", GREEN)}
MODEL_PATHS = {"intermediate": "models/checkpoint_100000.zip", "selected": "models/best_model.zip"}
TITLE_STILL = "presentation/stills/crossq_12000_approach.png"
TEST_JSON, TEST_CSV = "results/evaluation_results.json", "results/evaluation_results.csv"


@lru_cache(maxsize=None)
def face(size, style="regular"):
    file = {"regular":"DejaVuSans", "bold":"DejaVuSans-Bold", "mono":"DejaVuSansMono"}[style]
    return ImageFont.truetype(str(Path(matplotlib.get_data_path())/"fonts"/"ttf"/(file+".ttf")), size)


def label(draw, x, y, value, size, color=WHITE, style="regular", anchor="lt"):
    draw.text((x,y), str(value), fill=color, font=face(size,style), anchor=anchor)


def line(draw, x1, y1, x2, y2, color=LINE, width=2):
    draw.line((x1,y1,x2,y2), fill=color, width=width)


def canvas():
    return Image.new("RGB", (WIDTH,HEIGHT), NAVY)


def signed(value):
    return f"{value:+.2f}".replace("-", "−")


def title_card():
    im = canvas()
    d = ImageDraw.Draw(im)
    label(d,80,108,"Team 19  |  Reinforcement Learning Final Project",33,TEAL)
    line(d,82,231,169,231,TEAL,6)
    for y, words in [(306,"CrossQ for"),(432,"Continuous"),(558,"Lunar Landing")]:
        label(d,75,y,words,108,WHITE,"bold")
    label(d,82,774,"From Random Actions to Controlled Landing",39)
    with Image.open(ROOT/TITLE_STILL) as still:
        im.paste(still.convert("RGB").resize((774,516),Image.Resampling.LANCZOS),(1062,254))
    d.rectangle((1061,253,1836,770), outline=LINE,width=2)
    label(d,1062,801,"Selected policy on approach",31,TEAL)
    label(d,1062,852,"Seed 12000   /   Step 64",27,MUTED,"mono")
    return im


def result_card(test, rows):
    im = canvas()
    d = ImageDraw.Draw(im)
    label(d,92,76,"HELD-OUT EVALUATION",30,TEAL,"bold")
    label(d,86,155,"How often did the lander settle?",74,WHITE,"bold")
    n = test["policies"]["crossq"]["n"]
    label(d,93,283,f"{n} matched held-out test seeds per policy. The six filmed flights used separate seeds.",34,MUTED)
    xs = (879, 1425)
    for x, name, color in zip(xs, ("Random agent","Trained CrossQ"), (MUTED,TEAL)):
        label(d,x,415,name,38,color,"bold")
    for y in (490,606,725,845):
        line(d,92,y,1830,y)
    for y, name in [(526,"Mean episode return"), (648,"Simulator-classified landings"), (767,"Landing rate")]:
        label(d,94,y,name,36)
    for x, key in zip(xs, ("random","crossq")):
        s = test["policies"][key]
        color = WHITE if key == "random" else TEAL
        for y, value in [(514,signed(s["mean_reward"])), (634,f'{s["counts"]["landed"]}/{s["n"]}'),
                         (754,f'{s["landing_rate"]:.0%}')]:
            label(d,x,y,value,58,color,"bold")
        for i, r in enumerate(sorted((r for r in rows if r["policy"]==key), key=lambda r:int(r["seed"]))):
            left, top = x + (i%25)*15, 877 + (i//25)*18
            d.rectangle((left,top,left+10,top+10), fill=GREEN if r["outcome"]=="landed" else LINE)
    label(d,94,878,"One square per seed",27,MUTED)
    label(d,94,982,"Green marks a sleeping lander. The simulator can register a landing outside the flags.",30,MUTED)
    return im


class ObserveAction(gym.Wrapper):
    """Store the last command for the overlay; forward it to Box2D unmodified."""
    last = None

    def step(self, action):
        self.last = np.array(action, copy=True)
        return self.env.step(action)


def power(action):
    if action is None:  # No policy action has been taken at reset.
        return 0., 0.
    main, side = np.clip(action, -1, 1)
    return ((1+float(main))/2 if main > 0 else 0., abs(float(side)) if abs(side) > .5 else 0.)


def caption(stage, seed, step, status):
    if status != "flying":
        return {"crash":("The hull hits","the terrain."), "out_of_bounds":("The lander exits","the flight area."),
                "time_limit":("Still airborne at","the 1,000-step limit."),
                "landed":("The simulator registers","the lander at rest.")}[status]
    if stage == "random":
        return ("Uniform actions give","no consistent braking.")
    if stage == "intermediate":
        if step < 220:
            return ("Thrust adjustments","steady the approach.")
        if step < 700:
            return ("The policy holds","the lander aloft.")
        return ("It hovers near the","surface as time runs out.")
    return {DEMO_SEED:("Watch the pilot brake","during descent."),
            DEMO_SEED+1:("Watch the angle change","as the lander levels."),
            DEMO_SEED+2:("Thrust eases as the","legs meet the ground.")}[seed]


def flight_template(stage, seed, checkpoint):
    im = canvas()
    d = ImageDraw.Draw(im)
    label(d,45,24,"Gymnasium   /   LunarLander-v3",31,WHITE)
    label(d,1875,28,"Continuous actions    ·    1× playback",26,MUTED,anchor="rt")
    vx,vy,w,h = VIEW
    d.rectangle((vx-2,vy-2,vx+w+1,vy+h+1),outline=LINE,width=2)
    x = 1524
    line(d,x,103,x+70,103,TEAL,5)
    for i,title in enumerate(STAGES[stage]):
        label(d,x,132+i*53,title,42,WHITE,"bold")
    label(d,x,251, "No learned policy" if checkpoint is None else f"{checkpoint:,} training steps",27,TEAL)
    line(d,x,310,1876,310)
    label(d,x,345,"Episode seed",25,MUTED)
    label(d,1747,345,"Step",25,MUTED)
    label(d,x,391,seed,41,WHITE,"mono")
    label(d,x,476,"Cumulative return",26,MUTED)
    label(d,x,622,"Main thrust",25,MUTED)
    label(d,x,705,"Side thrust",25,MUTED)
    line(d,x,797,1876,797)
    return im


def flight_image(template, raw, stage, seed, step, total, status, action):
    vx,vy,w,h = VIEW
    im = template.copy()
    assert raw.shape == (400,600,3)
    im.paste(Image.fromarray(raw).resize((w,h), Image.Resampling.BILINEAR),(vx,vy))
    d = ImageDraw.Draw(im)
    x = 1524
    label(d,1747,391,step,41,WHITE,"mono")
    label(d,x,525,signed(total),57,WHITE,"mono")
    for y,val in zip((622,705), power(action)):
        label(d,1875,y,f"{val:.0%}",25,WHITE,"mono",anchor="rt")
        d.rectangle((x,y+40,1875,y+49),fill=TRACK)
        if val:
            d.rectangle((x,y+40,x+round(351*val),y+49),fill=TEAL)
    name, color = OUTCOMES[status]
    label(d,x,833, "OUTCOME" if status != "flying" else "STATUS",22,MUTED,"bold")
    label(d,x,871,name,39,color,"bold")
    for i,s in enumerate(caption(stage, seed, step, status)):
        label(d,x,953+i*38,s,27)
    return im


def inputs():
    previous = json.loads((ROOT/"videos/demo.json").read_text())
    test = json.loads((ROOT/TEST_JSON).read_text())
    with (ROOT/TEST_CSV).open() as f:
        rows = [dict(r, reward=float(r["reward"]), length=int(r["length"])) for r in csv.DictReader(f)]
    for policy in ("random","crossq"):
        sample = [r for r in rows if r["policy"]==policy]
        assert sorted(int(r["seed"]) for r in sample) == test["seeds"]
        assert summarize(sample) == test["policies"][policy]
    train = json.loads((ROOT/"results/training_summary.json").read_text())
    models, identities = {}, {}
    for stage,path in MODEL_PATHS.items():
        model = CrossQ.load(ROOT/path, device="cpu")
        sha = digest(ROOT/path)
        expected = previous["intermediate_sha256" if stage=="intermediate" else "model_sha256"]
        assert sha == expected and model.num_timesteps == (100_000 if stage=="intermediate" else 180_000)
        models[stage] = model
        identities[stage] = dict(path=path, sha256=sha, training_steps=model.num_timesteps, profile=train["profile"])
    assert identities["selected"]["sha256"] == train["model_sha256"] == test["model_sha256"]
    assert train["selected_step"] == test["model_steps"] == models["selected"].num_timesteps
    assert digest(ROOT/TEST_JSON) == previous["evaluation_sha256"]
    return previous, test, rows, models, identities


def record(out):
    torch.set_num_threads(1)
    previous, test, rows, models, identities = inputs()
    out = Path(out)
    if out.resolve() == (ROOT/"videos/demo.mp4").resolve():
        raise ValueError("Choose a different output path to preserve the previous film.")
    out.parent.mkdir(parents=True, exist_ok=True)
    flights, scenes, count = [], [], 0
    with TemporaryDirectory(dir=out.parent) as tmp:
        movie = Path(tmp)/"film.mp4"
        writer = imageio_ffmpeg.write_frames(str(movie),(WIDTH,HEIGHT),fps=FPS,codec="libx264",
            pix_fmt_out="yuv420p",macro_block_size=1,output_params=["-vf","setsar=1","-crf","18","-preset","veryfast","-movflags","+faststart"])
        writer.send(None)

        def emit(im, repeat=1):
            nonlocal count
            pixels = im.tobytes()
            for _ in range(repeat):
                writer.send(pixels)
            count += repeat

        def card(name, im, length):
            start = count
            # Only the two editorial cards fade in. The first simulator frame is always fully visible.
            for i in range(1,7):
                emit(Image.blend(canvas(),im,i/7))
            emit(im,length-6)
            scenes.append(dict(name=name,start_frame=start,end_frame_exclusive=count))

        try:
            card("introduction",title_card(),INTRO)
            for saved in previous["episodes"]:
                stage, seed = saved["stage"], saved["seed"]
                actor = models.get(stage)
                checkpoint = actor.num_timesteps if actor else None
                assert seed not in test["seeds"]
                if not scenes or scenes[-1]["name"] != stage:
                    scenes.append(dict(name=stage,start_frame=count))
                start = count
                template = flight_template(stage,seed,checkpoint)
                env = ObserveAction(make_env("rgb_array"))
                assert env.metadata["render_fps"] == 2*FPS
                def capture(sim, total, step, status):
                    if step % 2 and status == "flying":
                        return
                    raw = sim.render()
                    if stage=="selected" and seed==DEMO_SEED and step==64:
                        with Image.open(ROOT/TITLE_STILL) as title:
                            np.testing.assert_array_equal(raw,np.asarray(title.convert("RGB")))
                    im = flight_image(template,raw,stage,seed,step,total,status,sim.last)
                    emit(im, 1 if status=="flying" else HOLDS[stage])
                try:
                    flight = episode(env,seed,actor,capture)
                finally:
                    env.close()
                for key in ("length","outcome","terminated","truncated","final_reward"):
                    assert flight[key] == saved[key], (stage, seed, key)
                np.testing.assert_allclose(flight["reward"],saved["reward"],rtol=0,atol=1e-5)
                flights.append(dict(**flight,stage=stage,checkpoint_steps=checkpoint,
                                    start_frame=start, terminal_frame=count-HOLDS[stage],end_frame_exclusive=count,
                                    displayed_return=signed(flight["reward"]), displayed_outcome=OUTCOMES[flight["outcome"]][0]))
                scenes[-1]["end_frame_exclusive"] = count
                print(stage,seed,flight["length"],flight["outcome"],signed(flight["reward"]),flush=True)
            card("held_out_results",result_card(test,rows),END)
        finally:
            writer.close()
        assert 50 <= count/FPS <= 58
        for s in scenes:
            s.update(start_seconds=s["start_frame"]/FPS,end_seconds=s["end_frame_exclusive"]/FPS)
        manifest = dict(models=identities, demonstration_seeds=sorted({r["seed"] for r in flights}),episodes=flights,
            environment="LunarLander-v3",continuous=True,environment_fps=2*FPS,fps=FPS,
            resolution=[WIDTH,HEIGHT],source_viewport=[600,400],viewport_xywh=list(VIEW),
            frames=count,duration_seconds=count/FPS,playback_speed=1.0,codec="h264",pixel_format="yuv420p",audio=False,
            timing="Complete episodes, reset and every even simulator step plus any odd terminal step. Terminal frames are held for reading.",
            outcome_hold_frames=HOLDS,scenes=scenes,
            reference_manifest=dict(path="videos/demo.json",sha256=digest(ROOT/"videos/demo.json")),
            evaluation=dict(path=TEST_JSON,sha256=digest(ROOT/TEST_JSON),episodes_path=TEST_CSV,episodes_sha256=digest(ROOT/TEST_CSV),
                seed_range=[min(test["seeds"]),max(test["seeds"])],policies={k:{field:v[field] for field in
                ("n","mean_reward","landing_rate","counts")} for k,v in test["policies"].items()}),
            video_sha256=digest(movie))
        movie.replace(out)
        save_json(out.with_suffix(".json"),manifest)
    print(f"{out}  ·  {count} frames  ·  {count/FPS:.2f} seconds")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"videos/demo_final.mp4")
    record(p.parse_args().out)
