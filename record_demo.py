"""Record full demonstration flights and a held-out results card at native simulation speed."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio_ffmpeg
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import CrossQ

from experiment import DEMO_SEED, ROOT, digest, episode, make_env, save_json

WIDTH, HEIGHT, FPS = 960, 640, 25
INTRO_FRAMES, OUTCOME_FRAMES, SUMMARY_FRAMES = 55, 20, 115
BG, FG, ACCENT = "#101d2d", "#edf4fa", "#8ce5d7"
STATUS = dict(flying="In flight", landed="Landed", crash="Hull contact",
              time_limit="Time limit", out_of_bounds="Horizontal exit")


def font(size, bold=False):
    prop = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(prop), size)


def record(model_file, out, stills=None, intermediate=None, evaluation=None):
    torch.set_num_threads(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    if stills:
        stills.mkdir(parents=True, exist_ok=True)
    intermediate = intermediate or ROOT / "models/checkpoint_100000.zip"
    evaluation = evaluation or ROOT / "results/evaluation_results.json"
    test = json.loads(evaluation.read_text())
    final_sha, early_sha = digest(model_file), digest(intermediate)
    if final_sha != test["model_sha256"]:
        raise ValueError("Supply --evaluation for the same model as --model.")
    early = CrossQ.load(intermediate, device="cpu")
    selected = CrossQ.load(model_file, device="cpu")
    if early.num_timesteps >= selected.num_timesteps:
        raise ValueError("The intermediate checkpoint must precede the selected checkpoint.")
    flights = [
        ("random", DEMO_SEED, None), ("random", DEMO_SEED + 1, None),
        ("intermediate", DEMO_SEED, early),
        ("selected", DEMO_SEED, selected), ("selected", DEMO_SEED + 1, selected),
        ("selected", DEMO_SEED + 2, selected),
    ]
    small, medium, large = font(23), font(28, True), font(47, True)
    temp = out.with_name(out.stem + ".partial.mp4")
    stream = imageio_ffmpeg.write_frames(str(temp), (WIDTH, HEIGHT), fps=FPS, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=1,
                                         output_params=["-crf", "20", "-movflags", "+faststart"])
    stream.send(None)
    frame_count = 0

    def write(im, copies=1):
        nonlocal frame_count
        pixels = np.asarray(im).tobytes()
        for _ in range(copies):
            stream.send(pixels)
        frame_count += copies

    title = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(title)
    draw.text((54, 150), "TEAM 19   /   CONTINUOUS LUNAR LANDING", font=small, fill=ACCENT)
    draw.text((50, 221), "Learning when to cut thrust", font=large, fill=FG)
    draw.text((54, 335), "Random actions, an intermediate policy and the selected policy", font=small, fill=FG)
    draw.text((54, 395), "Six example flights   ·   Complete episodes   ·   Real time", font=small, fill=ACCENT)
    rows = []
    try:
        write(title, INTRO_FRAMES)
        for stage, seed, policy in flights:
            start = frame_count
            step_count = None if policy is None else policy.num_timesteps
            if policy is None:
                label = "Uniform random actions"
            else:
                label = f"{stage.capitalize()} CrossQ  ·  {step_count:,} steps"
            env = make_env("rgb_array")

            def frame(env, reward, step, status):
                if step % 2 and status == "flying":
                    return
                raw = Image.fromarray(env.render())
                if stills and stage == "selected" and step == 64 and seed in (DEMO_SEED, DEMO_SEED+1):
                    raw.save(stills / f"crossq_{seed}_approach.png")
                if stills and stage != "intermediate" and seed == DEMO_SEED and status != "flying":
                    raw.save(stills / f"{'random' if policy is None else 'crossq'}_{seed}_end.png")
                im = Image.new("RGB", (WIDTH, HEIGHT), BG)
                im.paste(raw.resize((792,528), Image.Resampling.NEAREST), (84, 54))
                panel = ImageDraw.Draw(im)
                panel.text((26, 10), label, font=medium, fill=FG)
                panel.text((750, 14), f"Seed {seed}", font=small, fill=ACCENT)
                panel.text((26, 591), f"Step {step:04d}", font=small, fill=FG)
                panel.text((285, 591), f"Return {reward:+.1f}", font=small, fill=FG)
                panel.text((657, 589), STATUS[status], font=medium, fill=ACCENT)
                write(im, 1 if status == "flying" else OUTCOME_FRAMES)

            try:
                row = episode(env, seed, policy, frame)
            finally:
                env.close()
            rows.append(dict(**row, stage=stage, checkpoint_steps=step_count,
                             start_frame=start, end_frame_exclusive=frame_count))

        base, learned = (test["policies"][k] for k in ("random", "crossq"))
        end = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(end)
        draw.text((54, 68), "HELD-OUT EVALUATION", font=small, fill=ACCENT)
        draw.text((51, 128), f"{learned['n']} matched test seeds", font=large, fill=FG)
        for x, heading in [(54,"Policy"),(440,"Mean return"),(700,"Landings")]:
            draw.text((x, 253), heading, font=small, fill=ACCENT)
        for y, label, stats in [(314,"Uniform random",base),(386,"Selected CrossQ",learned)]:
            draw.text((54,y),label,font=medium,fill=FG)
            draw.text((440,y),f"{stats['mean_reward']:+.2f}",font=medium,fill=FG)
            draw.text((700,y),f"{stats['counts']['landed']} / {stats['n']}",font=medium,fill=FG)
        draw.text((54, 520), "The six filmed examples use separate demonstration seeds.", font=small, fill=ACCENT)
        write(end, SUMMARY_FRAMES)
    finally:
        stream.close()
    temp.replace(out)
    manifest = dict(model_sha256=final_sha, intermediate_sha256=early_sha,
                    evaluation_sha256=digest(evaluation), video_sha256=digest(out),
                    fps=FPS, environment_fps=50, frames=frame_count, duration_seconds=frame_count/FPS,
                    intro_frames=INTRO_FRAMES, outcome_frames=OUTCOME_FRAMES, summary_frames=SUMMARY_FRAMES,
                    seed_rule="Random on the first two consecutive demo seeds, intermediate on the first, selected on the first three.",
                    episodes=rows, held_out={key:{k:v[k] for k in ("n","mean_reward","counts")} for key,v in test["policies"].items()})
    save_json(out.with_suffix(".json"), manifest)
    print(f"{len(rows)} flights, {frame_count} frames, {frame_count/FPS:.2f} seconds")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/best_model.zip")
    p.add_argument("--intermediate", type=Path, default=ROOT / "models/checkpoint_100000.zip")
    p.add_argument("--evaluation", type=Path, default=ROOT / "results/evaluation_results.json")
    p.add_argument("--out", type=Path, default=ROOT / "videos/demo.mp4")
    p.add_argument("--stills", type=Path)
    a = p.parse_args()
    record(a.model, a.out, a.stills, a.intermediate, a.evaluation)
