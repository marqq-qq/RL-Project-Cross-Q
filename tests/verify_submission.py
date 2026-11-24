"""Reconcile the training, evaluation and defense artifacts with the archived evidence."""
import base64
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio_ffmpeg
import nbformat
import numpy as np
import torch
from lxml import html
from pptx import Presentation
from pypdf import PdfReader
from sb3_contrib import CrossQ

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_final_demo import verify_final_demo
from experiment import DEMO_SEED, ROOT, TEST_SEEDS, VALIDATION_SEEDS, digest, episode, make_env, summarize


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text())


def verify_run(run, results):
    folder = results / run["logs"]
    progress = rows(folder / "training_progress.csv")
    validation = rows(folder / "validation_episodes.csv")
    episodes = rows(folder / "training_episodes.csv")
    summary = read_json(folder / "training_summary.json")
    assert summary["steps"] == run["steps"] == 200_000
    assert summary["episodes"] == len(episodes)
    assert sum(int(r["length"]) for r in episodes) <= summary["steps"]
    assert [int(p["step"]) for p in progress] == list(range(10_000, 200_001, 10_000))
    assert len(validation) == len(progress) * len(VALIDATION_SEEDS)
    for p in progress:
        block = [r for r in validation if r["step"] == p["step"]]
        assert Counter(int(r["seed"]) for r in block) == Counter(VALIDATION_SEEDS)
        np.testing.assert_allclose(float(p["validation_mean"]), np.mean([float(r["reward"]) for r in block]))
    best = max(progress, key=lambda p: float(p["validation_mean"]))
    assert int(best["step"]) == run["best_step"] == summary["selected_step"]
    assert float(best["validation_mean"]) == run["validation_mean"] == summary["selected_validation_mean"]
    cfg = summary["config"]
    expected = (summary["steps"]-cfg["learning_starts"]) // cfg["train_freq"] * cfg["gradient_steps"]
    assert summary["updates"] == run["updates"] == expected
    return progress


def check_flight(saved, live):
    np.testing.assert_allclose(live["reward"], float(saved["reward"]), atol=1e-5, rtol=0)
    assert (live["length"], live["outcome"]) == (int(saved["length"]), saved["outcome"])


def verify_notebook(results):
    path = ROOT / "CrossQ_LunarLander_Final.ipynb"
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    code = [c for c in book.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code)+1))
    outputs = [o for c in code for o in c.outputs]
    assert all(o.output_type != "error" for o in outputs)
    streams = "\n".join(o.text for o in outputs if o.output_type == "stream")
    assert all(s in streams for s in ("target 3.790", "1.60205", "Q 2.000 → 2.179", "variance: 5.0 17.0"))
    tables = [html.fromstring(o.data["text/html"]) for o in outputs if "text/html" in o.get("data", {})]
    replay = next(t for t in tables if "matches log" in t.text_content())
    assert replay.xpath(".//tbody/tr/td[last()]/text()") == ["True"] * 6
    charts = [(results / f).read_bytes() for f in ("training_curve.png", "comparison.png")]
    assert [base64.b64decode(o.data["image/png"]) for o in outputs if "image/png" in o.get("data",{})] == charts
    page = html.parse(path.with_suffix(".html"))
    assert len(page.xpath("//h2")) == 5 and not page.xpath("//script[@src]|//link[@href]")
    embeds = [base64.b64decode(img.split(",",1)[1]) for img in page.xpath("//img/@src")]
    assert embeds == charts
    print(f"Notebook: {len(code)} executed cells, numerical update, six matching flights and two original plots in IPYNB and HTML.")


def verify_slides(test, progress):
    slides = Presentation(ROOT / "Final_Presentation.pptx").slides
    pdf = PdfReader(ROOT / "Final_Presentation.pdf")
    assert len(slides) == len(pdf.pages) == 5
    assert all(s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip() for s in slides)
    chart = next(sh.chart for sh in slides[3].shapes if sh.has_chart)
    np.testing.assert_allclose(chart.series[0].values, [float(p["validation_mean"]) for p in progress], atol=1e-7)
    table = next(sh.table for sh in slides[4].shapes if sh.has_table)
    for col, policy in [(1,"random"), (2,"crossq")]:
        stats = test["policies"][policy]
        values = [f"{stats[k]:.1f}" for k in ("mean_reward","sd_reward","median_reward","mean_length")]
        values += [str(stats["counts"][key]) for key in ("landed","crash","out_of_bounds","time_limit")]
        assert [table.cell(row,col).text for row in range(1,9)] == values
        assert values[0] in pdf.pages[4].extract_text()
    assert "180,000" in pdf.pages[3].extract_text() and "280.91" in pdf.pages[3].extract_text()
    print("PPTX/PDF: five slides, notes, editable validation curve and held-out results.")


def verify_video(test, selected, results):
    path = ROOT / "videos/demo.mp4"
    demo = read_json(path.with_suffix(".json"))
    early_path = ROOT / "models/checkpoint_100000.zip"
    early = CrossQ.load(early_path, device="cpu")
    assert early.num_timesteps == 100_000
    assert demo["video_sha256"] == digest(path) and demo["intermediate_sha256"] == digest(early_path)
    assert demo["model_sha256"] == test["model_sha256"]
    assert demo["evaluation_sha256"] == digest(results / "evaluation_results.json")
    for key, stats in test["policies"].items():
        assert demo["held_out"][key] == {k:stats[k] for k in ("n", "mean_reward", "counts")}
    env = make_env()
    try:
        block = [r for r in rows(results/"validation_episodes.csv") if int(r["step"]) == early.num_timesteps]
        assert tuple(int(r["seed"]) for r in block) == VALIDATION_SEEDS
        for r in block:
            check_flight(r, episode(env, int(r["seed"]), early))
    finally:
        env.close()

    stages = ["random", "random", "intermediate", "selected", "selected", "selected"]
    seeds = [DEMO_SEED + n for n in [0,1,0,0,1,2]]
    assert set(seeds).isdisjoint(set(TEST_SEEDS) | set(VALIDATION_SEEDS))
    flights = demo["episodes"]
    assert [(r["stage"],r["seed"]) for r in flights] == list(zip(stages,seeds))
    actors = {"random":None, "intermediate":early, "selected":selected}
    frame = demo["intro_frames"]
    for r in flights:
        assert r["start_frame"] == frame
        frame += (r["length"]+1)//2 + demo["outcome_frames"]
        assert r["end_frame_exclusive"] == frame
        actor = actors[r["stage"]]
        assert r["checkpoint_steps"] == (actor.num_timesteps if actor else None)
        # The renderer can affect Gymnasium's RNG. Reproduce the writer's render cadence.
        env = make_env("rgb_array")
        def render_frame(sim, reward, step, status):
            if step % 2 == 0 or status != "flying":
                sim.render()
        try:
            live = episode(env, r["seed"], actor, render_frame)
        finally:
            env.close()
        check_flight(r, live)
        for key in ("final_reward", "terminated", "truncated", "policy"):
            assert live[key] == r[key]

    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(frames)
    assert meta["size"] == (960,640) and meta["fps"] == demo["fps"] == 25 and "h264" in meta["codec"]
    assert demo["environment_fps"] == 2 * demo["fps"]
    # Compare two simulator images from within each actual flight, excluding labels.
    samples = {i for r in flights for i in (r["start_frame"]+2, r["start_frame"]+22)}
    pictures = {}
    count = 0
    for i, pixels in enumerate(frames):
        if i in samples:
            a = np.frombuffer(pixels,dtype=np.uint8).reshape(640,960,3)
            pictures[i] = a[54:582:3,84:876:3].copy()
        count += 1
    for r in flights:
        a,b = (pictures[r["start_frame"]+k] for k in (2,22))
        assert a.std() > 10 and np.any(a != b)
    assert count == demo["frames"] == frame + demo["summary_frames"]
    assert 45 <= count / meta["fps"] == demo["duration_seconds"] <= 55
    print(f"Video: all {count} H.264 frames decoded ({demo['duration_seconds']:.2f} s), six flights and ten intermediate validation episodes replayed.")


def verify():
    torch.set_num_threads(1)
    results = ROOT / "results"
    experiments = read_json(results/"experiments.json")
    assert experiments["total_training_steps"] == sum(r["steps"] for r in experiments["runs"]) == 600_000
    histories = {r["profile"]:verify_run(r, results) for r in experiments["runs"]}
    winner = max(experiments["runs"], key=lambda r:r["validation_mean"])
    train = read_json(results/"training_summary.json")
    test = read_json(results/"evaluation_results.json")
    model_path = ROOT / "models/best_model.zip"
    assert winner["profile"] == experiments["selected_profile"] == "standard"
    assert digest(model_path) == winner["model_sha256"] == train["model_sha256"] == test["model_sha256"]
    model = CrossQ.load(model_path, device="cpu")
    assert model.num_timesteps == winner["best_step"] == test["model_steps"] == 180_000
    assert tuple(test["seeds"]) == TEST_SEEDS and set(TEST_SEEDS).isdisjoint(VALIDATION_SEEDS)
    raw = rows(results/"evaluation_results.csv")
    assert len(raw) == 2 * len(TEST_SEEDS)
    env = make_env()
    try:
        for policy in ["random","crossq"]:
            entries = [dict(r,reward=float(r["reward"]),length=int(r["length"])) for r in raw if r["policy"]==policy]
            assert Counter(int(r["seed"]) for r in entries) == Counter(TEST_SEEDS)
            assert summarize(entries) == test["policies"][policy]
            for r in entries:
                check_flight(r, episode(env, int(r["seed"]), model if policy == "crossq" else None))
    finally:
        env.close()
    print(f"Evidence: 60 checkpoints across three runs, selection digest and all {len(raw)} held-out episodes verified.")
    verify_notebook(results)
    verify_slides(test, histories[winner["profile"]])
    verify_video(test, model, results)
    verify_final_demo()


if __name__ == "__main__":
    verify()
