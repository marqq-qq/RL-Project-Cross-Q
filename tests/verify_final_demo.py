"""Replay the six filmed flights and compare decoded MP4 frames with the simulator."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, digest, episode, make_env
from record_demo_final import inputs, result_card, title_card


def same_image(actual, expected, context):
    # The delivery uses lossy H.264 with chroma subsampling.
    delta = np.abs(actual.astype(np.int16) - np.asarray(expected, dtype=np.int16))
    assert delta.mean() < 2.8 and np.mean(delta > 28) < 0.022, (context, delta.mean())


def verify_final_demo(path=None):
    torch.set_num_threads(1)
    path = path or ROOT/"videos/demo_final.mp4"
    manifest = json.loads(path.with_suffix(".json").read_text())
    reference, test, test_rows, models, identities = inputs()
    assert manifest["models"] == identities and manifest["video_sha256"] == digest(path)
    assert manifest["playback_speed"] == 1 and manifest["environment_fps"] == 50
    assert manifest["fps"] == 25 and manifest["resolution"] == [1920,1080]
    ref = manifest["reference_manifest"]
    assert digest(ROOT/ref["path"]) == ref["sha256"]
    evaluation = manifest["evaluation"]
    assert digest(ROOT/evaluation["path"]) == evaluation["sha256"]
    assert digest(ROOT/evaluation["episodes_path"]) == evaluation["episodes_sha256"]
    for key, stats in evaluation["policies"].items():
        assert stats == {k:test["policies"][key][k] for k in ("n","mean_reward","landing_rate","counts")}
    flights = manifest["episodes"]
    assert len(flights) == len(reference["episodes"]) == 6
    assert manifest["demonstration_seeds"] == sorted({r["seed"] for r in flights})
    assert set(manifest["demonstration_seeds"]).isdisjoint(test["seeds"])

    scenes = manifest["scenes"]
    assert [s["name"] for s in scenes] == ["introduction","random","intermediate","selected","held_out_results"]
    cursor = 0
    for s in scenes:
        assert s["start_frame"] == cursor and s["start_seconds"] == cursor / manifest["fps"]
        cursor = s["end_frame_exclusive"]
        assert cursor > s["start_frame"] and s["end_seconds"] == cursor / manifest["fps"]
    assert cursor == manifest["frames"] and 50 <= cursor / manifest["fps"] == manifest["duration_seconds"] <= 58
    for s in scenes[1:-1]:
        group = [r for r in flights if r["stage"] == s["name"]]
        assert (s["start_frame"],s["end_frame_exclusive"]) == (group[0]["start_frame"],group[-1]["end_frame_exclusive"])

    # Collect one airborne and one terminal frame per episode plus each editorial card.
    image_indices = {scenes[0]["start_frame"]+12, scenes[-1]["start_frame"]+12}
    for r in flights:
        image_indices.update((r["start_frame"] + 32, r["terminal_frame"]+3))
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=["-vf","scale=960:540"])
    meta = next(frames)
    assert meta["source_size"] == (1920,1080) and meta["fps"] == 25
    assert meta["codec"] == "h264" and meta["pix_fmt"].startswith("yuv420p")
    images = {}
    count = 0
    for i,pixels in enumerate(frames):
        if i in image_indices:
            images[i] = np.frombuffer(pixels,np.uint8).reshape(540,960,3).copy()
        count += 1
    assert count == manifest["frames"] and images.keys() == image_indices
    for idx,card in [(12,title_card()), (scenes[-1]["start_frame"]+12,result_card(test,test_rows))]:
        same_image(images[idx],card.resize((960,540),Image.Resampling.LANCZOS),"card")

    x,y,w,h = (n//2 for n in manifest["viewport_xywh"])
    assert w/h == 600/400
    cursor = scenes[0]["end_frame_exclusive"]
    for r, old in zip(flights,reference["episodes"]):
        stage,seed = r["stage"],r["seed"]
        assert r["start_frame"] == cursor
        for key in ("stage","seed","length","outcome","policy","checkpoint_steps","terminated","truncated","final_reward"):
            assert r[key] == old[key], (stage,seed,key)
        actor = models.get(stage)
        assert r["checkpoint_steps"] == (actor.num_timesteps if actor else None)
        np.testing.assert_allclose(r["reward"],old["reward"],atol=1e-5,rtol=0)
        assert r["displayed_return"] == f'{r["reward"]:+.2f}'.replace("-","−")
        assert r["terminal_frame"] == cursor + (r["length"]+1)//2
        cursor = r["terminal_frame"] + manifest["outcome_hold_frames"][stage]
        assert r["end_frame_exclusive"] == cursor

        def check_picture(env, total, step, status):
            if step % 2 and status == "flying":
                return
            rgb = env.render()
            if step == 64 or status != "flying":
                i = r["start_frame"] + 32 if status == "flying" else r["terminal_frame"]+3
                expected = Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR)
                same_image(images[i][y:y+h,x:x+w],expected,(stage,seed,step))

        env = make_env("rgb_array")
        try:
            live = episode(env,seed,actor,check_picture)
        finally:
            env.close()
        for key in ("length","outcome","terminated","truncated"):
            assert live[key] == r[key], (stage,seed,key)
        np.testing.assert_allclose(live["reward"],r["reward"],atol=1e-5,rtol=0)
    assert cursor == scenes[-1]["start_frame"]
    print(f"Final film passed: {count} decoded frames, {manifest['duration_seconds']:.2f} s, six real episodes, "
          "twelve matched simulator views, title and held-out results cards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT/"videos/demo_final.mp4")
    verify_final_demo(parser.parse_args().video)
