"""Serve a local flight console backed by the project's CrossQ checkpoints."""
import argparse
import hashlib
import io
import json
import math
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W, LunarLander
from sb3_contrib import CrossQ

from experiment import ROOT, make_env, outcome


ASSETS = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path("models/lateral45_finetuned.zip")
RESULTS_PATH = ROOT / "results/lateral45/comparison.json"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ManualStartLander(LunarLander):
    """Replace the built-in force immediately before the first Box2D step."""

    def __init__(self, force, direction, rotation):
        super().__init__(continuous=True)
        self.force = force
        self.heading = math.radians(direction)
        self.rotation = math.radians(rotation)

    def reset(self, *, seed=None, options=None):
        self.starting = True
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.starting:
            self.starting = False
            self.world.ClearForces()
            self.lander.ApplyForceToCenter(
                (self.force * math.cos(self.heading), self.force * math.sin(self.heading)), True
            )
            center_x, center_y = self.lander.position
            cosine, sine = math.cos(self.rotation), math.sin(self.rotation)
            for leg in self.legs:
                dx, dy = leg.position.x - center_x, leg.position.y - center_y
                leg.position = (center_x + cosine * dx - sine * dy, center_y + sine * dx + cosine * dy)
                leg.angle += self.rotation
            self.lander.angle += self.rotation
        return super().step(action)


class Checkpoint:
    def __init__(self):
        payload = (ROOT / CHECKPOINT_PATH).read_bytes()
        self.model = CrossQ.load(io.BytesIO(payload), device="cpu")
        sha256 = hashlib.sha256(payload).hexdigest()
        self.info = {
            "file": CHECKPOINT_PATH.as_posix(),
            "sha256": sha256,
            "steps": self.model.num_timesteps,
            "results": model_results(sha256),
        }


class Flight:
    def __init__(self, checkpoint, options):
        self.id = uuid.uuid4().hex
        self.checkpoint = checkpoint
        self.seed = options["seed"]
        if options["mode"] == "manual":
            self.env = gym.wrappers.TimeLimit(
                ManualStartLander(options["force"], options["direction"], options["rotation"]),
                max_episode_steps=1000,
            )
        else:
            self.env = make_env()
        self.engine = self.env.unwrapped
        self.obs, _ = self.env.reset(seed=self.seed)
        self.action = [0.0, 0.0]
        self.policy_input = self.obs
        self.total = self.reward = 0.0
        self.steps = 0
        self.status = "flying"

    def open_scene(self):
        engine = self.engine
        terrain = [list(poly[0]) for poly in engine.sky_polys]
        terrain.append(list(engine.sky_polys[-1][1]))
        return {
            "id": self.id,
            "model": self.checkpoint.info,
            "seed": self.seed,
            "scene": {
                "width": VIEWPORT_W / SCALE,
                "height": VIEWPORT_H / SCALE,
                "fps": FPS,
                "terrain": terrain,
                "pad": [engine.helipad_x1, engine.helipad_x2, engine.helipad_y],
                "gravity": engine.gravity,
            },
            "frame": self.frame(),
        }

    def advance(self, count):
        frames = []
        for _ in range(count):
            if self.status != "flying":
                break
            self.policy_input = self.obs
            self.action = self.checkpoint.model.predict(self.policy_input, deterministic=True)[0].tolist()
            self.obs, self.reward, term, trunc, _ = self.env.step(self.action)
            self.steps += 1
            self.total += self.reward
            self.status = outcome(self.env, self.obs, self.reward, term, trunc)
            frames.append(self.frame())
        return frames

    def frame(self):
        body = self.engine.lander
        legs = self.engine.legs
        return {
            "step": self.steps,
            "time": self.steps / FPS,
            "status": self.status,
            "reward": self.reward,
            "total": self.total,
            "position": list(body.position),
            "velocity": list(body.linearVelocity),
            "angle": body.angle,
            "angular_velocity": body.angularVelocity,
            "contacts": [leg.ground_contact for leg in legs],
            "shapes": [
                [list(part.GetWorldPoint(point)) for point in part.fixtures[0].shape.vertices]
                for part in (body, *legs)
            ],
            "observation": self.policy_input.tolist(),
            "action": self.action,
        }


class Console:
    def __init__(self):
        self.lock = threading.Lock()
        self.checkpoint = Checkpoint()
        self.flights = {}

    def launch(self, options):
        with self.lock:
            flight = Flight(self.checkpoint, options)
            if len(self.flights) >= 8:
                self.flights.pop(next(iter(self.flights))).env.close()
            self.flights[flight.id] = flight
            return flight.open_scene()

    def step(self, flight_id, count):
        with self.lock:
            return self.flights[flight_id].advance(count)


class ConsoleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in STATIC:
            self.send_error(404)
            return
        filename, content_type = STATIC[self.path]
        self.send_payload(200, content_type, (ASSETS / filename).read_bytes())

    def do_POST(self):
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected Content-Type application/json.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/launch":
                response = self.server.console.launch(launch_options(body))
            elif self.path == "/api/step":
                flight_id = body.get("id")
                if not isinstance(flight_id, str):
                    raise ValueError("A flight ID is required.")
                response = self.server.console.step(flight_id, integer(body, "count", 1, 12))
            else:
                self.send_error(404)
                return
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)
            return
        except KeyError:
            self.send_json({"error": "Flight session expired. Reset to start a new flight."}, 404)
            return
        self.send_json(response)

    def send_json(self, payload, status=200):
        self.send_payload(status, "application/json; charset=utf-8",
                          json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    def send_payload(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        if self.path != "/api/step":
            super().log_message(fmt, *args)


def model_results(sha256):
    try:
        comparison = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    results = comparison["after"]
    if results["sha256"] != sha256:
        return None
    return {
        "episodes": results["n"],
        "landings": results["counts"]["landed"],
        "timeouts": results["counts"]["time_limit"],
        "median_steps": results["median_length"],
        "median_seconds": results["median_length"] / FPS,
    }


def integer(values, key, low, high):
    value = values.get(key)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}.")
    return value


def launch_options(values):
    mode = values.get("mode")
    if mode not in ("manual", "random"):
        raise ValueError("Choose Manual or Random launch mode.")
    return {
        "mode": mode,
        "seed": integer(values, "seed", 0, 2**32 - 1),
        "force": integer(values, "force", 0, 1500),
        "direction": integer(values, "direction", -180, 180),
        "rotation": integer(values, "rotation", -90, 90),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with ThreadingHTTPServer(("127.0.0.1", args.port), ConsoleHandler) as server:
        server.console = Console()
        print(f"CrossQ Flight Lab · http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
