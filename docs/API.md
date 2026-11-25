# Flight Lab API

## Overview

Flight Lab is a local web application for running and visualizing LunarLander episodes using trained CrossQ checkpoints.

The backend is implemented in `demo_ui/server.py` using Python's built-in HTTP server. The frontend communicates with it through HTTP requests containing JSON.

The server listens on `127.0.0.1` and uses port `8765` by default.

Start the application from the repository root:

```bash
python -m demo_ui.server
```

Then open:

```text
http://127.0.0.1:8765/
```

The server can use a different port:

```bash
python -m demo_ui.server --port 9000
```

## Checkpoints

The application supports two checkpoints:

| Model ID | Checkpoint | Training steps |
|---|---|---:|
| `original` | `models/best_model.zip` | 180,000 |
| `intermediate` | `models/checkpoint_100000.zip` | 100,000 |

The default model is `original`.

Models are loaded on the CPU. Once loaded, a checkpoint is cached by the backend and reused for subsequent flights.

## HTTP endpoints

### 1. `GET /`

Returns the main Flight Lab interface (`demo_ui/index.html`).

Other static resources:

| Endpoint | Resource |
|---|---|
| `GET /style.css` | Application styles |
| `GET /app.js` | Frontend JavaScript |

Unknown GET routes return HTTP `404`.

### 2. `POST /api/launch`

Creates a new flight session and initializes the LunarLander environment.

**Request headers**

```http
Content-Type: application/json
```

**Request body**

```json
{
  "model": "original",
  "mode": "manual",
  "seed": 9200,
  "force": 750,
  "direction": -20,
  "rotation": 0
}
```

| Parameter | Type | Allowed values | Description |
|---|---|---|---|
| `model` | string | `original`, `intermediate` | Checkpoint to use |
| `mode` | string | `manual`, `random` | Launch mode |
| `seed` | integer | 0–4,294,967,295 | Environment seed |
| `force` | integer | 0–1500 | Initial launch force magnitude |
| `direction` | integer | -180–180 | Initial force direction in degrees |
| `rotation` | integer | -90–90 | Initial lander rotation in degrees |

All six fields are validated by the backend. The launch parameters `force`, `direction`, and `rotation` are applied in manual mode. In random mode, the environment uses its standard seeded initialization.

**Successful response**

The response contains:

- `id`: unique flight session identifier.
- `model`: checkpoint information.
- `seed`: environment seed.
- `scene`: simulation dimensions, terrain, landing pad, and gravity.
- `frame`: initial flight state.

Example structure:

```json
{
  "id": "flight-session-id",
  "model": {
    "file": "models/best_model.zip",
    "sha256": "checkpoint-sha256",
    "steps": 180000,
    "results": null
  },
  "seed": 9200,
  "scene": {
    "width": 0,
    "height": 0,
    "fps": 50,
    "terrain": [],
    "pad": [],
    "gravity": 0
  },
  "frame": {}
}
```

The values above illustrate the response structure; actual simulation values are generated at runtime. The `results` field contains recorded evaluation metrics when they match the loaded checkpoint, or `null` otherwise.

### 3. `POST /api/step`

Advances an existing flight session by a specified number of simulation steps.

**Request body**

```json
{
  "id": "flight-session-id",
  "count": 4
}
```

| Parameter | Type | Allowed values | Description |
|---|---|---|---|
| `id` | string | Existing flight ID | Identifies the flight session |
| `count` | integer | 1–12 | Maximum number of simulation steps to advance |

For every simulation step, the backend:

1. Reads the current observation.
2. Gets an action from the selected CrossQ policy.
3. Uses deterministic policy prediction.
4. Advances the environment.
5. Updates the cumulative reward and flight status.
6. Records the new frame.

The endpoint returns an array of frame objects. If the episode ends before the requested number of steps is completed, fewer frames are returned.

**Frame structure**

Each frame contains:

| Field | Description |
|---|---|
| `step` | Current simulation step |
| `time` | Elapsed simulation time in seconds |
| `status` | Current flight status |
| `reward` | Reward from the latest step |
| `total` | Cumulative episode reward |
| `position` | Lander position `[x, y]` |
| `velocity` | Linear velocity `[vx, vy]` |
| `angle` | Lander angle |
| `angular_velocity` | Angular velocity |
| `contacts` | Ground-contact state of both legs |
| `shapes` | World coordinates of the lander and leg shapes |
| `observation` | Observation used to predict the action |
| `action` | Predicted