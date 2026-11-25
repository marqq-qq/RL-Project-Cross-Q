"use strict";

const byId = (id) => document.getElementById(id);
const scene = byId("scene");
const dial = byId("dial");
const play = byId("play");
const reset = byId("reset");
const setup = byId("setup");
const observationCells = [...byId("observation").querySelectorAll("td")];
const number = new Intl.NumberFormat("en-US");
const statuses = {
  ready: "Ready", running: "In flight", paused: "Paused",
  landed: "Landed", crash: "Crashed", out_of_bounds: "Out of bounds",
  time_limit: "Time limit", loading: "Preparing", error: "Error",
};
const state = {flight: null, frame: null, trace: [], running: false, busy: false, generation: 0};
let previewTimeout;

function fixed(value, digits = 2) {
  return value.toFixed(digits).replace("-", "−");
}

function degrees(radians) {
  return radians * 180 / Math.PI;
}

function launchOptions() {
  return {
    mode: setup.querySelector("input[name=mode]:checked").value,
    seed: byId("seed").valueAsNumber,
    force: byId("force").valueAsNumber,
    direction: byId("direction").valueAsNumber,
    rotation: byId("rotation").valueAsNumber,
  };
}

async function request(path, payload) {
  const result = await fetch(path, payload === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await result.json();
  if (!result.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function status(value) {
  byId("status").dataset.state = value;
  byId("status").querySelector("span").textContent = statuses[value];
}

function setButtons() {
  const loaded = Boolean(state.flight);
  play.disabled = !loaded || state.busy;
  reset.disabled = !byId("seed").checkValidity();
  const resume = state.frame?.step > 0 && state.frame?.status === "flying";
  const icon = state.running ? "Ⅱ" : "▶";
  const label = state.running ? "Pause" : (resume ? "Resume" : "Launch");
  play.innerHTML = `<span aria-hidden="true">${icon}</span> ${label}`;
}

function showError(error) {
  byId("error").textContent = error.message;
  byId("error").hidden = false;
  status("error");
}

async function prepare() {
  if (!byId("seed").checkValidity()) {
    showError(new Error("Seed must be an integer from 0 to 4,294,967,295."));
    return;
  }
  const ownGeneration = ++state.generation;
  state.running = false;
  state.busy = true;
  byId("error").hidden = true;
  status("loading");
  setButtons();
  try {
    const flight = await request("/api/launch", launchOptions());
    if (ownGeneration !== state.generation) return;
    state.flight = flight;
    state.frame = flight.frame;
    state.trace = [flight.frame.position];
    showModel(flight.model);
    showFrame();
    status("ready");
  } catch (error) {
    if (ownGeneration === state.generation) {
      state.flight = null;
      showError(error);
    }
  } finally {
    if (ownGeneration === state.generation) {
      state.busy = false;
      setButtons();
    }
  }
}

async function startOrPause() {
  if (state.running) {
    state.running = false;
    status("paused");
    setButtons();
    return;
  }
  if (state.frame?.status !== "flying") await prepare();
  if (!state.flight || state.busy) return;
  state.running = true;
  status("running");
  setButtons();
  if (!state.ticking) void run(state.generation);
}

async function run(generation) {
  state.ticking = true;
  try {
    while (state.running && state.frame.status === "flying" && generation === state.generation) {
      const start = performance.now();
      const speed = Number(byId("play-speed").value);
      const count = Math.max(1, speed * 2);
      const frames = await request("/api/step", {id: state.flight.id, count});
      if (generation !== state.generation) break;
      for (const frame of frames) {
        state.frame = frame;
        state.trace.push(frame.position);
      }
      if (frames.length) showFrame();
      const interval = count * 1000 / (speed * state.flight.scene.fps);
      await new Promise((resolve) => setTimeout(resolve, Math.max(0, interval - (performance.now() - start))));
    }
    if (generation === state.generation && state.frame.status !== "flying") {
      state.running = false;
      status(state.frame.status);
      setButtons();
    }
  } catch (error) {
    if (generation === state.generation) {
      state.running = false;
      showError(error);
      setButtons();
    }
  } finally {
    state.ticking = false;
    if (state.running) void run(state.generation);
  }
}

function showFrame() {
  const f = state.frame;
  byId("time").textContent = `${fixed(f.time)} s`;
  byId("step").textContent = `${f.step} / 1000`;
  byId("position").textContent = f.position.map((x) => fixed(x)).join(" / ");
  byId("altitude").textContent = fixed(f.position[1] - state.flight.scene.pad[2]);
  byId("velocity").textContent = f.velocity.map((x) => fixed(x)).join(" / ");
  byId("attitude").textContent = `${fixed(degrees(f.angle), 1)}° / ${fixed(degrees(f.angular_velocity), 1)}°/s`;
  byId("left-leg").classList.toggle("active", f.contacts[1]);
  byId("right-leg").classList.toggle("active", f.contacts[0]);
  byId("reward").textContent = `${fixed(f.reward)} / ${fixed(f.total)}`;
  for (let i = 0; i < observationCells.length; i++) observationCells[i].textContent = fixed(f.observation[i], 3);
  byId("main-action").textContent = fixed(f.action[0], 3);
  byId("side-action").textContent = fixed(f.action[1], 3);
  byId("main-engine").value = f.action[0] > 0 ? (1 + f.action[0]) / 2 : 0;
  byId("side-engine").value = Math.abs(f.action[1]) > 0.5 ? Math.abs(f.action[1]) : 0;
  drawScene();
}

function showModel(model) {
  byId("filename").textContent = model.file;
  byId("model-steps").textContent = number.format(model.steps);
  byId("sha").textContent = model.sha256;

  const results = model.results;
  byId("benchmark").hidden = !results;
  byId("benchmark-empty").hidden = Boolean(results);
  if (!results) return;
  const episodes = number.format(results.episodes);
  const landingRate = results.landings / results.episodes;
  byId("benchmark-context").textContent = `${episodes} held-out episodes · lateral starts within ±45°`;
  byId("landing-rate").textContent = number.format(landingRate * 100) + "%";
  byId("landing-count").textContent = `${number.format(results.landings)} of ${episodes} landed`;
  byId("timeout-count").textContent = number.format(results.timeouts);
  byId("timeout-total").textContent = `of ${episodes} episodes`;
  byId("median-time").textContent = `${fixed(results.median_seconds)} s`;
  byId("median-steps").textContent = `${number.format(results.median_steps)} steps`;
}

function canvasContext(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const w = Math.round(width * ratio);
  const h = Math.round(height * ratio);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, width, height};
}

function path(ctx, points, close = false) {
  ctx.beginPath();
  points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  if (close) ctx.closePath();
}

function drawDial() {
  const {ctx, width, height} = canvasContext(dial);
  const x = width / 2;
  const y = height / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111319";
  ctx.strokeStyle = "#3b3d47";
  ctx.beginPath();
  ctx.arc(x, y, x - 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  path(ctx, [[x, 8], [x, height - 8]]);
  ctx.stroke();
  path(ctx, [[8, y], [width - 8, y]]);
  ctx.stroke();
  const angle = -byId("direction").valueAsNumber * Math.PI / 180;
  const tip = [x + Math.cos(angle) * (x - 13), y + Math.sin(angle) * (x - 13)];
  ctx.strokeStyle = "#f5b96d";
  ctx.fillStyle = "#f5b96d";
  ctx.lineWidth = 2;
  path(ctx, [[x, y], tip]);
  ctx.stroke();
  path(ctx, [tip, [tip[0] - 8 * Math.cos(angle - .5), tip[1] - 8 * Math.sin(angle - .5)],
    [tip[0] - 8 * Math.cos(angle + .5), tip[1] - 8 * Math.sin(angle + .5)]], true);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  ctx.fill();
}

function drawScene() {
  const {ctx, width, height} = canvasContext(scene);
  const sky = ctx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "#10131c");
  sky.addColorStop(1, "#202633");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);
  for (let i = 1; i <= 65; i++) {
    ctx.fillStyle = i % 7 === 0 ? "#a3aab8" : "#464e5c";
    ctx.fillRect(width * ((i * 31.37) % 100) / 100, height * ((i * 47.93) % 91) / 100, i % 7 === 0 ? 1.6 : 1, 1);
  }

  if (!state.flight) return;
  const world = state.flight.scene;
  const f = state.frame;
  const maxY = Math.max(world.height + 2.2, f.position[1] + 1.5);
  const scale = Math.min((width - 26) / world.width, (height - 14) / maxY);
  const left = (width - world.width * scale) / 2;
  const bottom = height - 10;
  const transform = ([x, y]) => [left + x * scale, bottom - y * scale];
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#2b303d";
  for (let x = 0; x <= world.width; x += 2) {
    path(ctx, [transform([x, 0]), transform([x, maxY])]);
    ctx.stroke();
  }
  for (let y = 0; y <= maxY; y += 2) {
    path(ctx, [transform([0, y]), transform([world.width, y])]);
    ctx.stroke();
  }

  const ground = world.terrain.map(transform);
  const dirt = ctx.createLinearGradient(0, height * .63, 0, height);
  dirt.addColorStop(0, "#38404e");
  dirt.addColorStop(1, "#181c25");
  ctx.fillStyle = dirt;
  path(ctx, [[0, ground[0][1]], ...ground, [width, ground.at(-1)[1]], [width, height], [0, height]], true);
  ctx.fill();
  ctx.strokeStyle = "#656c7a";
  path(ctx, ground);
  ctx.stroke();

  const [padLeft, padRight, padY] = world.pad;
  const pad = [transform([padLeft, padY]), transform([padRight, padY])];
  ctx.strokeStyle = "#88d5b0";
  ctx.lineWidth = 2;
  path(ctx, pad);
  ctx.stroke();
  for (const [x, y] of pad) {
    path(ctx, [[x, y], [x, y - 10]]);
    ctx.stroke();
    ctx.fillStyle = "#88d5b0";
    ctx.fillRect(x - 2, y - 10, 4, 3);
  }

  if (state.trace.length > 1) {
    ctx.strokeStyle = "#f5b96d";
    ctx.lineWidth = 1.35;
    ctx.setLineDash([3, 4]);
    ctx.globalAlpha = .8;
    path(ctx, state.trace.map(transform));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  const [sx, sy] = transform(f.position);
  if (f.step === 0 && world.width) {
    const vx = f.velocity[0];
    const vy = f.velocity[1];
    const speed = Math.hypot(vx, vy);
    if (speed > .05) {
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(Math.atan2(-vy, vx));
      const length = 15 + Math.min(50, speed * 4);
      ctx.strokeStyle = "#f5b96d";
      ctx.lineWidth = 1.5;
      path(ctx, [[10, 0], [length, 0], [length - 6, -4]]);
      ctx.stroke();
      path(ctx, [[length, 0], [length - 6, 4]]);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (f.status === "flying" && f.step && f.action[0] > 0) {
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(-f.angle);
    const flame = scale * (.9 + f.action[0] * .55);
    ctx.fillStyle = "#ffaf63";
    path(ctx, [[-scale * .17, scale * .52], [0, flame], [scale * .17, scale * .52]], true);
    ctx.fill();
    ctx.restore();
  }

  f.shapes.forEach((shape, i) => {
    ctx.fillStyle = i === 0 ? "#d7e2ed" : f.contacts[i - 1] ? "#88d5b0" : "#a5aebe";
    ctx.strokeStyle = i === 0 ? "#5b728a" : "#536174";
    ctx.lineWidth = i === 0 ? 1.4 : 1;
    path(ctx, shape.map(transform), true);
    ctx.fill();
    ctx.stroke();
  });
  ctx.fillStyle = "#26394a";
  ctx.beginPath();
  ctx.arc(sx, sy, Math.max(2, scale * .12), 0, Math.PI * 2);
  ctx.fill();
}

function updateControls() {
  for (const key of ["direction", "force", "rotation"]) {
    byId(`${key}-value`).textContent = `${fixed(byId(key).valueAsNumber, 0)}${key === "force" ? "" : "°"}`;
  }
  byId("manual").disabled = launchOptions().mode !== "manual";
  drawDial();
}

function schedulePreview() {
  updateControls();
  clearTimeout(previewTimeout);
  state.generation++;
  state.running = false;
  state.busy = true;
  setButtons();
  status("loading");
  if (!byId("seed").checkValidity()) {
    showError(new Error("Seed must be an integer from 0 to 4,294,967,295."));
    return;
  }
  previewTimeout = setTimeout(prepare, 130);
}

play.addEventListener("click", startOrPause);
reset.addEventListener("click", () => {
  clearTimeout(previewTimeout);
  void prepare();
});
setup.addEventListener("submit", (event) => {
  event.preventDefault();
  clearTimeout(previewTimeout);
  void prepare();
});
setup.addEventListener("input", schedulePreview);
byId("random-seed").addEventListener("click", () => {
  byId("seed").value = crypto.getRandomValues(new Uint32Array(1))[0];
  schedulePreview();
});
byId("compass").addEventListener("click", (event) => {
  const rect = dial.getBoundingClientRect();
  const x = event.clientX - rect.left - rect.width / 2;
  const y = rect.top + rect.height / 2 - event.clientY;
  if (Math.hypot(x, y) > 5) {
    byId("direction").value = Math.round(degrees(Math.atan2(y, x)));
    schedulePreview();
  }
});
new ResizeObserver(drawScene).observe(scene);
new ResizeObserver(drawDial).observe(dial);

updateControls();
void prepare();
