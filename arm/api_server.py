"""FastAPI server for high-level SO-101 arm control."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .api import EndEffectorDeltaResult, SO101ArmAPI
from .commands import EndEffectorDeltaCommand
from .config import ArmSettings, load_settings
from .controller import SO101ArmController
from .ik import ArmPose, SO101IKTranslator

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_URDF_PATH = Path("models/so101.urdf")
KEYBOARD_UI_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 FastAPI Control</title>
  <style>
    :root {
      --bg: #f3efe7;
      --panel: rgba(255, 251, 244, 0.9);
      --ink: #1d2528;
      --muted: #617177;
      --line: #d4cab8;
      --accent: #005f73;
      --accent-2: #bb3e03;
      --ok: #1b7f4d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      background:
        radial-gradient(circle at top left, rgba(0, 95, 115, 0.12), transparent 28%),
        radial-gradient(circle at bottom right, rgba(187, 62, 3, 0.14), transparent 34%),
        linear-gradient(135deg, #f8f4eb, #ece7dc 55%, #f4efe6);
      padding: 24px;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(29, 37, 40, 0.08);
      padding: 20px;
      backdrop-filter: blur(8px);
    }
    h1, h2 { margin: 0 0 10px; }
    p { margin: 0; color: var(--muted); line-height: 1.55; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.72);
    }
    .card strong {
      display: block;
      margin-bottom: 8px;
      color: var(--accent);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    code, pre {
      font-family: inherit;
    }
    #status {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(0, 95, 115, 0.08);
      color: var(--accent);
      border: 1px solid rgba(0, 95, 115, 0.16);
    }
    #status.error {
      background: rgba(187, 62, 3, 0.08);
      color: var(--accent-2);
      border-color: rgba(187, 62, 3, 0.16);
    }
    #status.ok {
      background: rgba(27, 127, 77, 0.08);
      color: var(--ok);
      border-color: rgba(27, 127, 77, 0.16);
    }
    .pillbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
      min-height: 34px;
    }
    .pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 6px 10px;
      background: white;
      color: var(--ink);
      font-size: 13px;
    }
    .debug {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .debug-box {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.72);
    }
    .debug-box strong {
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
      font-size: 13px;
    }
    .debug-box span {
      font-size: 22px;
      line-height: 1.1;
    }
    .target-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
    }
    .target-pair {
      display: grid;
      gap: 2px;
    }
    .target-pair label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .target-pair code {
      font-size: 14px;
    }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: #192226;
      color: #e8f0ef;
      overflow: auto;
      min-height: 180px;
    }
    .callout {
      color: var(--accent-2);
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>SO-101 Keyboard Command UI</h1>
      <p>Click this page, keep it focused, and use the keyboard to send mixed reach, shoulder-pan, and end-effector delta commands to <code>/api/end-effector-delta</code>.</p>
      <div class="grid">
        <div class="card"><strong>Move X</strong><code>w</code> positive, <code>s</code> negative forward reach</div>
        <div class="card"><strong>Move Y</strong><code>a</code> negative, <code>d</code> positive shoulder pan</div>
        <div class="card"><strong>Move Z</strong><code>i</code> positive, <code>k</code> negative</div>
        <div class="card"><strong>Tool Roll</strong><code>j</code> negative, <code>l</code> positive</div>
        <div class="card"><strong>Gripper</strong><code>t</code> open, <code>g</code> close</div>
        <div class="card"><strong>Send</strong>Commands stream while keys are held</div>
      </div>
      <div class="row">
        <label>Reach/Z step (meters)
          <input id="linearStep" type="number" step="0.001" value="__LINEAR_STEP__">
        </label>
        <label>Shoulder pan step (radians)
          <input id="panStep" type="number" step="0.01" value="__PAN_STEP__">
        </label>
        <label>Roll step (degrees)
          <input id="rollStep" type="number" step="0.1" value="__ROLL_STEP__">
        </label>
        <label>Jaw step
          <input id="jawStep" type="number" step="0.1" value="__JAW_STEP__">
        </label>
        <label>Repeat interval (ms)
          <input id="repeatMs" type="number" step="10" min="20" value="__REPEAT_MS__">
        </label>
      </div>
      <div id="status">Idle. Press and hold control keys to send commands.</div>
      <div class="pillbar" id="keys"></div>
      <p class="callout">This UI binds only to the same FastAPI server process. Use <code>--dry-run</code> first before sending commands to real hardware.</p>
    </section>
    <section class="panel">
      <h2>Command + Targets</h2>
      <div class="debug">
        <div class="debug-box">
          <strong>Browser Command</strong>
          <pre id="commandPayload">{}</pre>
        </div>
        <div class="debug-box">
          <strong>LeRobot Send / Merged State</strong>
          <div class="target-grid" id="targets">
            <div class="target-pair"><label>Shoulder Pan</label><code id="sent-shoulder_pan">send: --</code><code id="merged-shoulder_pan">merged: --</code></div>
            <div class="target-pair"><label>Shoulder Lift</label><code id="sent-shoulder_lift">send: --</code><code id="merged-shoulder_lift">merged: --</code></div>
            <div class="target-pair"><label>Elbow Flex</label><code id="sent-elbow_flex">send: --</code><code id="merged-elbow_flex">merged: --</code></div>
            <div class="target-pair"><label>Wrist Flex</label><code id="sent-wrist_flex">send: --</code><code id="merged-wrist_flex">merged: --</code></div>
            <div class="target-pair"><label>Wrist Roll</label><code id="sent-wrist_roll">send: --</code><code id="merged-wrist_roll">merged: --</code></div>
            <div class="target-pair"><label>Gripper</label><code id="sent-gripper">send: --</code><code id="merged-gripper">merged: --</code></div>
          </div>
        </div>
      </div>
      <h2>Last Response</h2>
      <pre id="response">{}</pre>
    </section>
  </main>
  <script>
    const activeKeys = new Set();
    const keyPills = document.getElementById("keys");
    const status = document.getElementById("status");
    const commandPayload = document.getElementById("commandPayload");
    const response = document.getElementById("response");
    const targetFields = {
      shoulder_pan: {
        sent: document.getElementById("sent-shoulder_pan"),
        merged: document.getElementById("merged-shoulder_pan"),
      },
      shoulder_lift: {
        sent: document.getElementById("sent-shoulder_lift"),
        merged: document.getElementById("merged-shoulder_lift"),
      },
      elbow_flex: {
        sent: document.getElementById("sent-elbow_flex"),
        merged: document.getElementById("merged-elbow_flex"),
      },
      wrist_flex: {
        sent: document.getElementById("sent-wrist_flex"),
        merged: document.getElementById("merged-wrist_flex"),
      },
      wrist_roll: {
        sent: document.getElementById("sent-wrist_roll"),
        merged: document.getElementById("merged-wrist_roll"),
      },
      gripper: {
        sent: document.getElementById("sent-gripper"),
        merged: document.getElementById("merged-gripper"),
      },
    };
    let timerId = null;

    const mappings = {
      w: ["dx", 1],
      s: ["dx", -1],
      a: ["dy", -1],
      d: ["dy", 1],
      i: ["dz", 1],
      k: ["dz", -1],
      j: ["d_rot", -1],
      l: ["d_rot", 1],
      t: ["d_jaw", 1],
      g: ["d_jaw", -1],
    };

    function readFloat(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isFinite(value) ? value : 0;
    }

    function renderKeys() {
      const keys = Array.from(activeKeys).sort();
      keyPills.innerHTML = "";
      if (keys.length === 0) {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = "No active keys";
        keyPills.appendChild(pill);
        return;
      }
      for (const key of keys) {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = key;
        keyPills.appendChild(pill);
      }
    }

    function formatTargetValue(value) {
      if (!Number.isFinite(value)) return "--";
      return value.toFixed(3);
    }

    function renderTargets(sentTargets, mergedTargets) {
      for (const [joint, fields] of Object.entries(targetFields)) {
        fields.sent.textContent = `send: ${formatTargetValue(sentTargets?.[joint])}`;
        fields.merged.textContent = `merged: ${formatTargetValue(mergedTargets?.[joint])}`;
      }
    }

    function buildCommand() {
      const linearStep = readFloat("linearStep");
      const panStep = readFloat("panStep");
      const rollStep = readFloat("rollStep");
      const jawStep = readFloat("jawStep");
      const command = {dx: 0, dy: 0, dz: 0, d_rot: 0, d_jaw: 0};

      for (const key of activeKeys) {
        const mapping = mappings[key];
        if (!mapping) continue;
        const [axis, direction] = mapping;
        const step =
          axis === "dy" ? panStep : axis === "d_rot" ? rollStep : axis === "d_jaw" ? jawStep : linearStep;
        command[axis] += direction * step;
      }
      return command;
    }

    async function sendCommand() {
      if (activeKeys.size === 0) return;
      const command = buildCommand();
      try {
        const res = await fetch("/api/end-effector-delta", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(command),
        });
        const payload = await res.json();
        commandPayload.textContent = JSON.stringify(payload.command ?? command, null, 2);
        renderTargets(payload.sent_targets, payload.merged_targets);
        response.textContent = JSON.stringify(payload, null, 2);
        status.className = "ok";
        status.textContent = `Sent ${JSON.stringify(command)}`;
      } catch (error) {
        status.className = "error";
        status.textContent = `Request failed: ${error}`;
      }
    }

    function startLoop() {
      if (timerId !== null) return;
      sendCommand();
      const interval = Math.max(20, readFloat("repeatMs"));
      timerId = window.setInterval(sendCommand, interval);
    }

    function stopLoop() {
      if (timerId !== null) {
        window.clearInterval(timerId);
        timerId = null;
      }
    }

    function normalizeKey(event) {
      if (event.key.length === 1) return event.key.toLowerCase();
      return null;
    }

    window.addEventListener("keydown", (event) => {
      const key = normalizeKey(event);
      if (!(key in mappings)) return;
      event.preventDefault();
      activeKeys.add(key);
      renderKeys();
      startLoop();
    });

    window.addEventListener("keyup", (event) => {
      const key = normalizeKey(event);
      if (!(key in mappings)) return;
      event.preventDefault();
      activeKeys.delete(key);
      renderKeys();
      if (activeKeys.size === 0) {
        stopLoop();
        status.className = "";
        status.textContent = "Idle. Press and hold control keys to send commands.";
      }
    });

    window.addEventListener("blur", () => {
      activeKeys.clear();
      renderKeys();
      stopLoop();
      status.className = "";
      status.textContent = "Focus lost. Active keys cleared.";
    });

    renderKeys();
  </script>
</body>
</html>
"""


def render_keyboard_ui_html(settings: ArmSettings) -> str:
    step_sizes = settings.api_ui_step_sizes
    return (
        KEYBOARD_UI_HTML_TEMPLATE.replace("__LINEAR_STEP__", str(step_sizes["linear"]))
        .replace("__PAN_STEP__", str(step_sizes["pan"]))
        .replace("__ROLL_STEP__", str(step_sizes["roll"]))
        .replace("__JAW_STEP__", str(step_sizes["jaw"]))
        .replace("__REPEAT_MS__", str(max(20, settings.api_ui_repeat_ms)))
    )


class EndEffectorDeltaRequest(BaseModel):
    dx: float = Field(default=0.0)
    dy: float = Field(default=0.0)
    dz: float = Field(default=0.0)
    d_rot: float = Field(default=0.0)
    d_jaw: float = Field(default=0.0)


class JointPositionsResponse(BaseModel):
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float


class ArmPoseResponse(BaseModel):
    x: float
    y: float
    z: float
    tool_roll: float


class EndEffectorDeltaResponse(BaseModel):
    command: EndEffectorDeltaRequest
    sent_targets: dict[str, float]
    merged_targets: JointPositionsResponse


@dataclass
class ArmServerRuntime:
    arm_api: SO101ArmAPI


def build_arm_api(settings: ArmSettings, urdf_path: str | None = None) -> SO101ArmAPI:
    resolved_urdf_path = urdf_path
    if resolved_urdf_path is None and DEFAULT_URDF_PATH.exists():
        resolved_urdf_path = str(DEFAULT_URDF_PATH)

    controller = SO101ArmController(settings=settings)
    ik_translator = SO101IKTranslator(
        urdf_path=resolved_urdf_path,
        use_degrees=settings.use_degrees,
        joint_names=settings.ik_joint_names,
        end_effector_link=settings.ik_end_effector_link,
    )
    return SO101ArmAPI(controller=controller, ik_translator=ik_translator)


def create_app(arm_api: SO101ArmAPI, settings: ArmSettings | None = None) -> FastAPI:
    settings = settings or ArmSettings()
    runtime = ArmServerRuntime(arm_api=arm_api)
    keyboard_ui_html = render_keyboard_ui_html(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.arm_api.connect()
        try:
            yield
        finally:
            runtime.arm_api.disconnect()

    app = FastAPI(title="SO-101 Arm API", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return keyboard_ui_html

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/joints", response_model=JointPositionsResponse)
    def get_joint_positions() -> JointPositionsResponse:
        return JointPositionsResponse(**runtime.arm_api.get_joint_positions())

    @app.get("/api/pose", response_model=ArmPoseResponse)
    def get_end_effector_pose() -> ArmPoseResponse:
        pose = runtime.arm_api.get_end_effector_pose()
        return _pose_response(pose)

    @app.post("/api/end-effector-delta", response_model=EndEffectorDeltaResponse)
    def send_end_effector_delta(command: EndEffectorDeltaRequest) -> EndEffectorDeltaResponse:
        logger.info("API /api/end-effector-delta command=%s", command.model_dump())
        result = runtime.arm_api.send_end_effector_delta(
            EndEffectorDeltaCommand(
                dx=command.dx,
                dy=command.dy,
                dz=command.dz,
                d_rot=command.d_rot,
                d_jaw=command.d_jaw,
            )
        )
        logger.info(
            "API /api/end-effector-delta sent_targets=%s merged_targets=%s",
            result.sent_targets,
            result.merged_targets,
        )
        return EndEffectorDeltaResponse(
            command=command,
            sent_targets=result.sent_targets,
            merged_targets=JointPositionsResponse(**result.merged_targets),
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FastAPI server for high-level SO-101 arm control.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument(
        "--urdf-path",
        help="Optional local path to so101.urdf. Defaults to models/so101.urdf if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware by using the controller dry-run mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.dry_run:
        settings = settings.with_overrides({"dry_run": True})

    app = create_app(build_arm_api(settings=settings, urdf_path=args.urdf_path), settings=settings)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _pose_response(pose: ArmPose) -> ArmPoseResponse:
    return ArmPoseResponse(
        x=pose.x,
        y=pose.y,
        z=pose.z,
        tool_roll=pose.tool_roll,
    )


if __name__ == "__main__":
    raise SystemExit(main())
