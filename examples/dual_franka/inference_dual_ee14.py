"""Existing dual-Franka robot client connected to the native GR00T ZeroMQ policy."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import logging
from pathlib import Path
import time

from examples.dual_franka.support.robot_control import (
    check_state,
    execute_chunk,
    execution_commands,
    robot_session,
    rpc_state,
)
import numpy as np
import yaml


CAMERAS = ("head", "left_wrist", "right_wrist")
KEYS = ("left_ee", "right_ee", "left_gripper", "right_gripper")
SLICES = ((0, 6), (6, 12), (12, 13), (13, 14))


def observation(state, images, text):
    state = check_state(state)
    video = {}
    for key in CAMERAS:
        rgb = np.asarray(images[key])
        if rgb.dtype != np.uint8 or rgb.shape != (240, 424, 3):
            raise ValueError(
                f"{key}: expected raw RGB uint8[240,424,3], got {rgb.shape}/{rgb.dtype}"
            )
        video[key] = rgb[None, None]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing task text")
    return {
        "video": video,
        "state": {k: state[s:e][None, None] for k, (s, e) in zip(KEYS, SLICES)},
        "language": {"annotation.human.task_description": [[text]]},
    }


def validate_action(action):
    arrays = []
    for key, (start, end) in zip(KEYS, SLICES):
        value = np.asarray(action[key], dtype=np.float32)
        if value.shape != (1, 40, end - start) or not np.isfinite(value).all():
            raise ValueError(f"Invalid physical action response: {key}")
        arrays.append(value)
    return np.concatenate(arrays, axis=-1)[0]


class Client:
    def __init__(self, config):
        from gr00t.policy.server_client import PolicyClient

        self.policy = PolicyClient(
            config.get("host", "127.0.0.1"),
            config.get("port", 5555),
            timeout_ms=config.get("timeout_ms", 30000),
        )
        try:
            if not self.policy.ping():
                raise ConnectionError("GR00T policy server unavailable")
            modalities = self.policy.get_modality_config()
            for kind, keys, indices in [
                ("action", KEYS, list(range(40))),
                ("state", KEYS, [0]),
                ("video", CAMERAS, [0]),
            ]:
                value = modalities[kind]
                if list(value.modality_keys) != list(keys) or value.delta_indices != indices:
                    raise ValueError(f"Wrong dual-Franka policy modality: {kind}")
        except BaseException:
            self.close()
            raise

    def infer(self, obs, seed=None):
        begin = time.perf_counter()
        action, _ = self.policy.get_action(obs, options={} if seed is None else {"seed": seed})
        return validate_action(action), time.perf_counter() - begin

    def close(self):
        self.policy.close()


class Camera:
    """Raw color-only RealSense capture; all model resize happens at the server."""

    def __init__(self, serial, fps=30):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, fps)
        self.pipeline.start(config)

    def read(self):
        frame = self.pipeline.wait_for_frames(timeout_ms=5000).get_color_frame()
        if not frame:
            raise RuntimeError("Missing camera frame")
        return np.asanyarray(frame.get_data()).copy()

    def close(self):
        self.pipeline.stop()


def cleanup(resource):
    try:
        resource.close()
    except Exception:
        logging.exception("Resource cleanup failed")


def validate_config(cfg):
    runtime = cfg.get("runtime", {})
    mode = runtime.get("mode", "replay")
    if mode not in ("replay", "robot"):
        raise ValueError("runtime.mode must be replay or robot")
    if (runtime.get("execute", False) is True) != (mode == "robot"):
        raise ValueError("robot mode requires execute: true; replay requires execute: false")
    n = runtime.get("iterations", 100)
    if type(n) is not int or n < 1:
        raise ValueError("iterations must be a positive integer")
    duration = cfg.get("robot", {}).get("go_home_duration_sec", 5.0)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid home duration")
    execution = cfg.get("execution", {})
    for key in ["action_fps", "max_pos_delta", "max_rot_delta", "gripper_speed", "gripper_force"]:
        value = execution.get(
            key,
            {
                "action_fps": 30,
                "max_pos_delta": 0.3,
                "max_rot_delta": 0.12,
                "gripper_speed": 0.3,
                "gripper_force": 10,
            }[key],
        )
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}")
    h = execution.get("execute_horizon", 10)
    if type(h) is not int or not 1 <= h <= 40:
        raise ValueError("execute_horizon must be 1..40")
    threshold = execution.get("close_threshold_m", 0.03)
    if not np.isfinite(threshold) or not 0 <= threshold <= 0.085:
        raise ValueError("Invalid close threshold")
    return mode, n, duration


def run(cfg):
    mode, iterations, duration = validate_config(cfg)
    # Replay loads only a fixture, never imports or initializes robot/camera SDKs.
    fixture = None
    if mode == "replay":
        with np.load(cfg["fixture"], allow_pickle=False) as data:
            fixture = observation(
                data["state"], {k: data[k] for k in CAMERAS}, str(data["text"].item())
            )
    out = Path(cfg["runtime"]["output"])
    if cfg["runtime"].get("timestamp_output", False):
        out = out / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"[OUTPUT] {out.resolve()}", flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    with ExitStack() as resources:
        client = Client(cfg["server"])
        resources.callback(cleanup, client)
        rpc = None
        cameras = {}
        if mode == "robot":
            from examples.dual_franka.support import dual_franka_robotiq_rpc_client as sdk

            for key in CAMERAS:
                cameras[key] = Camera(cfg["cameras"][key])
                resources.callback(cleanup, cameras[key])
            rpc = sdk.DualFrankaRobotiqRpcClient(ip=cfg["robot"]["ip"], port=cfg["robot"]["port"])
            resources.callback(cleanup, rpc)
        with robot_session(rpc, mode == "robot", duration), (out / "steps.jsonl").open("w") as log:
            for step in range(iterations):
                begin = time.perf_counter()
                if rpc is not None:
                    images = {k: cam.read() for k, cam in cameras.items()}
                    state = rpc_state(rpc.get_observation(), sdk._pose_from_side_observation)
                    obs = observation(state, images, cfg["task"])
                else:
                    obs = fixture
                actions, latency = client.infer(obs, seed=cfg["runtime"].get("seed"))
                # Record the prediction before dispatch, including if hardware rejects a command.
                log.write(
                    json.dumps(
                        {
                            "event": "prediction",
                            "step": step,
                            "model_actions": actions.tolist(),
                            "inference_roundtrip_seconds": latency,
                        }
                    )
                    + "\n"
                )
                log.flush()
                execution = cfg.get("execution", {})
                if rpc is not None:
                    commands = execute_chunk(rpc, actions, execution)
                else:
                    keys = (
                        "execute_horizon",
                        "close_threshold_m",
                        "max_pos_delta",
                        "max_rot_delta",
                    )
                    commands = execution_commands(
                        actions, **{k: execution[k] for k in keys if k in execution}
                    )
                log.write(
                    json.dumps(
                        {
                            "event": "result",
                            "step": step,
                            "executed": rpc is not None,
                            "execution_commands": commands.tolist(),
                            "cycle_seconds": time.perf_counter() - begin,
                        }
                    )
                    + "\n"
                )
                log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("Stopped; robot session exit reset was attempted if robot mode was active.")


if __name__ == "__main__":
    main()
