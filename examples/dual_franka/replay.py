"""Configuration-driven, hardware-free direct or ZeroMQ policy replay."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml


KEYS = ["left_ee", "right_ee", "left_gripper", "right_gripper"]
SLICES = [(0, 6), (6, 12), (12, 13), (13, 14)]


def load_config(path):
    """Direct replay can share the active server's checkpoint selection."""
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if config["mode"] == "direct" and config.get("server_config"):
        if "checkpoint" in config:
            raise ValueError("Choose server_config or checkpoint, not both")
        server = yaml.safe_load((path.parent / config["server_config"]).read_text())
        if server.get("mode") != "server":
            raise ValueError("server_config must refer to a policy server configuration")
        config["checkpoint"] = server["model_path"]
        config.setdefault("device", server.get("device", "cuda"))
    return config


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    args = a.parse_args()
    c = load_config(args.config)
    f = np.load(c["fixture"], allow_pickle=False)
    state = f["state"].astype(np.float32)
    assert state.shape == (14,) and np.isfinite(state).all()
    obs = {
        "video": {k: f[k][None, None] for k in ["head", "left_wrist", "right_wrist"]},
        "state": {k: state[s:e][None, None] for k, (s, e) in zip(KEYS, SLICES)},
        "language": {"annotation.human.task_description": [[str(f["text"].item())]]},
    }
    if c["mode"] == "direct":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            "new_embodiment", c["checkpoint"], device=c.get("device", "cuda"), strict=True
        )
    elif c["mode"] == "remote":
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(
            c.get("host", "127.0.0.1"), c.get("port", 5555), timeout_ms=c.get("timeout_ms", 120000)
        )
        if not policy.ping():
            raise RuntimeError("Policy server unavailable")
        config = policy.get_modality_config()
        assert config["action"].delta_indices == list(range(40))
        assert config["action"].modality_keys == KEYS
    else:
        raise ValueError(c["mode"])
    latencies = []
    outputs = []
    for _ in range(c.get("repeats", 5)):
        start = time.monotonic()
        action, _ = policy.get_action(obs, options={"seed": c.get("seed", 42)})
        values = np.concatenate([action[k] for k in KEYS], axis=-1)
        assert values.shape == (1, 40, 14) and np.isfinite(values).all()
        latencies.append(time.monotonic() - start)
        outputs.append(values)
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "actions.npy", np.stack(outputs))
    result = {
        "mode": c["mode"],
        "latency_seconds": latencies,
        "warm_p50": float(np.median(latencies[1:])) if len(latencies) > 1 else latencies[0],
        "warm_p95": float(np.percentile(latencies[1:], 95)) if len(latencies) > 1 else latencies[0],
        "shape": list(outputs[0].shape),
        "max_repeat_difference": float(np.max(np.abs(np.stack(outputs) - outputs[0]))),
    }
    if c["mode"] == "direct" and torch.cuda.is_available():
        result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if hasattr(policy, "close"):
        policy.close()


if __name__ == "__main__":
    main()
