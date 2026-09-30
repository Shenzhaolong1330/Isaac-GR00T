"""Reload native full/LoRA/freeze checkpoints and replay without robot hardware."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path

from examples.dual_franka.replay import KEYS, SLICES
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
from safetensors import safe_open
import torch
import yaml


def weight_map(path):
    indices = list(path.glob("*.safetensors.index.json"))
    if indices:
        return json.loads(indices[0].read_text())["weight_map"]
    file = next(path.glob("*.safetensors"))
    with safe_open(file, framework="pt", device="cpu") as stream:
        return {key: file.name for key in stream.keys()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    fixture = np.load(config["fixture"], allow_pickle=False)
    state = fixture["state"].astype(np.float32)
    observation = {
        "video": {key: fixture[key][None, None] for key in ["head", "left_wrist", "right_wrist"]},
        "state": {key: state[start:end][None, None] for key, (start, end) in zip(KEYS, SLICES)},
        "language": {"annotation.human.task_description": [[str(fixture["text"].item())]]},
    }
    results = []
    base = Path(config["base_vlm"])
    base_map = weight_map(base)
    for checkpoint in config["checkpoints"]:
        path = Path(checkpoint)
        policy = Gr00tPolicy("new_embodiment", str(path), device="cuda", strict=True)
        mode = policy.model.config.vlm_mode
        saved_map = weight_map(path)
        checked = frozen_checked = 0
        with ExitStack() as stack:
            saved = {
                name: stack.enter_context(safe_open(path / name, framework="pt", device="cpu"))
                for name in set(saved_map.values())
            }
            original = {
                name: stack.enter_context(safe_open(base / name, framework="pt", device="cpu"))
                for name in set(base_map.values())
            }
            for name, parameter in policy.model.named_parameters():
                assert name in saved_map, f"Parameter missing in checkpoint: {name}"
                disk = saved[saved_map[name]].get_tensor(name).reshape(-1)
                actual = parameter.detach().reshape(-1)
                expected = torch.cat([disk[:64], disk[-64:]]).to(actual.dtype)
                loaded = torch.cat([actual[:64], actual[-64:]]).cpu()
                assert torch.equal(expected, loaded), f"Checkpoint load mismatch: {name}"
                checked += 1
                if (
                    mode in ("lora", "freeze")
                    and name.startswith("backbone.model.")
                    and ".lora_" not in name
                ):
                    key = name.removeprefix("backbone.model.").replace(".base_layer.", ".")
                    assert key in base_map, key
                    raw = original[base_map[key]].get_tensor(key).reshape(-1)
                    unchanged = torch.cat([raw[:64], raw[-64:]]).to(actual.dtype)
                    assert torch.equal(unchanged, loaded), f"Frozen VLM changed: {name}"
                    frozen_checked += 1
        arrays = []
        for _ in range(2):
            action, _ = policy.get_action(observation, options={"seed": 42})
            values = np.concatenate([action[key] for key in KEYS], axis=-1)
            assert values.shape == (1, 40, 14) and np.isfinite(values).all()
            arrays.append(values)
        np.testing.assert_allclose(arrays[0], arrays[1], atol=1e-6, rtol=0)
        result = {
            "mode": mode,
            "checkpoint": str(path),
            "parameter_samples_loaded": checked,
            "frozen_vlm_tensor_samples_unchanged": frozen_checked,
            "action_shape": list(arrays[0].shape),
            "max_repeat_difference": float(np.abs(arrays[0] - arrays[1]).max()),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del policy, parameter, actual, arrays
        gc.collect()
        torch.cuda.empty_cache()
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
