"""Export a native GR00T checkpoint with an embedded offline PhysBrain architecture."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    source = Path(cfg["checkpoint"])
    vlm = Path(cfg["vlm"])
    out = Path(cfg["output"])
    if out.exists():
        raise FileExistsError(out)
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=False)
    weights = list(source.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("No safetensors checkpoint")
    for file in weights + list(source.glob("*.index.json")):
        shutil.copy2(file, tmp / file.name)
    model = json.loads((source / "config.json").read_text())
    arch = json.loads((vlm / "config.json").read_text())
    arch["text_config"]["num_hidden_layers"] = int(model["select_layer"])
    model.update(model_name="vlm", backbone_config=arch)
    (tmp / "config.json").write_text(json.dumps(model, indent=2))
    processor = source if (source / "processor_config.json").exists() else source / "processor"
    for name in ["processor_config.json", "statistics.json", "embodiment_id.json"]:
        shutil.copy2(processor / name, tmp / name)
    pc = json.loads((tmp / "processor_config.json").read_text())
    pc["processor_kwargs"]["model_name"] = "vlm"
    (tmp / "processor_config.json").write_text(json.dumps(pc, indent=2))
    (tmp / "vlm").mkdir()
    for f in vlm.iterdir():
        if f.is_file() and f.suffix in [".json", ".jinja", ".txt", ".model"]:
            shutil.copy2(f, tmp / "vlm" / f.name)
    if (source / "experiment_cfg").exists():
        shutil.copytree(source / "experiment_cfg", tmp / "experiment_cfg")
    contract = {
        "format": "dual_franka_ee14_v1",
        "gripper_action_semantics": "heuristic_closed_plateau_zero_v1",
        "state_order": [
            "left_position_xyz",
            "left_rotation_vector",
            "right_position_xyz",
            "right_rotation_vector",
            "left_gripper_width",
            "right_gripper_width",
        ],
        "action_order": [
            "left_delta_xyz",
            "left_delta_rotation_vector",
            "right_delta_xyz",
            "right_delta_rotation_vector",
            "left_gripper_width",
            "right_gripper_width",
        ],
        "units": {"translation": "metres", "rotation": "radians", "gripper": "metres"},
        "external_dimension": 14,
        "action_horizon": 40,
        "camera_order": ["head", "left_wrist", "right_wrist"],
        "client_image": "RGB uint8 HWC 240x424",
        "server_processor": pc["processor_kwargs"],
        "statistics_sha256": hashlib.sha256((tmp / "statistics.json").read_bytes()).hexdigest(),
    }
    (tmp / "dual_franka_contract.json").write_text(json.dumps(contract, indent=2))
    manifest = {
        "checkpoint": str(source),
        "backbone_layers": arch["text_config"]["num_hidden_layers"],
        "inference_only": True,
        "files": {},
    }
    for f in sorted(tmp.rglob("*")):
        if f.is_file():
            h = hashlib.sha256()
            with f.open("rb") as stream:
                for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(b)
            manifest["files"][str(f.relative_to(tmp))] = {
                "size": f.stat().st_size,
                "sha256": h.hexdigest(),
            }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tmp.rename(out)
    print(out)


if __name__ == "__main__":
    main()
