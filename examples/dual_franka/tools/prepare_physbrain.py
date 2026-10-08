"""Create a Transformers4.57-compatible metadata view; original weights stay unchanged."""

import argparse
import json
from pathlib import Path


def tokenizer_compatibility(source, output):
    source, output = Path(source).resolve(), Path(output)
    src = source / "tokenizer_config.json"
    if not src.exists():
        return
    cfg = json.loads(src.read_text())
    extra = cfg.pop("extra_special_tokens", None)
    if isinstance(extra, list):
        cfg["additional_special_tokens"] = list(
            dict.fromkeys(cfg.get("additional_special_tokens", []) + extra)
        )
        target = output / "tokenizer_config.json"
        if target.is_symlink():
            target.unlink()
        target.write_text(json.dumps(cfg, indent=2))
    elif extra is not None:
        cfg["extra_special_tokens"] = extra


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_vl":
        raise ValueError("Only Qwen3-VL PhysBrain is supported")
    text = cfg["text_config"]
    rope = text.pop("rope_parameters", None)
    if rope is not None:
        if text.get("rope_scaling") is not None:
            raise ValueError("Ambiguous rope_scaling and rope_parameters")
        text["rope_theta"] = rope.pop("rope_theta", 5000000.0)
        text["rope_scaling"] = rope
    if text.get("rope_scaling") is None:
        raise ValueError("Missing explicit MRoPE configuration")
    output.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.name != "config.json":
            (output / file.name).symlink_to(file)
    tokenizer_compatibility(source, output)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    (output / "compatibility.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "target_transformers": "4.57.3",
                "change": "rope_parameters -> rope_theta + rope_scaling; weights unchanged",
            },
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="checkpoints/PhysBrain1.5-2B")
    p.add_argument("--output", default="checkpoints/PhysBrain1.5-2B-gr00t")
    a = p.parse_args()
    print(prepare(a.source, a.output))
