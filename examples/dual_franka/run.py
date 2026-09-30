"""YAML adapter to GR00T native training and policy server; no robot actions."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def read_config(path, parents=()):
    import yaml

    path = Path(path).resolve()
    if path in parents:
        raise ValueError("Cyclic YAML extends")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected YAML mapping")
    base = data.pop("extends", None)

    def merge(a, b):
        result = dict(a)
        for k, v in b.items():
            result[k] = (
                merge(result[k], v)
                if isinstance(v, dict) and isinstance(result.get(k), dict)
                else v
            )
        return result

    return merge(read_config(path.parent / base, (*parents, path)), data) if base else data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check", action="store_true", help="Validate without model loading or training"
    )
    args = parser.parse_args()
    raw = read_config(args.config)
    deployment = raw.pop("deployment", {})
    mode = raw.pop("mode")
    if mode == "server":
        from gr00t.eval.run_gr00t_server import ServerConfig, main as serve

        cfg = ServerConfig(**raw)
        if args.check:
            print(cfg)
        else:
            serve(cfg)
        return
    if mode != "train":
        raise ValueError(f"Unknown mode {mode}")
    from examples.dual_franka import modality
    from gr00t.configs.base_config import Config

    config = Config().load_dict(raw)
    config.data.modality_configs = {"new_embodiment": modality.CONFIG}
    for ds in config.data.datasets:
        for value in ds.dataset_paths:
            path = Path(value)
            info = json.loads((path / "meta/info.json").read_text())
            if not str(info.get("codebase_version", "")).startswith("v2"):
                raise ValueError(
                    f"{path}: GR00T requires a separate v2 conversion, never use the v3 source directly"
                )
            if info.get("gripper_action_semantics") != "heuristic_closed_plateau_zero_v1":
                raise ValueError(f"{path}: unexpected gripper label semantics")
            if not (path / "meta/modality.json").is_file():
                raise FileNotFoundError(path / "meta/modality.json")
    config.validate()
    expected_batch = deployment.get("expected_global_batch_size", 64)
    if deployment and config.training.accumulated_batch_size != expected_batch:
        raise ValueError(f"Dual-Franka configuration requires effective batch {expected_batch}")
    if config.training.resume_from_checkpoint:
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.model.modules.trainability import POLICY_FIELDS
        from transformers.trainer_utils import get_last_checkpoint

        output = Path(config.training.output_dir)
        if config.training.experiment_name:
            output = output / config.training.experiment_name
        checkpoint = config.training.resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(str(output)) if output.exists() else None
        if not checkpoint:
            raise FileNotFoundError(f"No checkpoint to resume in {output}")
        if config.training.lr_schedule_style == "openpi_cosine":
            import yaml

            saved_config = yaml.safe_load(
                (Path(checkpoint) / "experiment_cfg/config.yaml").read_text()
            )
            saved_training = saved_config["training"]
            optimizer_fields = (
                "learning_rate",
                "adam_beta1",
                "adam_beta2",
                "adam_epsilon",
                "weight_decay",
                "weight_decay_all_parameters",
                "max_grad_norm",
                "optim",
                "lr_schedule_style",
                "lr_min",
                "lr_decay_steps",
                "warmup_steps",
                "global_batch_size",
                "gradient_accumulation_steps",
                "num_gpus",
                "max_steps",
            )
            changed_optimizer = [
                k for k in optimizer_fields if saved_training.get(k) != getattr(config.training, k)
            ]
            if changed_optimizer:
                raise ValueError(
                    f"Cannot resume the aligned experiment with changed optimizer/batch: {changed_optimizer}"
                )
        saved = Gr00tN1d7Config(**json.loads((Path(checkpoint) / "config.json").read_text()))
        changed = [
            name for name in POLICY_FIELDS if getattr(saved, name) != getattr(config.model, name)
        ]
        if changed:
            raise ValueError(f"Cannot resume with changed training policy: {changed}")
    if deployment.get("continuation_schedule"):
        if config.training.lr_schedule_style != "native":
            raise ValueError("Continuation restart and OpenPI schedule cannot be combined")
        stage = deployment["continuation_schedule"]
        if not config.training.resume_from_checkpoint:
            raise ValueError("Continuation requires a complete native checkpoint")
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        start, steps = stage["start_step"], stage["steps"]
        if (
            config.training.max_steps != start + steps
            or not start <= state["global_step"] < start + steps
        ):
            raise ValueError("Checkpoint step does not match continuation stage")
        if not (Path(checkpoint) / "scheduler.pt").is_file():
            raise FileNotFoundError("Resume checkpoint lacks scheduler state")
        if not any(Path(checkpoint).glob("global_step*/*optim_states.pt")):
            raise FileNotFoundError("Resume checkpoint lacks DeepSpeed optimizer shards")
    if args.check:
        print("Native training configuration and dataset metadata validated; model not loaded")
        return
    if not config.training.resume_from_checkpoint:
        output = Path(config.training.output_dir)
        if config.training.experiment_name:
            output = output / config.training.experiment_name
        if any(output.glob("checkpoint-*")):
            raise FileExistsError(
                f"{output}: checkpoints exist; use the resume YAML or a new output directory"
            )
    n = config.training.num_gpus
    if n > 1 and "LOCAL_RANK" not in os.environ:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={n}",
                "-m",
                "examples.dual_franka.run",
                "--config",
                args.config,
            ],
            check=True,
        )
        return
    if deployment:
        os.environ["WANDB_ENTITY"] = deployment["wandb_entity"]
        os.environ["WANDB_RUN_ID"] = deployment["wandb_id"]
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_INIT_TIMEOUT"] = str(deployment.get("wandb_init_timeout_seconds", 180))
    from examples.dual_franka.training_support import configure_trainer
    from gr00t.experiment.experiment import run

    run(
        config,
        trainer_setup=lambda trainer, pipeline, cfg: configure_trainer(
            trainer, pipeline, cfg, deployment
        ),
    )


if __name__ == "__main__":
    main()
