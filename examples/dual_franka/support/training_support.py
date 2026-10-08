"""Dual-Franka acceptance callbacks around the native GR00T Trainer."""

import copy
import json
import math
from pathlib import Path
import time

from examples.dual_franka.support.modality import CONFIG, KEYS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType
from gr00t.utils.dist_utils import run_or_wait_on_rank0
import numpy as np
import torch
from transformers import TrainerCallback


class AcceptanceCallback(TrainerCallback):
    def __init__(self, trainer, pipeline, config, deployment):
        self.trainer, self.pipeline, self.config, self.deployment = (
            trainer,
            pipeline,
            config,
            deployment,
        )
        self.out = Path(trainer.args.output_dir)
        self.started = time.monotonic()
        self.last_log_time = self.started
        self.last_log_step = 0
        self.samples = []
        self.grad = {}
        self.tracked = {}
        self.update_samples = {}
        model = pipeline.return_model()
        counts = {}
        groups = {
            "vision": [
                (n, p)
                for n, p in model.named_parameters()
                if n.startswith("backbone.model.model.visual.")
            ],
            "language": [
                (n, p)
                for n, p in model.named_parameters()
                if n.startswith("backbone.model.model.language_model.")
            ],
            "ae": [(n, p) for n, p in model.named_parameters() if n.startswith("action_head.")],
        }
        adapters = [(n, p) for n, p in model.named_parameters() if ".lora_" in n]
        if adapters:
            groups["lora"] = adapters
        for label, params in groups.items():
            counts[label] = {
                "total": sum(p.numel() for _, p in params),
                "trainable": sum(p.numel() for _, p in params if p.requires_grad),
            }
            active = [(n, p) for n, p in params if p.requires_grad and p.numel() > 128]
            # B starts at zero, so it gets the first LoRA gradient; A may initially have zero gradient.
            preferred = [(n, p) for n, p in active if ".lora_B." in n]
            n, p = (preferred or active or params)[0]
            self.tracked[label] = (n, p, p.detach().flatten()[:128].cpu().clone())
            if p.requires_grad:
                p.register_hook(lambda g, k=label, param=p: self.record_grad(k, g, param))
        with run_or_wait_on_rank0(label="dual_franka.validation_setup") as rank0:
            if rank0:
                (self.out / "parameter_policy.json").write_text(json.dumps(counts, indent=2))
                if hasattr(model, "trainability_report"):
                    (self.out / "trainable_parameters.json").write_text(
                        json.dumps(model.trainability_report, indent=2)
                    )
                self.prepare_validation()

    def log_metrics(self, metrics):
        # Trainer.log triggers on_log and clears control.should_log. Calling it
        # only on rank 0 desynchronizes the ranks' next loss all_reduce.
        step = self.trainer.state.global_step
        self.trainer.state.log_history.append({**metrics, "step": step})
        if self.config.training.use_wandb:
            import wandb

            wandb.log({**metrics, "train/global_step": step})

    def record_grad(self, key, g, param):
        if key not in self.grad:
            self.grad[key] = float(g.detach().float().norm())
            indices = g.detach().flatten().abs().topk(32).indices
            self.update_samples[key] = (indices, param.detach().flatten()[indices].cpu().clone())
        return g

    def prepare_validation(self):
        path = Path(self.config.data.datasets[0].val_dataset_path)
        loader = LeRobotEpisodeLoader(path, CONFIG, decoder_kwargs={"num_ffmpeg_threads": 1})
        groups = {}
        for i, e in enumerate(loader.episodes_metadata):
            if e["length"] >= 40:
                groups.setdefault((e["source"], tuple(e["tasks"])), []).append(i)
        rng = np.random.default_rng(42)
        strata = sorted(groups)
        chosen = []
        for i in range(self.deployment.get("validation_samples", 32)):
            ids = groups[strata[i % len(strata)]]
            ep = int(rng.choice(ids))
            frame = int(rng.integers(loader.episodes_metadata[ep]["length"] - 39))
            data = loader[ep]
            sample = extract_step_data(data, frame, CONFIG, EmbodimentTag.NEW_EMBODIMENT)
            sample.images = {
                k: [np.asarray(v).copy() for v in vals] for k, vals in sample.images.items()
            }
            self.samples.append(sample)
            chosen.append({"episode": ep, "frame": frame})
            del data
        (self.out / "validation_samples.json").write_text(json.dumps(chosen, indent=2))

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_log_step = state.global_step
        self.last_log_time = time.monotonic()
        # Snapshot after DeepSpeed dtype conversion; BF16 rounding is not an optimizer update.
        self.tracked = {
            k: (n, p, p.detach().flatten()[:128].cpu().clone())
            for k, (n, p, _) in self.tracked.items()
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 and self.trainer.is_world_process_zero():
            now = time.monotonic()
            count = state.global_step - self.last_log_step
            elapsed = now - self.last_log_time
            self.log_metrics(
                {
                    "system/seconds_per_optimizer_step": elapsed / max(1, count),
                    "system/samples_per_second": count
                    * self.config.training.accumulated_batch_size
                    / max(elapsed, 1e-9),
                    "system/peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            self.last_log_step = state.global_step
            self.last_log_time = now
        if state.global_step == 2 and self.trainer.is_world_process_zero():
            report = {}
            for key, (name, param, old) in self.tracked.items():
                if param.requires_grad:
                    change = (
                        float(
                            (
                                param.detach().flatten()[self.update_samples[key][0]].cpu()
                                - self.update_samples[key][1]
                            )
                            .abs()
                            .max()
                        )
                        if key in self.update_samples
                        else None
                    )
                else:
                    change = float((param.detach().flatten()[:128].cpu() - old).abs().max())
                report[key] = {
                    "parameter": name,
                    "trainable": param.requires_grad,
                    "gradient_norm": self.grad.get(key),
                    "sample_max_change": change,
                }
            (self.out / "first_update.json").write_text(json.dumps(report, indent=2))
            self.log_metrics(
                {
                    f"update/{group}/{key}": v
                    for group, entry in report.items()
                    for key, v in entry.items()
                    if isinstance(v, (float, int))
                }
            )
        if state.global_step % self.deployment.get("validation_interval", 250) == 0:
            with run_or_wait_on_rank0(label="dual_franka.validation") as rank0:
                if rank0:
                    self.evaluate(state.global_step)
        stop = self.deployment.get("stop_after_step")
        if stop and state.global_step >= stop:
            control.should_save = True
            control.should_training_stop = True
        return control

    def evaluate(self, step):
        model = self.pipeline.return_model()
        processor = self.pipeline.return_processor()
        was_training = model.training
        model.eval()
        processor.eval()
        losses = []
        errors = []
        targets = []
        predictions = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            for sample in self.samples:
                messages = [{"type": MessageType.EPISODE_STEP.value, "content": sample}]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    batch = processor.collator([processor(messages)])
                    losses.append(float(model(**copy.deepcopy(batch))["loss"]))
                    batch["inputs"].pop("action", None)
                    pred = model.get_action(**batch)["action_pred"].float().cpu().numpy()
                states = {k: v[None] for k, v in sample.states.items()}
                actions = processor.decode_action(pred, EmbodimentTag.NEW_EMBODIMENT, states)
                actual = np.concatenate([actions[k][0] for k in KEYS], axis=-1)
                target = np.concatenate([sample.actions[k] for k in KEYS], axis=-1)
                assert actual.shape == target.shape == (40, 14) and np.isfinite(actual).all()
                errors.append(actual - target)
                targets.append(target)
                predictions.append(actual)
        e = np.concatenate(errors)
        target = np.concatenate(targets)
        prediction = np.concatenate(predictions)
        from scipy.spatial.transform import Rotation

        rotations = {}
        for side, start in [("left", 3), ("right", 9)]:
            relative = (
                Rotation.from_rotvec(prediction[:, start : start + 3])
                * Rotation.from_rotvec(target[:, start : start + 3]).inv()
            )
            rotations[f"validation/{side}_rotation_deg"] = float(
                np.rad2deg(relative.magnitude()).mean()
            )
        metrics = {
            "validation/flow_loss": float(np.mean(losses)),
            "validation/left_translation_mm": float(
                np.linalg.norm(e[:, :3], axis=-1).mean() * 1000
            ),
            "validation/right_translation_mm": float(
                np.linalg.norm(e[:, 6:9], axis=-1).mean() * 1000
            ),
            "validation/left_gripper_mae_mm": float(np.abs(e[:, 12]).mean() * 1000),
            "validation/right_gripper_mae_mm": float(np.abs(e[:, 13]).mean() * 1000),
            "system/peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "system/elapsed_seconds": time.monotonic() - self.started,
        }
        metrics.update(rotations)
        for side, index in [("left", 12), ("right", 13)]:
            closed = target[:, index] < 0.001
            if closed.any():
                metrics[f"validation/{side}_closed_gripper_mae_mm"] = float(
                    np.abs(e[closed, index]).mean() * 1000
                )
        metrics.update(
            {f"validation/mae_dim_{i}": float(np.abs(e[:, i]).mean()) for i in range(14)}
        )
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite validation metric")
        self.log_metrics(metrics)
        with (self.out / "validation_metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        model.train(was_training)
        processor.train()


class ContinuationScheduleCallback(TrainerCallback):
    """Restart only the LR curve after native model/optimizer/RNG restoration."""

    def __init__(self, trainer, schedule):
        self.trainer = trainer
        self.start = int(schedule["start_step"])
        self.steps = int(schedule["steps"])
        self.warmup = int(schedule["warmup_steps"])
        if not (self.start >= 0 and 0 < self.warmup < self.steps):
            raise ValueError("Invalid continuation schedule")
        if trainer.args.max_steps != self.start + self.steps:
            raise ValueError("Continuation schedule must end at training.max_steps")
        if trainer.args.lr_scheduler_type != "cosine":
            raise ValueError("Continuation requires the existing cosine scheduler")

    def factor(self, global_step):
        step = max(0, global_step - self.start)
        if step < self.warmup:
            return step / self.warmup
        progress = min(1.0, (step - self.warmup) / (self.steps - self.warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.start <= state.global_step <= self.start + self.steps:
            raise ValueError("Resume checkpoint is outside the continuation stage")
        scheduler = self.trainer.lr_scheduler
        while hasattr(scheduler, "scheduler"):
            scheduler = scheduler.scheduler
        if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            raise TypeError("Expected native Transformers LambdaLR scheduler")
        engine = getattr(self.trainer, "deepspeed", None)
        # Accelerate may own the scheduler while DeepSpeed only owns optimizer steps.
        # In that supported path engine.lr_scheduler is None and Trainer steps its wrapper.
        engine_scheduler = getattr(engine, "lr_scheduler", None)
        while hasattr(engine_scheduler, "scheduler"):
            engine_scheduler = engine_scheduler.scheduler
        if engine_scheduler is not None and engine_scheduler is not scheduler:
            raise RuntimeError("DeepSpeed and Trainer use different active LR schedulers")
        if not all(math.isclose(lr, args.learning_rate) for lr in scheduler.base_lrs):
            raise ValueError("Cannot silently change the resumed optimizer base learning rate")
        # Mutate the shared scheduler AFTER checkpoint restore, not the optimizer moments.
        # A later resume uses the saved GLOBAL step, so warmup is not restarted again.
        scheduler.lr_lambdas = [self.factor for _ in scheduler.base_lrs]
        scheduler.last_epoch = state.global_step
        rates = [lr * self.factor(state.global_step) for lr in scheduler.base_lrs]
        for group, lr in zip(scheduler.optimizer.param_groups, rates):
            group["lr"] = lr
        scheduler._last_lr = rates
        if state.is_world_process_zero:
            report = {
                "start_step": self.start,
                "steps": self.steps,
                "warmup_steps": self.warmup,
                "restored_global_step": state.global_step,
                "learning_rates": rates,
                "optimizer_state_reset": False,
                "scheduler_owner": "deepspeed" if engine_scheduler is not None else "trainer",
            }
            (Path(args.output_dir) / "continuation_schedule.json").write_text(
                json.dumps(report, indent=2)
            )
            print(f"Continuation LR schedule: {report}", flush=True)
        return control


def configure_trainer(trainer, pipeline, config, deployment):
    if deployment:
        if deployment.get("continuation_schedule"):
            trainer.add_callback(
                ContinuationScheduleCallback(trainer, deployment["continuation_schedule"])
            )
        trainer.add_callback(AcceptanceCallback(trainer, pipeline, config, deployment))
