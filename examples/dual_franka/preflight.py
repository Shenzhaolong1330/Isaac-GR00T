"""Submitted-job checks. No training, no robot hardware, no credential output."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def check_wandb_access(cfg, *, api_factory=None, sleep=time.sleep):
    """Check one project; retry transport failures without hiding authentication errors."""
    import requests
    import wandb

    deployment = cfg["deployment"]
    timeout = deployment.get("wandb_timeout_seconds", 90)
    attempts = deployment.get("wandb_check_attempts", 3)
    if (
        type(timeout) is not int
        or timeout <= 0
        or type(attempts) is not int
        or not 1 <= attempts <= 5
    ):
        raise ValueError("W&B timeout must be positive; attempts must be 1..5")
    factory = api_factory or wandb.Api
    entity = deployment["wandb_entity"]
    project = cfg["training"]["wandb_project"]
    for attempt in range(1, attempts + 1):
        print(
            f"Checking W&B {entity}/{project} (attempt {attempt}/{attempts}, timeout {timeout}s)",
            flush=True,
        )
        try:
            api = factory(timeout=timeout)
            if not api.viewer:
                raise RuntimeError("W&B authentication failed; configure WANDB_API_KEY in the job")
            # Project construction is lazy: accessing id forces the actual target-project query.
            if not api.project(project, entity=entity).id:
                raise RuntimeError(f"W&B project not accessible: {entity}/{project}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            print(f"W&B transport failure: {type(exc).__name__}", flush=True)
            if attempt == attempts:
                raise RuntimeError(
                    "W&B network check failed after bounded retries. Check the submitted job's "
                    "outbound HTTPS/proxy access to api.wandb.ai:443. Training was not started; "
                    "credentials were not printed and offline mode was not enabled."
                ) from exc
            sleep(5)
        else:
            print(
                "W&B authenticated; target project readable; upload checked at training init",
                flush=True,
            )
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    import torch

    if args.distributed:
        import torch.distributed as dist

        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
        value = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(value)
        assert value.item() == 10.0, value
        dist.destroy_process_group()
        if rank == 0:
            print("Four-rank NCCL all_reduce passed", flush=True)
        return
    from examples.dual_franka.run import read_config

    cfg = read_config(args.config)
    assert torch.cuda.device_count() == 4, "Exactly four visible GPUs are required"
    assert Path(sys.executable).resolve().is_relative_to("/user/users/szl/envs")
    subprocess.run(
        [sys.executable, "-m", "examples.dual_franka.check_environment", "--cuda"], check=True
    )
    subprocess.run(
        [sys.executable, "-m", "examples.dual_franka.run", "--config", args.config, "--check"],
        check=True,
    )
    for ds in cfg["data"]["datasets"]:
        train = Path(ds["dataset_paths"][0])
        val = Path(ds["val_dataset_path"])
        assert (train / "meta/stats.json").read_bytes() == (val / "meta/stats.json").read_bytes()
    check_wandb_access(cfg)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "examples.dual_franka.preflight",
            "--config",
            args.config,
            "--distributed",
        ],
        check=True,
    )
    out = Path("deployment_records/job_preflight.json")
    out.write_text(
        json.dumps(
            {
                "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
                "python": str(Path(sys.executable).resolve()),
                "config": args.config,
                "wandb_team_access": True,
                "nccl": True,
            },
            indent=2,
        )
    )
    print("Job preflight passed; training has NOT been started", flush=True)


if __name__ == "__main__":
    main()
