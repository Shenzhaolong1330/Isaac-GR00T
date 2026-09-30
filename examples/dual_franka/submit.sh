#!/usr/bin/env bash
# Run this once inside a four-GPU submitted task. No robot interfaces.
set -euo pipefail
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
# With an explicit YAML, run exactly that experiment (no automatic second stage).
if (( $# > 0 )); then
  python -m examples.dual_franka.preflight --config "$1"
  python -m examples.dual_franka.run --config "$1"
  exit 0
fi
python -m examples.dual_franka.preflight --config examples/dual_franka/train_job.yaml
python -m examples.dual_franka.run --config examples/dual_franka/train_job.yaml
python -m examples.dual_franka.run --config examples/dual_franka/train_job_resume.yaml
