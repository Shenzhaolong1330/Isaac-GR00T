#!/usr/bin/env bash
# Run this once inside a four-GPU submitted task. No robot interfaces.
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: bash $0 examples/dual_franka/configs/train/train_<experiment>.yaml" >&2
  exit 2
fi
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.tools.preflight --config "$1"
exec python -m examples.dual_franka.run --config "$1"
