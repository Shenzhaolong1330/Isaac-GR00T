#!/usr/bin/env bash
# Foreground entry for a four-GPU submitted task; YAML is mandatory.
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: bash submit.sh examples/dual_franka/train_{full,lora,freeze}[_resume].yaml" >&2
  exit 2
fi
cd /user/users/szl/szl_ws/Isaac-GR00T
source examples/dual_franka/env.sh
python -m examples.dual_franka.preflight --config "$1"
python -m examples.dual_franka.run --config "$1"
