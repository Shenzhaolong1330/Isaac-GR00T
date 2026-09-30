#!/usr/bin/env bash
set -euo pipefail
cd /user/users/szl/szl_ws/Isaac-GR00T
exec bash examples/dual_franka/submit.sh examples/dual_franka/train_full_100k.yaml
