# Source this file in the submitted task (Linux x86_64, NVIDIA driver required).
export VIRTUAL_ENV=/user/users/szl/envs/gr00t
export GR00T_SUPPORT=/user/users/szl/envs/gr00t-support
export GR00T_RUNTIME=/user/users/szl/envs/gr00t-runtime
export CUDA_HOME="$GR00T_SUPPORT/cuda"
export PATH="$VIRTUAL_ENV/bin:$GR00T_SUPPORT/bin:$CUDA_HOME/bin:$GR00T_RUNTIME/bin:$PATH"
export LD_LIBRARY_PATH="$GR00T_SUPPORT/ffmpeg/lib:$CUDA_HOME/lib64"
export CC="$GR00T_RUNTIME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$GR00T_RUNTIME/bin/x86_64-conda-linux-gnu-g++"
unset PYTHONPATH LD_PRELOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME=/user/users/szl/cache/gr00t-huggingface
export TORCH_EXTENSIONS_DIR=/user/users/szl/cache/gr00t-torch-extensions
export TRITON_CACHE_DIR=/user/users/szl/cache/gr00t-triton
export WANDB_DIR=/user/users/szl/runs/gr00t_n17
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR"
