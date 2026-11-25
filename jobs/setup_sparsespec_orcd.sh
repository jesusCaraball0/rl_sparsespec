#!/usr/bin/env bash
# Run deps/download on the login node, then build in an interactive GPU allocation.
set -eo pipefail
cd "$(dirname "$0")/.."
project=$PWD
env_path="$project/.conda-sparsespec"
source_path="$project/third_party/SparseSpec"
revision=5c9616c9af109ecec0fdf193221d3f5be6f05922

case "${1:-}" in
    deps|build|download) ;;
    *) printf 'Usage: bash %s {deps|download|build}\n' "$0" >&2; exit 2 ;;
esac
if [[ "$1" == build && ! -f "$source_path/3rdparty/raft/build.sh" ]]; then
    printf 'RAFT source is missing. First run: bash jobs/setup_sparsespec_orcd.sh deps\n' >&2
    exit 1
fi
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "$1" == deps ]]; then
    if [[ ! -x "$env_path/bin/python" ]]; then
        conda create --yes --override-channels -c rapidsai -c conda-forge -c nvidia \
            --prefix "$env_path" python=3.12 pip git 'cmake>=3.30.4,<4' ninja \
            gcc_linux-64=13 gxx_linux-64=13 cuda-toolkit=12.8 cuda-version=12.8 librmm=25.10
    fi
    conda activate "$env_path"
    if [[ ! -d "$source_path" ]]; then
        mkdir -p "$project/third_party"
        git clone --filter=blob:none --no-checkout \
            https://github.com/sspec-project/SparseSpec.git "$source_path"
        git -C "$source_path" checkout "$revision"
    fi
    test "$(git -C "$source_path" rev-parse HEAD)" = "$revision"
    git -C "$source_path" submodule update --init --recursive
    # Load vLLM's all-reduce extension only when a multi-GPU communicator is used.
    patch_file="$project/jobs/sparsespec_lazy_allreduce.patch"
    if ! git -C "$source_path" apply --reverse --check "$patch_file" 2>/dev/null; then
        git -C "$source_path" apply "$patch_file"
    fi
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    python -m pip install -r requirements.txt
    # Keep the upstream fork available for future multi-GPU work.
    VLLM_USE_PRECOMPILED=1 python -m pip install --no-build-isolation \
        -e "$source_path/3rdparty/vllm" 'torch==2.8.0' 'transformers==4.55.2'
    python -m pip install --no-build-isolation -e "$source_path/3rdparty/flashinfer"
    python -m pip check
    exit 0
fi

conda activate "$env_path"
export HF_HOME="${HF_HOME:-$project/.cache/huggingface}"
if [[ "$1" == download ]]; then
    # Keep the model directory's name: upstream infers model size from its suffix.
    python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="models/Qwen3-1.7B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
)
print(f"Model ready at {path}")
PY
else
    export CUDA_HOME="$CONDA_PREFIX"
    export INSTALL_PREFIX="$CONDA_PREFIX"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-4}"
    export MAX_JOBS="$PARALLEL_LEVEL"
    # FlashInfer's JIT kernels must target the allocated GPU, not old inherited targets.
    TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')
    export TORCH_CUDA_ARCH_LIST
    (cd "$source_path/3rdparty/raft" && bash build.sh libraft)
    # Run the toy prompt once to warm the CUDA kernels.
    HF_HUB_OFFLINE=1 python -u -X faulthandler inference.py
fi
