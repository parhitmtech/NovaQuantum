#!/bin/bash
# The GPU statevector simulator's checks and benchmark as a SOL batch job: from this directory on SOL,
# `sbatch run_sol.sh` (any arguments go to ./qsim_bench, e.g. `sbatch run_sol.sh --max 30 --cpu-max 26`). The output
# lands in qsim-<job id>.out here.
#SBATCH --job-name=nova-qsim
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=qsim-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:-.}"

# A CUDA toolkit: CUDA_MODULE names one (see `module avail cuda`), or a default one is tried
module load "${CUDA_MODULE:-cuda-12.6.1-gcc-12.1.0}" 2>/dev/null || module load cuda 2>/dev/null || true
command -v nvcc >/dev/null || { echo "no nvcc: set CUDA_MODULE to one of \`module avail cuda\`"; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
lscpu | grep "Model name"
make clean > /dev/null
make ARCH="${ARCH:-sm_80}"

echo; echo "== The drop-in runtime: CPU and GPU, the same output"
make dropin-check
cat dropin_gpu.txt

echo; echo "== The benchmark: every amplitude compared up to --cpu-max qubits"
./qsim_bench "$@"
