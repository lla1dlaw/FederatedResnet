#!/bin/bash
#SBATCH --job-name=Federated-ResNet-Job
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liamlaidlaw@boisestate.edu

# ==============================================================================
#                      Federated Learning Experiment Script
# ==============================================================================
# This is a reusable SLURM script for running a federated learning experiment.
# It accepts five command-line arguments:
#   1. Architecture (e.g., 'WS', 'DN', 'IB')
#   2. Number of Clients (e.g., 10)
#   3. Number of Epochs (e.g., 200)
#   4. Activation Function (e.g., 'crelu')
#   5. Number of Trials (e.g., 3)
#
# Example Usage: sbatch run_experiment.sh WS 10 200 crelu 3
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Argument Validation ---
if [ "$#" -ne 5 ]; then
  echo "ERROR: Illegal number of parameters."
  echo "Usage: sbatch $0 [ARCHITECTURE] [NUM_CLIENTS] [NUM_EPOCHS] [ACTIVATION] [NUM_TRIALS]"
  exit 1
fi

# --- Assign Arguments to Variables ---
ARCH=$1
CLIENTS=$2
EPOCHS=$3
ACTIVATION=$4
TRIALS=$5

# --- JOB SETUP ---
echo "======================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $HOSTNAME"
echo "---"
echo "EXPERIMENT PARAMETERS:"
echo "Architecture: ${ARCH}"
echo "Clients: ${CLIENTS}"
echo "Epochs: ${EPOCHS}"
echo "Activation: ${ACTIVATION}"
echo "Trials: ${TRIALS}"
echo "======================================================"

# 1. Purge modules and load Conda/CUDA
module purge
module load conda
module load cudnn8.5-cuda11.7/8.5.0.96
module load gcc/10.2.0
echo "Modules loaded."

# 2. Activate your Conda environment
conda activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 3. Diagnostic checks
echo "--- Running Diagnostics ---"
nvidia-smi
echo "Python path: $(which python)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo "---------------------------"

# --- EXPERIMENT CONFIGURATION ---
# AGGREGATIONS=("arithmetic" "circular" "hybrid")
AGGREGATIONS=("hybrid")

# --- JOB EXECUTION ---
if [ $SLURM_ARRAY_TASK_ID -le 3 ]; then
  # --- Run ComplexResNet Permutations (Jobs 1-3) ---
  AGG_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
  AGG=${AGGREGATIONS[$AGG_INDEX]}

  echo "Running ComplexResNet: Arch=${ARCH}, Activation=${ACTIVATION}, Aggregation=${AGG}"

  python ../main.py \
    --model ComplexResNet \
    --arch $ARCH \
    --act $ACTIVATION \
    --agg $AGG \
    --learn_imaginary \
    --numclients $CLIENTS \
    --epochs $EPOCHS \
    --num_trials $TRIALS \
    --tqdm_mode local

else
  # --- Run RealResNet (Job 4) ---
#  AGG="arithmetic"
#
#  echo "Running RealResNet: Arch=${ARCH}, Aggregation=${AGG}"
#
#  python ../main.py \
#    --model RealResNet \
#    --arch $ARCH \
#    --agg $AGG \
#    --numclients $CLIENTS \
#    --epochs $EPOCHS \
#    --num_trials $TRIALS \
#    --tqdm_mode local
fi

echo "Job finished successfully."
