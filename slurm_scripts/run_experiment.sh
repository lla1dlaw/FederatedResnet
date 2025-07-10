#!/bin/bash
#
# This is a reusable SLURM script for running a federated learning experiment.
# It accepts three command-line arguments:
#   1. Architecture (e.g., 'WS', 'DN', 'IB')
#   2. Number of Clients (e.g., 10)
#   3. Number of Epochs (e.g., 200)
#
# Example Usage: sbatch run_experiment.sh WS 10 200
#
set -e # Exit immediately if a command exits with a non-zero status.

# --- Argument Validation ---
if [ "$#" -ne 3 ]; then
  echo "ERROR: Illegal number of parameters."
  echo "Usage: sbatch $0 [ARCHITECTURE] [NUM_CLIENTS] [NUM_EPOCHS]"
  exit 1
fi

# --- SLURM Configuration ---
ARCH=$1
CLIENTS=$2
EPOCHS=$3

#SBATCH --job-name=FL_${ARCH}             # Dynamic job name based on architecture
#SBATCH --partition=gpu                 # Specify the GPU partition
#SBATCH --gres=gpu:l40:1                # Request one L40 GPU
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Allocate 4 CPUs per task
#SBATCH --mem=16G                       # Request 16GB of memory
#SBATCH --time=48:00:00                 # Set a 48-hour time limit per job
#SBATCH --output=slurm_logs/slurm-${ARCH}-%A_%a.out # Dynamic log file name
#SBATCH --array=1-4                     # Run 4 jobs per architecture
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liamlaidlaw@boisestate.edu

# --- JOB SETUP ---
echo "======================================================"
echo "Starting job $SLURM_JOB_ID for architecture ${ARCH}"
echo "Host: $HOSTNAME"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Memory: ${SLURM_MEM_PER_NODE} MB"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Clients: ${CLIENTS}"
echo "Epochs: ${EPOCHS}"
echo "======================================================"

# 1. Purge modules and load Conda/CUDA
module purge
module load conda
module load cudnn8.5-cuda11.7/8.5.0.96
echo "Modules loaded."

# 2. Activate your Conda environment
source activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 3. Diagnostic checks
echo "--- Running Diagnostics ---"
nvidia-smi
echo "Python path: $(which python)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo "---------------------------"

# --- EXPERIMENT CONFIGURATION ---
# This script runs a targeted set of model permutations for the given ARCH.
# - ComplexResNet: 'crelu' activation, with 'arithmetic', 'circular', and 'hybrid' aggregations.
# - RealResNet: only the 'arithmetic' aggregation strategy.

AGGREGATIONS=("arithmetic" "circular" "hybrid")

# --- JOB EXECUTION ---
if [ $SLURM_ARRAY_TASK_ID -le 3 ]; then
  # --- Run ComplexResNet Permutations (Jobs 1-3) ---
  AGG_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
  AGG=${AGGREGATIONS[$AGG_INDEX]}
  ACT="crelu"

  echo "Running ComplexResNet: Arch=${ARCH}, Activation=${ACT}, Aggregation=${AGG}"

  python main.py \
    --model ComplexResNet \
    --architecture_type $ARCH \
    --complex_activations $ACT \
    --aggregation_strategy $AGG \
    --learn_imaginary \
    --numclients $CLIENTS \
    --epochs $EPOCHS \
    --tqdm_mode local \
    --save "ComplexResNet-${ARCH}-${ACT}-${CLIENTS}_clients-${AGG}-learn_imag"

else
  # --- Run RealResNet (Job 4) ---
  AGG="arithmetic"

  echo "Running RealResNet: Arch=${ARCH}, Aggregation=${AGG}"

  python main.py \
    --model RealResNet \
    --architecture_type $ARCH \
    --aggregation_strategy $AGG \
    --numclients $CLIENTS \
    --epochs $EPOCHS \
    --tqdm_mode local \
    --save "RealResNet-${ARCH}-${CLIENTS}_clients-${AGG}"
fi

echo "Job finished successfully."
