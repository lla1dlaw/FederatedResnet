#!/bin/bash
#SBATCH --job-name=train_models_final   # Job name for identification
#SBATCH --partition=gpu                 # Specify the GPU partition
#SBATCH --gres=gpu:l40:1                # Request one L40 GPU
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Allocate 4 CPUs per task
#SBATCH --mem=16G                       # Request 16GB of memory
#SBATCH --time=48:00:00                 # Set a 48-hour time limit per job
#SBATCH --output=slurm_logs/slurm-%A_%a.out # Standard output and error log
#SBATCH --array=1-4                     # Run 4 jobs in total
#SBATCH --mail-type=ALL             # Send email on job start, end, and failure
#SBATCH --mail-user=liamlaidlaw@boisestate.edu

# --- JOB EXECUTION ---

# Print job information
echo "======================================================"
echo "Starting job $SLURM_JOB_ID on host $HOSTNAME"
echo "Job allocated to partition: ${SLURM_JOB_PARTITION}"
echo "Job allocated CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Job allocated Memory: ${SLURM_MEM_PER_NODE} MB"
echo "Job allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: ${BATCH_SIZE:=128}" # Default to 128 if not set
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
echo "Which python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo "---------------------------"

# --- Configuration ---
# This script runs a targeted set of model permutations.
# - ComplexResNet: 'DN' architecture, 'crelu' activation, with 'arithmetic', 'circular', and 'hybrid' aggregations.
# - RealResNet: 'DN' architecture, with only the 'arithmetic' aggregation strategy.

# Define the array for aggregation strategies
AGGREGATIONS=("arithmetic" "circular" "hybrid")

# --- Job Execution ---

# Determine which model and parameters to run based on the job array ID
if [ $SLURM_ARRAY_TASK_ID -le 3 ]; then
  # --- Run ComplexResNet Permutations (Jobs 1-3) ---
  # These jobs train the ComplexResNet with a fixed architecture ('DN') and activation ('crelu').
  # The aggregation strategy is selected based on the array task ID.

  AGG_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
  AGG=${AGGREGATIONS[$AGG_INDEX]}
  ARCH="DN"
  ACT="crelu"

  echo "Running ComplexResNet: Arch=${ARCH}, Activation=${ACT}, Aggregation=${AGG}"

  python main.py \
    --model ComplexResNet \
    --architecture_type $ARCH \
    --complex_activations $ACT \
    --aggregation_strategy $AGG \
    --learn_imaginary \
    --numclients 10 \
    --epochs 200 \
    --tqdm_mode local \
    --save "ComplexResNet-${ARCH}-${ACT}-2_clients-${AGG}-learn_imag"

else
  # --- Run RealResNet (Job 4) ---
  # This job trains the RealResNet.
  # Architecture is fixed to 'DN' and the aggregation strategy is hardcoded to 'arithmetic',
  # as it is the only compatible one.

  AGG="arithmetic"
  ARCH="DN"

  echo "Running RealResNet: Arch=${ARCH}, Aggregation=${AGG}"

  python main.py \
    --model RealResNet \
    --architecture_type $ARCH \
    --aggregation_strategy $AGG \
    --numclients 10 \
    --epochs 200 \
    --tqdm_mode local \
    --save "RealResNet-${ARCH}-2_clients-${AGG}"
fi
