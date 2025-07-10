#!/bin/bash
# This script launches the SLURM job arrays for all architectures.
set -e

# --- Experiment Parameters ---
CLIENTS=10
EPOCHS=200

echo "Submitting jobs for WS, DN, and IB architectures..."
echo "Clients per job: ${CLIENTS}"
echo "Epochs per job: ${EPOCHS}"
echo ""

# Submit one job array for each architecture
sbatch run_experiment.sh WS $CLIENTS $EPOCHS
sbatch run_experiment.sh DN $CLIENTS $EPOCHS
sbatch run_experiment.sh IB $CLIENTS $EPOCHS

echo ""
echo "All jobs have been submitted to the SLURM scheduler."
squeue -u $USER
