#!/bin/bash
# This script launches the SLURM job arrays for all architectures.
set -e

# --- Experiment Parameters ---
CLIENTS=2
EPOCHS=1
ACTIVATION="crelu"
TRIALS=1 # Set the number of trials for all experiments

echo "Submitting jobs for WS, DN, and IB architectures..."
echo "Clients per job: ${CLIENTS}"
echo "Epochs per job: ${EPOCHS}"
echo "Activation function: ${ACTIVATION}"
echo "Trials per experiment: ${TRIALS}"
echo ""

# Submit one job array for each architecture, now including the number of trials
sbatch run_experiment.sh WS $CLIENTS $EPOCHS $ACTIVATION $TRIALS
sbatch run_experiment.sh DN $CLIENTS $EPOCHS $ACTIVATION $TRIALS
sbatch run_experiment.sh IB $CLIENTS $EPOCHS $ACTIVATION $TRIALS

echo ""
echo "All jobs have been submitted to the SLURM scheduler."
squeue -u $USER
