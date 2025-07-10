#!/bin/bash

sbatch run_WS_experiment.slurm
sbatch run_DN_experiment.slurm
sbatch run_IB_experiment.slurm

echo ""
echo "All experiments completed."
