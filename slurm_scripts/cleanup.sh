#!/bin/bash

ls -l *.out
echo ""
echo "Deleting files..."
echo ""
cd ./slurm_logs/
rm *.out
cd ..
