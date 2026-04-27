#!/bin/bash --login
#SBATCH -p gpuA             # A100 (80GB) GPUs  [up to 12 CPU cores per GPU permitted]
### Required flags
#SBATCH -G 2                # (or --gpus=N) Number of GPUs
#SBATCH -t 1-0              # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)
### Optional flags
#SBATCH -n 1                # (or --ntasks=) Number of CPU (host) cores (default is 1)
                            # See above for number of cores per GPU you can request.
                            # Also affects host RAM allocated to job unless --mem=num used.

module purge
module load libs/cuda       # See below for specific versions

echo "Displaying machine details ..."
echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"
echo "Job finishes running ..."
