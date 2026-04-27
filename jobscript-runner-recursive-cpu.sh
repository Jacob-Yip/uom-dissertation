#!/bin/bash --login
#SBATCH -p serial
### Required flags
#SBATCH -t 4-0              # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)
### Optional flags
#SBATCH -n 1                # (or --ntasks=) Number of CPU (host) cores (default is 1)

module purge
# Install PyTorch
module load apps/binapps/pytorch/2.3.0-311-cpu
# Check whether PyTorch is installed
echo "===================================================="
python -c "import torch; print(f'Torch installed: {torch.__version__}; Number of CPU threads detected: {torch.get_num_threads()}')"
echo "===================================================="

echo "===================================================="
echo "Job is using $SLURM_NTASKS CPU core(s)"
echo "===================================================="

echo "===================================================="
echo "Install python packages ..."
pip install -r requirements.txt
echo "===================================================="

echo "===================================================="
echo "Python version: "
python --version
echo "===================================================="

echo "===================================================="
echo "Running python -m src.runner_recursive_cpu ..."
python -m src.runner_recursive_cpu
echo "===================================================="

echo "Job finishes running ..."
