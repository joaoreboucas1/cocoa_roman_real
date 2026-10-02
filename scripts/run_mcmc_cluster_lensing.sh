#!/bin/bash
#SBATCH --job-name=MCMC_CLUSTER_LENSING
#SBATCH --output=./projects/roman_real/logs/%x_%j.out
#SBATCH --error=./projects/roman_real/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

# Submit from the cocoa main folder (Cocoa/):
#   sbatch ./projects/roman_real/scripts/run_mcmc_cluster_lensing.sh

echo Job starting at `date` on node `hostname`

YAML=./projects/roman_real/EXAMPLE_MCMC_CLUSTER_LENSING1.yaml

# Clear the environment from any previously loaded modules
# module purge > /dev/null 2>&1
module load micromamba
source ~/.bashrc

cd $SLURM_SUBMIT_DIR
micromamba activate cocoa
source start_cocoa.sh

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -r resumes an existing chain with the same output name (or starts a new one)
mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ob1 \
  --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
  --mca mpi_yield_when_idle 1 \
  --report-bindings \
  cobaya-run ${YAML} -r
