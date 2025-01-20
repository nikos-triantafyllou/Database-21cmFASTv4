#!/bin/bash -l
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # tasks/processes/cores per node max 112 for DCGP and 32 for Booster
#SBATCH --cpus-per-task=9            # threads/cpus per task
#SBATCH --mem=100GB
#SBATCH --time=12:00:00              # time limits: 12 hours
#SBATCH --error=myJob.err            # standard error file
#SBATCH --account=CNHPC_1497299_0    # account name
#SBATCH --output=myJob.out           # standard output file``
#SBATCH --partition=dcgp_usr_prod    # partition name
#SBATCH --qos=normal                 # quality of service

#SBATCH --array=1-21

source ~/.bashrc
module load python/3.10
module load gcc/12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gsl/2.7.1--gcc--12.2.0
module load openmpi/4.1.6--gcc--12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0


source /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/activate  

PROCESS_NAME="plus_minus"
CORES=9      # Number of cores pre simulation
PROCESSES=21 # Number of simulations


COUNTER_PATH="/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/counter.txt"
COUNTER=$(cat $COUNTER_PATH)


ARGS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 


srun /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/save_properties.py \
--counter ${ARGS[$SLURM_ARRAY_TASK_ID-1]}&

wait





