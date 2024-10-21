#!/bin/bash -l
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # tasks/processes/cores per node max 112 for DCGP and 32 for Booster
#SBATCH --cpus-per-task=4            # threads/cpus per task
#SBATCH --mem=42GB                   # memory per simulation
#SBATCH --time=12:00:00              # time limits: 12 hours
#SBATCH --account=CNHPC_1497299    # account name (CNHPC_1497299_0 or CNHPC_1497299 for dcgp and booster respectively)
#SBATCH --error=/leonardo_scratch/large/userexternal/ntriant1/database/final/logs/Job_%A_%a.err            # standard error file 
#SBATCH --output=/leonardo_scratch/large/userexternal/ntriant1/database/final/logs/Job_%A_%a.out           # standard output file``
#SBATCH --partition=boost_usr_prod    # partition name (<dcgp or boost>_usr_prod)
#SBATCH --qos=normal                 # quality of service

#SBATCH --array=1-8                 # number of simulations 

source ~/.bashrc
module load python/3.10
module load gcc/12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gsl/2.7.1--gcc--12.2.0
module load openmpi/4.1.6--gcc--12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0


source /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/activate  



COUNTER_PATH="/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/counter.txt"
COUNTER=$(cat $COUNTER_PATH)


# ARGS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
ARGS=({0..20})
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 


srun /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/run_single.py \
--counter ${ARGS[$SLURM_ARRAY_TASK_ID-1]} \
--threads $SLURM_CPUS_PER_TASK &


wait
