#!/bin/bash -l
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=84         # tasks/processes/cores per node max 112 for DCGP and 32 for Booster
#SBATCH --cpus-per-task=1            # threads/cpus per task
#SBATCH --time=12:00:00              # time limits: 12 hours
#SBATCH --error=myJob.err            # standard error file
#SBATCH --account=CNHPC_1497299_0    # account name
#SBATCH --output=myJob.out           # standard output file``
#SBATCH --partition=dcgp_usr_prod    # partition name
#SBATCH --qos=normal                 # quality of service

source ~/.bashrc
module load python/3.10
module load gcc/12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gsl/2.7.1--gcc--12.2.0
module load openmpi/4.1.6--gcc--12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0


source /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/activate  

PROCESS_NAME="plus_minus"
CORES=4      # Number of cores pre simulation
PROCESSES=21 # Number of simulations


COUNTER_PATH="/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/counter.txt"
COUNTER=$(cat $COUNTER_PATH)


#run 21cmfast
COUNTER_RANGE=$(seq $COUNTER $(($COUNTER + $PROCESSES-1)))
for counter in $COUNTER_RANGE; do
    OMP_NUM_THREADS=$CORES \
    /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/save_ps.py \
    --counter $counter&
done

wait

