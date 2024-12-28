#!/bin/bash -l
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=9          # tasks/processes/cores per node max 112 for GCGP and 32 for Booster
#SBATCH --cpus-per-task=1            # threads/cpus per task
#SBATCH --time=12:00:00              # time limits: 12 hours
#SBATCH --error=myJob.err            # standard error file
#SBATCH --account=CNHPC_1497299_0    # account name
#SBATCH --output=myJob.out           # standard output file``
#SBATCH --partition=dcgp_usr_prod   # partition name
#SBATCH --qos=normal                 # quality of service

source ~/.bashrc
module load python/3.11
module load gcc/12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gsl/2.7.1--gcc--12.2.0
module load openmpi/4.1.6--gcc--12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0


# Things I tried - you can skip this comment section
# source /cluster/shared/software/miniconda3/bin/activate 21cm-dev4
# module load anaconda3

# source activate database1

# Activate conda environment (this might raise an errror that conda module doesn't exist)
source /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3116/bin/activate  

# # To use conda commands, after activation, I have to unload and then reload (hacky version)
# module unload anaconda3/2023.09-0
# module load anaconda3/2023.09-0
# conda list 


# But to use the packages inside the activated conda environment (e.g. python) I have to activate it and leave it there
# source /leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/anaconda3-2023.09-0-zcre7pfofz45c3btxpdk5zvcicdq5evx/bin/deactivate database1
# source /leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/anaconda3-2023.09-0-zcre7pfofz45c3btxpdk5zvcicdq5evx/bin/activate database1

# export PYTHONHOME=/leonardo/home/userexternal/ntriant1/projects/database/database2_venv
# PROCESS_NAME="test2"
# CORES=1
# PROCESSES=1

which python
python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/run_single.py&
psrecord $! --plot  /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/psrecord_output.png --log /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/psrecord_log.txt --include-children&

wait



# PROCESS_NAME="plus_minus"
# CORES=9      # Number of cores pre simulation
# PROCESSES=1 # Number of simulations


# COUNTER_PATH="/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/counter.txt"
# COUNTER=0


# #run 21cmfast
# COUNTER_RANGE=$(seq $COUNTER $(($COUNTER + $PROCESSES-1)))
# for counter in $COUNTER_RANGE; do
#     OMP_NUM_THREADS=$CORES \
#     /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/run_single.py \
#     --counter $counter&
#     psrecord $! --plot /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/psrecord4/psrecord_output_${counter}.png \
#     --log /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/psrecord4/psrecord_log.txt --include-children&
# done

# wait