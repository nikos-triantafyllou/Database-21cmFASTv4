#!/bin/bash -l
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=8          # tasks/processes/cores per node max 112 for DCGP and 32 for Booster
#SBATCH --cpus-per-task=1            # threads/cpus per task
#SBATCH --time=01:00:00              # time limits: 12 hours
#SBATCH --error=myJob.err            # standard error file
#SBATCH --account=CNHPC_1497299_0    # account name
#SBATCH --output=myJob.out           # standard output file``
#SBATCH --partition=dcgp_usr_prod    # partition name
#SBATCH --qos=normal                 # quality of service

FILE='plot_lightcones.py'

module load python/3.10
source /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/venv3/bin/activate
python /leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/plots/$FILE