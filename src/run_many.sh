#!/bin/bash -l

# Wrapper script to submit the same job script sequentially to SLURM

# Job script to be submitted multiple times (modify this to your actual script)
job_script="/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/CREATE_DATABASE/run_single.sh"

# Number of jobs to submit
num_jobs=1

# Initialize the first job ID to empty
prev_job_id=""

# Loop through the job submissions
for i in $(seq 1 $num_jobs); do
    if [ -z "$prev_job_id" ]; then
        # Submit the first job without any dependency
        job_id=$(sbatch --parsable "$job_script")
        echo "Submitted $job_script (Job $i) with Job ID: $job_id"
    else
        # Submit subsequent jobs with dependency on the previous job
        job_id=$(sbatch --parsable --dependency=afterany:$prev_job_id "$job_script")
        echo "Submitted $job_script (Job $i) with Job ID: $job_id (dependent on $prev_job_id)"
    fi
    # Update the previous job ID to the current one
    prev_job_id=$job_id

done

echo "All $num_jobs jobs submitted sequentially."
