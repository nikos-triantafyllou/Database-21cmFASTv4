#!/bin/bash

# Define the interval for checking job status (in seconds)
check_interval=5

while true; do
    # Run the qstat command to get the status of PBS jobs
    squeue -u ntriant1

    # Sleep for the specified interval before checking again
    sleep $check_interval
done

# qstat -u ntriantafyllou

# sleep $check_interval


# qstat -u ntriantafyllou

# sleep $check_interval



# qstat -u ntriantafyllou

# sleep $check_interval


# qstat -u ntriantafyllou

# sleep $check_interval