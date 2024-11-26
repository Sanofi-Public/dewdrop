#!/bin/bash

# List of Python scripts to run
run_ids=("8_22_run_1-v1" "8_22_run_2-v1" )

# Loop through the scripts
for script in "${run_ids[@]}"; do
    # Run the script in the background using nohup
    nohup python run_training.py -c "${script}_config.ini" > "model_logs/nvidia-poc/${script}.log" 2>&1 &

    # Get the process ID of the last background job
    last_pid=$!

    # Wait for the last background job to finish
    wait "$last_pid"

    echo "Run $script has finished."
done

echo "All runs have finished."
