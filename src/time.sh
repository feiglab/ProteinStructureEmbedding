#!/bin/bash

runs=10
total_wall_time=0
total_user_time=0
total_memory=0

for ((i=1; i<=runs; i++)); do
  # Run the command and capture statistics
  output=$( /usr/bin/time -f "%e %U %M" ./predict.py /feig/s1/spencer/gnn/cases/time_test/500.pdb 2>&1 > /dev/null )
  
  # Extract values (Wall Clock Time, User CPU Time, Peak Memory Usage)
  wall_time=$(echo $output | awk '{print $1}')  # Seconds
  user_time=$(echo $output | awk '{print $2}')  # Seconds
  memory_kb=$(echo $output | awk '{print $3}')  # Peak memory in KB
  
  # Convert memory to GB
  memory_gb=$(echo "$memory_kb / 1024 / 1024" | bc -l)
  
  # Print metrics for this run
  echo "Run $i: Wall Clock Time=${wall_time}s, User CPU Time=${user_time}s, Peak Memory=${memory_gb}GB"
  
  # Accumulate totals
  total_wall_time=$(echo "$total_wall_time + $wall_time" | bc)
  total_user_time=$(echo "$total_user_time + $user_time" | bc)
  total_memory=$(echo "$total_memory + $memory_gb" | bc)
done

# Calculate averages
avg_wall_time=$(echo "$total_wall_time / $runs" | bc -l)
avg_user_time=$(echo "$total_user_time / $runs" | bc -l)
avg_memory=$(echo "$total_memory / $runs" | bc -l)

# Display averages
echo "Average Wall Clock Time: ${avg_wall_time}s"
echo "Average User CPU Time: ${avg_user_time}s"
echo "Average Peak Memory Usage: ${avg_memory}GB"

