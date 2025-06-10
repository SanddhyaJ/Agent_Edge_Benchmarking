#!/bin/bash

#SBATCH --job-name=METACOGNITION_mas_GPT4o
#SBATCH --partition=gpu,cpu
#SBATCH --time=24:00:00   # Set the maximum run time (e.g. 60 minutes), after this your process will be killed, maximum 7 days
#SBATCH --mem=4G         # How much RAM do you need?
#SBATCH --cpus-per-task 1 # Number of CPU cores (I think this is the number of ht-threads, so you can select number of physical cores * 2)
#SBATCH --mail-type=all # to receive E-Mail notifications
#SBATCH --mail-user=sanddhya.jayabalan@tu-dresden.de # Set E-Mail address to receive notifications


for f in $(ls ../../benchmarks/metacognition/*); do
    echo "Running benchmark for dataset: $f"
    name="$(basename "$f" .json)"
    python ../mas_metacognition.py gpt-4o-2024-08-06 $name 1.0 mas_metacognition_gpt4o ../../out
done
