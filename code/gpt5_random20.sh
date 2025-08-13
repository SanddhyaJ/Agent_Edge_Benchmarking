#!/bin/bash

#SBATCH --job-name=edge_bench_gpt5
#SBATCH --partition=gpu,cpu
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=all
#SBATCH --mail-user=sanddhya.jayabalan@tu-dresden.de

OUT_PATH=/mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/code/gpt5

#python gpt5_random_20percent.py --benchmark triage_ethics --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark mmlu_ethics --experiment_path $OUT_PATH/mmlu_ethics
python gpt5_random_20percent.py --benchmark truthfulqa_ethics --experiment_path $OUT_PATH/truthfulqa_ethics
python gpt5_random_20percent.py --benchmark medbullets_metacognition --experiment_path $OUT_PATH/medbullets_metacognition
python gpt5_random_20percent.py --benchmark medcalc_metacognition --experiment_path $OUT_PATH/medcalc_metacognition
python gpt5_random_20percent.py --benchmark metamedqa_metacognition --experiment_path $OUT_PATH/metamedqa_metacognition
python gpt5_random_20percent.py --benchmark mmlu_metacognition --experiment_path $OUT_PATH/mmlu_metacognition
python gpt5_random_20percent.py --benchmark pubmedqa_metacognition --experiment_path $OUT_PATH/pubmedqa_metacognition
python gpt5_random_20percent.py --benchmark bbq_safety --experiment_path $OUT_PATH/bbq_safety
python gpt5_random_20percent.py --benchmark casehold_safety --experiment_path $OUT_PATH/casehold_safety
python gpt5_random_20percent.py --benchmark mmlu_safety --experiment_path $OUT_PATH/mmlu_safety
python gpt5_random_20percent.py --benchmark mmlupro_safety --experiment_path $OUT_PATH/mmlupro_safety
