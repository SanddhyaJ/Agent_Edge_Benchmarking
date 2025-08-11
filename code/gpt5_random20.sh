#!/bin/bash

OUT_PATH=/mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/code/gpt5

python gpt5_random_20percent.py --benchmark triage_ethics --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark mmlu_ethics --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark truthfulqa_ethics --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark medbullets_metacognition --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark medcalc_metacognition --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark metamedqa_metacognition --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark mmlu_metacognition --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark pubmedqa_metacognition --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark bbq_safety --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark casehold_safety --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark mmlu_safety --experiment_path $OUT_PATH
python gpt5_random_20percent.py --benchmark mmlupro_safety --experiment_path $OUT_PATH
