#!/bin/bash

python mas_v2.py mmlu_safety /mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/out/gpt4o/mas_v2/mmlu_safety mas_v2 gpt4o
python mas_v2.py bbq_safety /mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/out/gpt4o/mas_v2/bbq_safety
python mas_v2.py casehold_safety /mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/out/gpt4o/mas_v2/casehold_safety mas_v2 gpt4o
python mas_v2.py mmlupro_safety /mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/out/gpt4o/mas_v2/mmlupro_safety mas_v2 gpt4o

