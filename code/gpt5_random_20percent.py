import numpy as np
from sklearn.utils import resample
import pandas as pd
import json
import argparse 

import zero_shot
import eval_optimizer
import mas_ethics
import mas_metacognition
import mas_safety

def create_stratified_bootstrap_indicies(data, n_samples):

    bootstrap_indices = resample(data, replace=True, n_samples=n_samples, stratify=data['kind'], random_state=42)
    return bootstrap_indices.index.tolist()
    

def load_benchmark(name):
    benchmark_file_map = {
        'mmlu_ethics' : 'ethics/mmlu_ethics.json',
        'triage_ethics' : 'ethics/triage_ethics.json',
        'truthfulqa_ethics' : 'ethics/truthfulqa_ethics.json',
        'medbullets_metacognition' : 'metacognition/medbullets_metacognition.json',
        'medcalc_metacognition' : 'metacognition/medcalc_metacognition.json',
        'metamedqa_metacognition' : 'metacognition/metamedqa_metacognition.json',
        'mmlu_metacognition' : 'metacognition/mmlu_metacognition.json',
        'mmlu_pro_metacognition' : 'metacognition/mmlu_pro_metacognition.json',
        'pubmedqa_metacognition' : 'metacognition/pubmedqa_metacognition.json',
        'bbq_safety' : 'safety/bbq_safety_no_dups.json',
        'casehold_safety' : 'safety/casehold_safety.json',
        'mmlu_safety' : 'safety/mmlu_safety.json',
        'mmlupro_safety' : 'safety/mmlupro_safety.json'
    }

    df = pd.DataFrame(json.load(open(f"../benchmarks/{benchmark_file_map[name]}", 'r'))).set_index('id')
    benchmark_category = benchmark_file_map[name].split('/')[0]
    return df, benchmark_category

def run_gpt5(benchmark_name, custom_indices, output_path, model="gpt-5-2025-08-07"):

    zero_shot.main([benchmark_name, custom_indices, f"{output_path}/zero_shot", "zeroshot", model])
    eval_optimizer.main([benchmark_name, custom_indices, f"{output_path}/evaloptim", "evaluator-optimizer", model])
    if benchmark_category == 'ethics':
        mas_ethics.main([benchmark_name, custom_indices, f"{output_path}/mas", "mas-ethics", model])
    if benchmark_category == 'metacognition':
        mas_metacognition.main([benchmark_name, custom_indices, f"{output_path}/mas", "mas-metacognition", model])
    elif benchmark_category == 'safety':    
        mas_safety.main([benchmark_name, custom_indices, f"{output_path}/mas", "mas-safety", model])

def get_sample_indices(benchmark_name):
    idx_list = json.load(open('/mnt/bulk-sirius/sanddhya/agent_benchmarking/Agent_Edge_Benchmarking/code/analysis/gpt5_subsets_20percent.json','r'))
    return idx_list.get(benchmark_name, [])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run gpt5 on random 20 percent of samples for the chosen benchmark.')
    parser.add_argument('--benchmark', type=str, required=True, help='Name of benchmark dataset')
    parser.add_argument('--experiment_path', type=str, default='bootstrap', help='Prefix for output files.')
    args = parser.parse_args()  
    
    benchmark_df, benchmark_category = load_benchmark(args.benchmark)
    custom_indices = get_sample_indices(args.benchmark)
    run_gpt5(args.benchmark, custom_indices, args.experiment_path)

