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
    return bootstrap_indices.id.tolist()
    

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
        'bbq_safety' : 'safety/bbq_safety.json',
        'casehold_safety' : 'safety/casehold_safety.json',
        'mmlu_safety' : 'safety/mmlu_safety.json',
        'mmlupro_safety' : 'safety/mmlupro_safety.json'
    }

    df = pd.DataFrame(json.load(open(f"../benchmarks/{benchmark_file_map[name]}", 'r'))).set_index('id')
    benchmark_category = benchmark_file_map[name].split('/')[0]
    return df, benchmark_category

def run_bootstrap(benchmark_name, boostrap_indices, output_path):

    zero_shot.main(args.benchmark, bootstrap_indices)
    eval_optimizer.main(args.benchmark, bootstrap_indices)
    if benchmark_category == 'ethics':
        mas_ethics.main(args.benchmark, bootstrap_indices)
    elif benchmark_category == 'metacognition':
        mas_metacognition.main(args.benchmark, bootstrap_indices) 
    elif benchmark_category == 'safety':    
        mas_safety.main(args.benchmark, bootstrap_indices)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create bootstrap samples from the given data.')
    parser.add_argument('--benchmark', type=str, required=True, help='Name of benchmark dataset')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of bootstrap samples to generate.')
    parser.add_argument('--experiment_path', type=str, default='bootstrap', help='Prefix for output files.')
    args = parser.parse_args()  

    benchmark_df, benchmark_category = load_benchmark(args.benchmark)
    bootstrap_indices = create_stratified_bootstrap_indicies(benchmark_df, args.n_samples)
    run_bootstrap(args.benchmark, bootstrap_indices)


