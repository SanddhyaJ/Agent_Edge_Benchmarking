import numpy as np
from sklearn.utils import resample
import pandas as pd
import json
import argparse 

#import zeroshot
import eval_optimizer
import mas_ethics
import mas_metacognition
import mas_safety

def create_stratified_bootstrap_indicies(data, n_samples):
    """
    Create stratified bootstrap indices for the given data.

    Parameters:
    - data: The input data to bootstrap.
    - n_samples: The number of bootstrap samples to generate.

    Returns:
    - A list of tuples, each containing the indices for a bootstrap sample.
    """

    unique_classes = np.unique(data['kind'])
    indices = []

    for _ in range(n_samples):
        sample_indices = []
        for cls in unique_classes:
            cls_indices = np.where(data['kind'] == cls)[0]
            sampled_indices = resample(cls_indices, replace=True)
            sample_indices.extend(sampled_indices)
        indices.append(tuple(sample_indices))

    return indices

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

    df = pd.DataFrame(json.load(open(benchmark_file_map[name], 'r'))).set_index('id')
    benchmark_category = benchmark_file_map[name].split('/')[0]
    return df, benchmark_category

def run_bootstrap():
    return 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create bootstrap samples from the given data.')
    parser.add_argument('--benchmark', type=str, required=True, help='Name of benchmark dataset')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of bootstrap samples to generate.')
    args = parser.parse_args()  

    bootstrap_indices = create_stratified_bootstrap_indicies(load_benchmark(args.benchmark), args.n_samples)


    #run on zeroshot
    #run on evaloptim
    #run on mas 




