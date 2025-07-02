import pandas as pd
import re
import time
import json 
from openai import OpenAI
from tqdm import tqdm
import sys 
import datetime 
from dotenv import load_dotenv
import os

load_dotenv()

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

    benchmark_df = pd.DataFrame(json.load(open(f"../benchmarks/{benchmark_file_map[name]}", 'r'))).set_index('id')
    return benchmark_df 

def create_client(model_name='gpt-4o-2024-08-06'):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name=model_name)
    return client 

def run_query(client, input_text, log_file) -> str:

    answer = None
    confidence = None
    invalid = True
    num_attempts = 0

    while invalid and num_attempts < 5:
        num_attempts += 1
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        output = response.choices[0].message.content
        log_file.write(f"Attempt {num_attempts}: \n {output}\n")
        log_file.write(f"RESPONSE FORMAT: {str(invalid)}\n")
        log_file.write('-' * 20 + '\n\n')
        if response_format_check(output):
            invalid = False
    
    if not invalid:
        final_response = parse_response(output)
        log_file.write(f"Final Response: {final_response}\n")
        return final_response
    else:
        log_file.write(f"Final Response: NONE\n")
        return {"answer": None, "confidence": None}

def response_format_check(response) -> bool:
    pattern = re.compile(r'^\s*[0-9]\s*,\s*[1-5]\s*$')
    return pattern.match(response)

def parse_response(response) -> dict[str : int]:
    parts = [x.strip() for x in response.split(',')]
    answer, confidence = int(parts[0]), int(parts[1])
    return {"answer" : answer, "confidence" : confidence}

def run_benchmark(benchmark_df, experiment_path, client, custom_indices):

    results = []
    answered_idx_counts = {}
    for idx in tqdm(custom_indices, desc=f"Running experiment {experiment_path}"):
        row = benchmark_df.loc[idx]
        if id in answered_idx_counts:
            answered_idx_counts[id] += 1
        else:
            answered_idx_counts[id] = 1

        f = open(f"{experiment_path}/logs/{id}_{str(answered_idx_counts[id]) if answered_idx_counts[id] > 1 else ''}.txt", "w")
        id = row['id']
        question = row['question']
        choices = row['options']
        correct = int(row['target'])
        
        prompt = (
            f"Question: {question}\n"
            f"Choices: {choices}\n"
            "Please answer with only two values separated by a comma: "
            "'ChoiceNumber, ConfidenceScore' where ChoiceNumber is the index (zero-based) of your answer "
            "(e.g., 0, 1, 2, 3) and ConfidenceScore is an integer from 1 to 5."
        )
        f.write(f"Prompt: {prompt}\n")
        f.write('-' * 20 + '\n\n')
        response = run_query(client, prompt, f)

        f.close()
        
        if response.answer is not None:
            is_correct = int(response.answer == correct)
        
            results.append({
                    'id': id,
                    'question': question,
                    'model_answer': response.answer,
                    'confidence': response.confidence,
                    'correct_answer': correct,
                    'is_correct': is_correct
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{experiment_path}/RESPONSES.csv", index=False)
    return results_df

def generate_summary(results_df, experiment_path):

    total = len(results_df)
    corrects = results_df['is_correct'].sum()
    accuracy = (corrects / total) * 100 if total > 0 else 0
    acc_by_conf = results_df.groupby('confidence')['is_correct'].mean() * 100

    with open(f"{experiment_path}/SUMMARY.txt", 'w') as f:
        f.write(f"Total questions: {total}\n")
        f.write(f"Correct: {corrects}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        f.write("Accuracy by confidence level:\n")
        for conf, acc in acc_by_conf.items():
            if pd.notna(conf):
                f.write(f"  {int(conf)}: {acc:.2f}%\n")

def setup_experiment_directory(experiment_path, dataset_name, bootstrap_indices):
    if os.path.exists(experiment_path):
        # Directory (or file) already there → error out
        sys.exit(f"Error: '{experiment_path}' already exists. Aborting.")
    try:
        os.makedirs(experiment_path, exist_ok=False)
        os.makedirs(f'{experiment_path}/logs', exist_ok=False)
        with open(f"{experiment_path}/INFO.txt", "w") as info_file:
            info_file.write(datetime.now().isoformat())
            info_file.write(f"\nWorkflow: zero-shot\n")
            info_file.write(f"Mode: gpt-4o-2024-08-06\n")
            info_file.write(f"Dataset: {dataset_name}\n")
            info_file.write(f"Bootstrap Indices: {bootstrap_indices}\n")
            info_file.write(f"Experiment path: {experiment_path}\n")
    except Exception as e:
        sys.exit(f"Error creating '{experiment_path}': {e}")

def main(args):
    
    benchmark = args[0]
    bootstrap_indices = args[1]
    experiment_path = args[2]

    setup_experiment_directory(experiment_path) 
    client = create_client()
    run_benchmark(benchmark, client, bootstrap_indices)

if __name__ == "__main__":
    main(sys.argv[1:])

