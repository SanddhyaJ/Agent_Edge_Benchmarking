import pandas as pd
import re
import time
import json 
from openai import OpenAI
from tqdm import tqdm

def run_benchmark(dataset_path, output_prefix):
    print()
    #metacognition - pubmedqa_labeled
    output_path = f'/Users/sanddhyajayabalan/Desktop/Projects/Prj_MetaMedQA/experiments/RESPONSES_{output_prefix}.csv'     # File to store detailed results
    summary_path = f'/Users/sanddhyajayabalan/Desktop/Projects/Prj_MetaMedQA/experiments/SUMMARY_{output_prefix}.txt'    # File to store summary statistics
    model_name = 'llama-3.3-70b-instruct-q4km'                 # Ollama model name (replace as needed)

    ds = json.load(open(dataset_path, 'r'))
    df = pd.DataFrame(ds)
    pattern = re.compile(r'^\s*[0-9]\s*,\s*[1-5]\s*$')

    results = []
    #client = OpenAI(api_key=...)
    #client = OpenAI(api_key=...)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=output_prefix):
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
        
        answer = None
        confidence = None
        
        invalid = True
        num_attempts = 0

        while invalid and num_attempts < 5:

            num_attempts += 1
            response = client.chat.completions.create(
                messages=[
                    { "role": "user","content": prompt}
                ], model=model_name,
            )

            output = response.choices[0].message.content
            
            if pattern.match(output):
                # Parse "Letter, Number"
                parts = [x.strip() for x in output.split(',')]
                answer, confidence = int(parts[0]), int(parts[1])
                invalid = False
        
                is_correct = int(answer == correct)
        
                results.append({
                    'id': id,
                    'question': question,
                    'model_answer': answer,
                    'confidence': confidence,
                    'correct_answer': correct,
                    'is_correct': is_correct
                })


    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    total = len(results_df)
    corrects = results_df['is_correct'].sum()
    accuracy = (corrects / total) * 100 if total > 0 else 0

    acc_by_conf = results_df.groupby('confidence')['is_correct'].mean() * 100

    with open(summary_path, 'w') as f:
        f.write(f"Total questions: {total}\n")
        f.write(f"Correct: {corrects}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        f.write("Accuracy by confidence level:\n")
        for conf, acc in acc_by_conf.items():
            if pd.notna(conf):
                f.write(f"  {int(conf)}: {acc:.2f}%\n")

if __name__ == "__main__":

    shared_benchmark_path = '/Users/sanddhyajayabalan/Desktop/Projects/Prj_MetaMedQA/benchmarks/v2'
    
    """
    #ethics - triage
    dataset_path = f'{shared_benchmark_path}/ethics/triage.json'
    output_prefix = f'Triage_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #ethics - mmlu 
    dataset_path = f'{shared_benchmark_path}/ethics/mmlu_moral_scenarios_test.json'
    output_prefix = f'mmlu_moral_scenarios_Qwen3_zeroshot'  
    run_benchmark(dataset_path, output_prefix)
    #ethics - truthfulqa
    dataset_path = f'{shared_benchmark_path}/ethics/TruthfulQA_best.json'
    output_prefix = f'TruthfulQA_best_Qwen3_zeroshot'  
    run_benchmark(dataset_path, output_prefix)
    """
    """
    #metacognition - metamedqa
    dataset_path = f'{shared_benchmark_path}/metacognition/metamedqa.json'
    output_prefix = f'metamedqa_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #metacognition - mmlu
    dataset_path = f'{shared_benchmark_path}/metacognition/mmlu_metacognition_test.json'
    output_prefix = f'mmlu_metacognition_test_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #metacognition - pubmedqa_labeled
    dataset_path = f'{shared_benchmark_path}/metacognition/pubmedqa_labeled.json'
    output_prefix = f'pubmedqa_labeled_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #metacognition - medbullets 
    dataset_path = f'{shared_benchmark_path}/metacognition/medbullets_op5.json'
    output_prefix = f'medbullets_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    """
    #metacognition - medcalc_filtered
    dataset_path = f'{shared_benchmark_path}/metacognition/medcalc_filtered.json'
    output_prefix = f'medcalc_filtered_llama3pt3_zeroshot'
    run_benchmark(dataset_path, output_prefix)

    """
    #reg - bbq
    dataset_path = f'{shared_benchmark_path}/safety/bbq_subset.json'
    output_prefix = f'bbq_subset_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #reg - mmlu pro
    dataset_path = f'{shared_benchmark_path}/safety/mmlu_pro_regulatory_test.json'
    output_prefix = f'mmlu_pro_regulatory_test_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #reg - mmlu 
    dataset_path = f'{shared_benchmark_path}/safety/mmlu_professional_law_test.json'
    output_prefix = f'mmlu_professional_law_test_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    #reg - casehold (filtered)
    dataset_path = f'{shared_benchmark_path}/safety/casehold_filtered.json'
    output_prefix = f'casehold_filtered_Qwen3_zeroshot'
    run_benchmark(dataset_path, output_prefix)
    """

