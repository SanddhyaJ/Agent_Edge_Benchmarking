from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import pandas as pd
import re
import json 
from tqdm import tqdm
from langchain_core.runnables.config import RunnableConfig
import sys 
from sklearn.model_selection import train_test_split
import os
from IPython.display import Image
from datetime import datetime
from io import TextIOWrapper
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    id: str
    response: int
    confidence: int
    prompt: str
    question: str
    choices: list[str]
    feedback: list[str]
    valid_or_not: str
    log_file: TextIOWrapper
    num_iters: int

class Feedback(BaseModel):
    grade: Literal["valid", "not valid"] = Field(
        description="Decide if the response chosen is valid for the provided question/choices or not.",
    )
    feedback: str = Field(
        description="If the response is not valid, provide few concise sentences of feedback (single paragraph) on how to modify the answer and crucial informaiton to consider when answering the question.",
    )

def check_response_format(response: str, num_choices: int) -> bool:
    """
    Check if the response is in the format "ChoiceNumber, ConfidenceScore"
    where ChoiceNumber is an integer and ConfidenceScore is an integer from 1 to 5.
    """
    last_choice_index = num_choices - 1
    regex_string = rf'^\s*[0-{last_choice_index:d}]\s*,\s*[1-5]\s*$'
    pattern = re.compile(regex_string)
    return bool(pattern.match(response))

# Nodes
def llm_call_generator(state: State):
    """LLM generates a response"""
    log_file = state.get("log_file")
    if state.get("feedback"):
        input = "Consider these feedback(s) when answering the following question. You do not have to agree with the feedback but you must use it in your reasoning when answering the question:\n\nFEEDBACK: \n"
        for val in state['feedback']:
            input += val + '\n'
        input += f"QUESTION:\n\n{state['prompt']}"
    else:
        input = f"{state['prompt']}"
    msg = llm.invoke(input)
    is_valid = check_response_format(msg.content, len(state['choices']))

    llm_name = "llm_call_generator"
    time = datetime.now().isoformat()
    output = f"{msg.content}"
    log_string = f"{time}|\n{llm_name}|\n{input}|\n{output}\n"
    log_file.write(log_string)
    log_file.write("******************************\n\n")

    num_attempts = 0
    while not is_valid and num_attempts < 5:
        num_attempts += 1
        msg = llm.invoke(input)
        is_valid = check_response_format(msg.content, len(state['choices']))
        time = datetime.now().isoformat()
        output = f"{msg.content}"
        log_string = f"{time}|\n{llm_name}|\n{input}|\n{output}\n"
        log_file.write(log_string)
        log_file.write("******************************\n\n")
    if is_valid:
        response = int(msg.content.split(',')[0].replace(' ','')) 
        confidence = int(msg.content.split(',')[1].replace(' ','')) 
    else:
        response = None
        confidence = None
    return {"response": response, "confidence" : confidence, 'num_iters': state.get("num_iters") + 1}

def llm_call_evaluator(state: State):
    """LLM evaluates the response"""

    grading_prompt = f"QUESTION:\n{state['question']}"
    i = 0
    for val in state['choices']:
        grading_prompt += f"{i}) {val}\n"
        i += 1
    grading_prompt += f"\nRESPONSE:\n{state['response']}\n\n"
    grading_prompt += "Grade the response as 'valid' or 'not valid' based on whether the response is appropriate for the question and choices provided. "
    grading_prompt += "If the response is not valid, provide concise few sentences of feedback (single paragraph) on how to modify the answer and crucial information to consider when answering the question."
    grading_prompt += "If you have provided any previous feedback (listed below), consider it in your evaluation and do not contradict yourself.\n"
    prev_feedback = state.get('feedback', [])
    grading_prompt += "PREVIOUS FEEDBACK:\n"
    for val in prev_feedback:
        grading_prompt += f"- {val}\n"

    grade = evaluator.invoke(grading_prompt)

    log_file = state.get("log_file")
    input = grading_prompt
    llm_name = "llm_call_evaluator"
    time = datetime.now().isoformat()
    output = f"Grade: {grade.grade}, Feedback: {grade.feedback}"
    log_string = f"{time}|\n{llm_name}|\n{input}|\n{output}\n"
    log_file.write(log_string)
    log_file.write("******************************\n\n")

    return {"valid_or_not": grade.grade, "feedback": state.get('feedback') + [grade.feedback]}

def route_response(state: State):
    """Route back to response generator or end based upon feedback from the evaluator"""

    if state["valid_or_not"] == "valid":
        return "Accepted"
    elif state["valid_or_not"] == "not valid":
        if state['num_iters'] < 5:
            return "Rejected + Feedback"
        else:
            state['answer'] = None
            state['confidence'] = None
            return "Accepted"

def sample_dataset(df, sample_size=0.1):
    _, sample = train_test_split(
        df,
        test_size=sample_size,
        stratify=df['kind'],
        random_state=42
    )
    return sample

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

def run_benchmark(benchmark_df, experiment_path, custom_indices):

    #ds = json.load(open(f"{dataset_path}/{dataset_paths[dataset]}", 'r'))
    #df = pd.DataFrame(ds)
    #if percent_sample < 1:
    #    df_subset = sample_dataset(df, sample_size=percent_sample)
    #else:
    #    df_subset = df
    
    results = []
    answered_idx_counts = {}
    for idx in tqdm(custom_indices, desc=f"Running experiment {experiment_path}"):
        row = benchmark_df.loc[idx]
        if idx in answered_idx_counts:
            answered_idx_counts[idx] += 1
        else:
            answered_idx_counts[idx] = 1

        f = open(f"{experiment_path}/logs/{idx}_{str(answered_idx_counts[idx]) if answered_idx_counts[idx] > 1 else ''}.txt", "w")
        f.write("TIME | LLM | INPUT | OUTPUT\n")
        f.write("******************************\n\n")
        id = row.index
        question = row['question']
        choices = row['options']
        correct = row['target']

        choice_string = ""
        i = 0
        i_string = ""
        for val in choices:
            choice_string += f"{i}) {val}\n"
            i_string += f", {i}"
            i += 1

        prompt = (
            f"Question: {question}\n"
            f"Choices:\n{choice_string}\n"
            "Please answer with only two values separated by a comma: "
            "'ChoiceNumber, ConfidenceScore' where ChoiceNumber is the index (zero-based) of your answer "
            f"(e.g.{i_string}) and ConfidenceScore is an integer from 1 to 5."
        )

        state = optimizer_workflow.invoke({"question": question, "prompt" : prompt,
                                   "choices": choices, 
                                   "log_file": f, "id":id, "num_iters" : 0, 'feedback' : []}, RunnableConfig(recursion_limit=100))
        answer = state["response"]
        confidence = state["confidence"]

        if answer is not None and confidence is not None:
        
            is_correct = int(answer == correct)
        
            results.append({
                    'id': id,
                    'model_answer': answer,
                    'confidence': confidence,
                    'correct_answer': correct,
                    'is_correct': is_correct
                    })
            
        f.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{experiment_path}/RESPONSES.csv', index=False)
    return results_df

def setup_experiment_directory(experiment_path, dataset_name, bootstrap_indices, workflow, model = 'gpt-4o-2024-08-06'):
    if os.path.exists(experiment_path):
        # Directory (or file) already there → error out
        sys.exit(f"Error: '{experiment_path}' already exists. Aborting.")
    try:
        os.makedirs(experiment_path, exist_ok=False)
        os.makedirs(f'{experiment_path}/logs', exist_ok=False)
        with open(f"{experiment_path}/INFO.txt", "w") as info_file:
            info_file.write(datetime.datetime.now().isoformat())
            info_file.write(f"\nWorkflow: {workflow}\n")
            info_file.write(f"Model: {model}\n")
            info_file.write(f"Dataset: {dataset_name}\n")
            info_file.write(f"Bootstrap Indices: {bootstrap_indices}\n")
            info_file.write(f"Experiment Path: {experiment_path}\n")
    except Exception as e:
        sys.exit(f"Error creating '{experiment_path}': {e}")

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

def generate_workflow(llm, evaluator):
    optimizer_builder = StateGraph(State)
    optimizer_builder.add_node("llm_call_generator", llm_call_generator(llm=llm))
    optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator(evaluator=evaluator))
    optimizer_builder.add_edge(START, "llm_call_generator")
    optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
    optimizer_builder.add_conditional_edges(
        "llm_call_evaluator",
        route_response,
        {  
            "Accepted": END,
            "Rejected + Feedback": "llm_call_generator",
        },
    )
    return optimizer_builder

def main(args):
    model_name = 'gpt-4o-2024-08-06'
    benchmark = args[0]
    bootstrap_indices = args[1]
    experiment_path = args[2]
    workflow = args[3]

    setup_experiment_directory(experiment_path, benchmark, bootstrap_indices, workflow, model=model_name)
    llm = ChatOpenAI(
        model_name=model_name,
        api_key=os.getenv("EKFZ_OPENAI_API_KEY"),
    )
    evaluator = llm.with_structured_output(Feedback)
    optimizer_workflow = generate_workflow().compile()
    img = Image(optimizer_workflow.get_graph().draw_mermaid_png())
    with open(f"{experiment_path}/WORKFLOW.png", "wb") as f:
        f.write(img.data)

    results = run_benchmark(
        benchmark_df=load_benchmark(benchmark), 
        experiment_path=experiment_path, 
        custom_indices=bootstrap_indices,
    )

    generate_summary(results_df=results,
                     experiment_path = experiment_path)

if __name__ == "__main__":
    main(sys.argv[1:])


