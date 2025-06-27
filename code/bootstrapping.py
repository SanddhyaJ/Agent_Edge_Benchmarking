import subprocess
import json
import random
import os
from pathlib import Path
from datetime import datetime

# Configuration
DATASET_PATH = "datasets/mydata.json"
WORKFLOWS = ["zero_shot.py", "agent_system.py"]  # Add script paths here
BOOTSTRAP_ITERATIONS = 100
SAMPLE_SIZE = None  # Set to None to use full dataset size
OUTPUT_DIR = "bootstrap_runs"

def load_dataset(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def create_bootstrap_sample(dataset, sample_size):
    return random.choices(dataset, k=sample_size)

def run_script(script_path, input_path, output_path):
    subprocess.run([
        "python", script_path,
        "--dataset", input_path,
        "--output", output_path
    ], check=True)

def main():
    dataset = load_dataset(DATASET_PATH)
    sample_size = SAMPLE_SIZE or len(dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(OUTPUT_DIR) / f"bootstrap_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for script in WORKFLOWS:
        script_name = Path(script).stem
        script_output_dir = base_output_dir / script_name
        script_output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(BOOTSTRAP_ITERATIONS):
            sample = create_bootstrap_sample(dataset, sample_size)
            sample_file = script_output_dir / f"sample_{i}.json"
            output_file = script_output_dir / f"output_{i}.json"

            save_json(sample, sample_file)
            print(f"Running {script_name} on bootstrap sample {i+1}/{BOOTSTRAP_ITERATIONS}")
            run_script(script, str(sample_file), str(output_file))

if __name__ == "__main__":
    main()











import subprocess
import json
import random
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# === CONFIGURATION ===
DATASET_PATH = "datasets/mydata.json"
WORKFLOW_SCRIPTS = {
    "zero_shot": "zero_shot.py",
    "agent_system": "agent_system.py",
    "cot": "cot.py"
}
BOOTSTRAP_ITERATIONS = 300
SAMPLE_SIZE = 1000
BASE_OUTPUT_DIR = "paired_bootstrap"
LOGS_DIR = "logs"

# === UTILITIES ===
def load_dataset(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def run_method_with_logs(script_path, sample_path, output_path, method_name, sample_index, question_id_map):
    log_dir = Path(LOGS_DIR) / method_name / f"bootstrap_{sample_index}"
    ensure_dir(log_dir)

    # Pass log_dir as an environment variable or argument if your scripts can use it
    env = os.environ.copy()
    env["LOG_DIR"] = str(log_dir)

    # Save per-question logs
    with open(sample_path) as f:
        sample = json.load(f)

    # Track counts of repeated questions
    qid_counts = defaultdict(int)

    for q in sample:
        qid = str(q.get("id", "noid"))
        qid_counts[qid] += 1
        suffix = "" if qid_counts[qid] == 1 else chr(ord("A") + qid_counts[qid] - 2)
        filename = f"{qid}{'_' + suffix if suffix else ''}.log"
        log_path = log_dir / filename

        # Optional: Save question to log file
        with open(log_path, "w") as log_file:
            log_file.write(f"# Method: {method_name}\n")
            log_file.write(f"# Bootstrap Iteration: {sample_index}\n")
            log_file.write(json.dumps(q, indent=2) + "\n")

    # Run the method
    with open(output_path, "w") as out_file, open(log_dir / f"run_stdout.log", "w") as stdout_log:
        subprocess.run(
            ["python", script_path, "--dataset", sample_path, "--output", output_path],
            stdout=stdout_log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True
        )

# === MAIN BOOTSTRAP LOGIC ===
def paired_bootstrap():
    dataset = load_dataset(DATASET_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(BASE_OUTPUT_DIR) / f"bootstrap_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for i in range(BOOTSTRAP_ITERATIONS):
        sample = random.choices(dataset, k=SAMPLE_SIZE)
        sample_file = run_dir / f"sample_{i}.json"
        save_json(sample, sample_file)

        print(f"\n[Bootstrap {i+1}/{BOOTSTRAP_ITERATIONS}]")

        # Track how often each question ID appears (to suffix _A, _B, etc.)
        qid_map = defaultdict(int)
        for q in sample:
            qid = str(q.get("id", "noid"))
            qid_map[qid] += 1

        for method_name, script_path in WORKFLOW_SCRIPTS.items():
            output_dir = run_dir / method_name
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"output_{i}.json"

            print(f"  → Running {method_name}")
            run_method_with_logs(
                script_path,
                str(sample_file),
                str(output_file),
                method_name,
                i,
                qid_map
            )

if __name__ == "__main__":
    paired_bootstrap()
