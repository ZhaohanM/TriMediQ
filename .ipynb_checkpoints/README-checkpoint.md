# TriMediQ: A Triplet-Structured Approach for Interactive Medical Question Answering

## 🚀 Overview
TriMediQ is a framework designed for interactive medical QA.  
It introduces a triplet-structured knowledge incorporation module to enhance reasoning in large language models (LLMs).  

The framework consists of two stages:
1. **Projection Training**: Fine-tuning a projection module that encodes UMLS-style triplets via a graph encoder + projector and injects them into a frozen expert LLM (open source) through prefix tuning.
2. **Interactive QA**: Using the trained projection to support inference in multi-turn patient–expert interactions.

## 🧩 Framework
![TriMediQ Framework](image/Medical_QA.pdf)

## 🔧 Installation

Create a new conda environment with all dependencies (requires GPU for PyTorch + CUDA):
```bash
conda env create -f environment.yml
conda activate TriMediQ
```

## 📂 Project Structure
- `projection_train.py`: Training the projection module (graph encoder + projector).
- `TriMediQ_benchmark.py`: Main script for running benchmarks.
- `patient.py`: Defines the `Patient` class simulating patient behaviour.
- `expert.py`: Defines the `Expert` classes for different expert strategies.
- `args.py`: Handles command-line arguments and configuration.
- `utils/`: Helper modules for triplet extraction, graph construction, and model handling.
- `data/`: Development and training datasets in JSONL format.


## ▶️ Running the Benchmark
Example run:
```bash
python TriMediQ_benchmark.py \
  --expert_module expert --expert_class ScaleExpert \
  --patient_module patient --patient_class FactSelectPatient \
  --data_dir ../data --dev_filename all_dev_good.jsonl \
  --output_filename out.jsonl --max_questions 10
```

## 📜 License
This repository is released under the MIT License.
