<div align="center">

# 🩺 GraphMed-LT

**Patient-Specific Graph Memory with Latent Clinical Thought Refinement for Multi-Turn Medical Conversations**

[![arXiv](https://img.shields.io/badge/arXiv-2510.03536-b31b1b.svg)](https://arxiv.org/abs/2510.03536)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📰 News

- **[2026-08]** 🎉 GraphMed-LT is accepted to **EMNLP 2026**!

## 📖 Overview

We propose GraphMed-LT, a patient-specific graph memory approach with latent clinical thought refinement for multi-turn medical conversations. GraphMed-LT extracts patient-specific clinical triplets from patient responses, retrieves relevant knowledge triplets, and organises them into an incrementally updated graph memory. The graph memory is projected into graph-conditioned evidence tokens and refined inside a trainable doctor agent through hidden-state feedback, enabling the agent to update its internal clinical context before asking follow-up questions or producing the final answer.

<p align="center">
  <img src="image/GraphMed-LT.png" alt="GraphMed-LT framework" width="100%">
</p>

## ✨ Highlights

- **Patient-specific graph memory** built from clinical triplets and incrementally updated across dialogue turns
- **Source-aware edge types** that distinguish patient-observed triplets from retrieved background knowledge
- **Latent clinical thought refinement**: graph-conditioned evidence tokens are refined inside a trainable doctor agent through hidden-state feedback
- **Multi-turn clinical reasoning**: the doctor agent updates its internal clinical context before asking follow-up questions or producing the final answer

## 🧩 Method at a Glance

GraphMed-LT consists of four components:

| Component | Responsibility |
| --- | --- |
| `Patient Agent` | Returns responses grounded in the complete patient record |
| `Triplet Agent` | Extracts patient-specific triplets from patient responses and retrieves the top-3 relevant knowledge triplets from an external triplet corpus |
| `Graph Memory` | Initialised as `G_0` from triplets extracted from the initial patient description `p_0`, and updated across turns by adding clinical entities and relation-labelled edges; source-aware edge types separate patient-observed evidence from retrieved background knowledge |
| `Doctor Agent (trainable)` | Receives the graph memory encoded with a GAT and projected into graph-conditioned evidence tokens, and refines its latent clinical context through latent clinical thought refinement before producing follow-up questions or the final answer |

## ⚙️ Installation

```bash
conda env create -f environment.yml
conda activate GraphMed-LT
```

## 🏋️ Training

The main training entry point is `projection_train.py`.

```bash
python projection_train.py \
  --train_file data/all_train_convo.jsonl \
  --expert_model Qwen/Qwen2.5-72B-Instruct \
  --triplet_model Qwen/Qwen2.5-72B-Instruct \
  --distributed_backend fsdp \
  --triplet_corpus path/to/primekg_triplets.jsonl \
  --retrieval_top_k 3 \
  --prefix_len 20 \
  --refinement_steps 5 \
  --gnn_model gat \
  --gnn_in_dim 256 \
  --gnn_hidden_dim 256 \
  --gat_heads 4 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --batch_size 64 \
  --epochs 5
```

> **Notes**
> - The code expects the external triplet corpus to be provided locally and does not include PrimeKG triples in this repository.
> - The BiCA encoder defaults to `bisectgroup/BiCA-base` and can be changed with `GRAPHMED_BICA_MODEL`.

## 📊 Benchmark

```bash
python GraphMedLT_benchmark.py \
  --expert_module expert --expert_class ScaleExpert \
  --expert_model save_model/doctor_agent \
  --patient_module patient --patient_class FactSelectPatient \
  --data_dir data --dev_filename all_dev_good.jsonl \
  --projection_ckpt save_model/graphmed_lt.ckpt \
  --triplet_corpus path/to/primekg_triplets.jsonl \
  --output_filename results/graphmed_lt_dev.jsonl \
  --max_questions 10
```

## 📎 Citation

If you find GraphMed-LT useful in your research, please give this repository a ⭐ and cite our paper:

```bibtex
@article{meng2026graphmed,
  title={GraphMed-LT: Patient-Specific Graph Memory with Latent Clinical Thought Refinement for Multi-Turn Medical Conversations},
  author={Meng, Zhaohan and Meng, Zaiqiao and Liu, Siwei and Xu, Hao and Yuan, Ke and Ounis, Iadh},
  journal={arXiv preprint arXiv:2510.03536},
  year={2026}
}
```

*The BibTeX will be updated to the EMNLP 2026 version once the proceedings are published.*

## 📄 License

This repository is released under the MIT License.
