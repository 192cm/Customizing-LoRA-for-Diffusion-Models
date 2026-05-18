# Customizing LoRA for Diffusion Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1.2%20%2B%20CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Diffusers-0.26.0-yellow" alt="Diffusers">
  <img src="https://img.shields.io/badge/LoRA-Stable%20Diffusion%20v1.5-7C3AED" alt="LoRA">
</p>

<p align="center">
  Fine-tune Stable Diffusion v1.5 with LoRA adapters on small custom image datasets.<br>
  Compare how dataset source and LoRA hyperparameters affect generated image quality.
</p>

---

## Results

### Dataset Samples

![Dataset samples](report_assets/figure3_dataset_samples.png)

### Before / After LoRA Fine-tuning

| Web-crawled | Real Photos | AI-generated |
|:-----------:|:-----------:|:------------:|
| ![Web crawled](report_assets/web_crawled_before_after_labeled.png) | ![Real](report_assets/real_before_after_labeled.png) | ![Generated](report_assets/generated_before_after_labeled.png) |

### Ablation Study

| LoRA Rank | LoRA Alpha |
|:---------:|:----------:|
| ![Rank comparison](report_assets/rank_comparison_labeled.png) | ![Alpha comparison](report_assets/alpha_comparison_labeled.png) |

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Dataset Format](#dataset-format)
- [Training Configuration](#training-configuration)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Notebook Guide](#notebook-guide)
- [Tips](#tips)

---

## Features

- Fine-tunes LoRA adapters on Stable Diffusion v1.5 using three types of custom datasets
- Compares **web-crawled images**, **real photographs**, and **AI-generated images** as training sources
- Uses Hugging Face `imagefolder` format with captions stored in `metadata.csv`
- Generates side-by-side before/after image grids for visual evaluation
- Runs ablation experiments over LoRA `rank` and `alpha` hyperparameters
- Exports labeled comparison images and summary CSV files for reporting

---

## Quick Start

**1. Create the conda environment**

```bash
conda env create -f environment.yml
conda activate genai-assignment2
python -m ipykernel install --user --name genai-assignment2 --display-name "genai-assignment2"
```

**2. Run the notebooks in order**

```
00 → 01 → 02 (rank & alpha) → 03 → 04
```

See [How to Run](#how-to-run) for details on each notebook.

---

## Repository Structure

```
.
├── 00_Customizing_LoRA.ipynb       # End-to-end single LoRA workflow
├── 01_dataset_training.ipynb       # Train on each dataset, generate comparisons
├── 02_ablation_rank.ipynb          # Rank ablation experiment
├── 02_ablation_alpha.ipynb         # Alpha ablation experiment
├── 03_test.ipynb                   # Additional inference and checkpoint testing
├── 04_monitor.ipynb                # Artifact and progress monitoring
├── experiment_utils.py             # Shared config and utilities
├── environment.yml
├── data/
│   ├── web_crawled_custom_dataset/train/
│   ├── real_custom_dataset/train/
│   └── generated_custom_dataset/train/
├── lora_experiments/
│   ├── dataset_training/
│   ├── comparisons/
│   ├── ablation/
│   ├── experiment_summary_rank.csv
│   └── experiment_summary_alpha.csv
├── report_assets/
└── sd_lora/
    ├── pytorch_lora_weights.safetensors
    ├── loss_log.csv
    └── checkpoint-*/
```

---

## Dataset Format

Each dataset follows the Hugging Face `imagefolder` layout:

```
data/{dataset_name}/
└── train/
    ├── metadata.csv
    ├── image_01.jpg
    └── image_02.png
```

`metadata.csv` schema:

```csv
file_name,caption
image_01.jpg,a sks building in pixel art style
```

Datasets used in this project:

| Name | Folder | Images | Base Prompt | Style Token |
|------|--------|-------:|-------------|-------------|
| `web_crawled` | `./data/web_crawled_custom_dataset` | 50 | `a building` | `sks` |
| `real` | `./data/real_custom_dataset` | 10 | `a city street in winter` | `sks` |
| `generated` | `./data/generated_custom_dataset` | 20 | `a house in a flower field` | `sks` |

> Dataset paths and prompts are defined in `experiment_utils.py`.

---

## Training Configuration

All training and inference settings are centralized in `experiment_utils.py`.

| Setting | Value |
|---------|-------|
| Base model | `runwayml/stable-diffusion-v1-5` |
| VAE | `stabilityai/sd-vae-ft-mse` |
| Seed | `2015` |
| Resolution | `512` |
| Batch size | `8` |
| Max train steps | `2000` |
| Checkpoint interval | `500` |
| Learning rate | `1e-4` |
| Default LoRA rank | `4` |
| Default LoRA alpha | `4` |
| LoRA target modules | `to_k`, `to_q`, `to_v`, `to_out.0` |

Trained weights are saved as `pytorch_lora_weights.safetensors` in each experiment directory.

---

## How to Run

Run notebooks in the order listed below.

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `00_Customizing_LoRA.ipynb` | Validate end-to-end workflow: data prep, captioning, metadata, training, inference |
| 2 | `01_dataset_training.ipynb` | Train `rank4_alpha4` LoRA on all three datasets; generate before/after grids |
| 3 | `02_ablation_rank.ipynb` | Compare rank `8` vs `16` on the web-crawled dataset |
| 4 | `02_ablation_alpha.ipynb` | Compare alpha `8` vs `16` on the web-crawled dataset |
| 5 | `03_test.ipynb` | Load trained weights, test prompts, export additional checkpoint results |
| 6 | `04_monitor.ipynb` | Inspect experiment state and generated artifacts |

---

## Outputs

**Default dataset training**

```
lora_experiments/dataset_training/
├── web_crawled/rank4_alpha4/
├── real/rank4_alpha4/
└── generated/rank4_alpha4/
```

**Before/after comparison images**

```
lora_experiments/comparisons/
├── web_crawled_before_after.png
├── real_before_after.png
└── generated_before_after.png
```

**Ablation experiments**

```
lora_experiments/ablation/
├── rank/
│   ├── rank8_alpha4/
│   └── rank16_alpha4/
├── alpha/
│   ├── rank4_alpha8/
│   └── rank4_alpha16/
├── rank_comparison.png
├── alpha_comparison.png
└── all_checkpoints/
```

**Summary CSVs**

- `lora_experiments/experiment_summary_rank.csv`
- `lora_experiments/experiment_summary_alpha.csv`

---

## Notebook Guide

| Notebook | Purpose |
|----------|---------|
| `00_Customizing_LoRA.ipynb` | End-to-end single LoRA workflow: data setup, captioning, metadata, training, and inference |
| `01_dataset_training.ipynb` | Dataset-wise LoRA training and before/after result generation |
| `02_ablation_rank.ipynb` | LoRA rank ablation experiment |
| `02_ablation_alpha.ipynb` | LoRA alpha ablation experiment |
| `03_test.ipynb` | Additional inference tests with trained LoRA weights and checkpoints |
| `04_monitor.ipynb` | Experiment artifact and progress monitoring |

---

## Tips

- **Limited GPU memory** — reduce `train_batch_size` or `max_train_steps`.
- **Outputs look too close to the base model** — verify that all captions include the `sks` style token.
- **LoRA overfits** — compare earlier checkpoints (`checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`).
- **For reports** — use the labeled assets in `report_assets/` rather than the raw grids in `lora_experiments/`.
