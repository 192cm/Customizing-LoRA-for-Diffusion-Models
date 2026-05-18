# Customizing LoRA for Diffusion Models

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2%20%2B%20CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/Diffusers-0.26.0-yellow)
![LoRA](https://img.shields.io/badge/LoRA-Stable%20Diffusion%20v1.5-7C3AED)

This project fine-tunes Stable Diffusion v1.5 with LoRA adapters on small custom image datasets, then compares how dataset source and LoRA hyperparameters affect generated images.

The repository provides a notebook-based workflow for data preparation, caption and metadata generation, LoRA training, checkpoint inference, rank/alpha ablation, and result visualization.

## Preview

### Dataset samples

![Dataset samples](report_assets/figure3_dataset_samples.png)

### Before / after LoRA

| Web-crawled dataset | Real dataset | Generated dataset |
| --- | --- | --- |
| ![Web crawled before after](report_assets/web_crawled_before_after_labeled.png) | ![Real before after](report_assets/real_before_after_labeled.png) | ![Generated before after](report_assets/generated_before_after_labeled.png) |

### Ablation results

| LoRA rank | LoRA alpha |
| --- | --- |
| ![Rank comparison](report_assets/rank_comparison_labeled.png) | ![Alpha comparison](report_assets/alpha_comparison_labeled.png) |

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Environment](#environment)
- [Dataset Format](#dataset-format)
- [Training Configuration](#training-configuration)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Notebook Guide](#notebook-guide)
- [Tips](#tips)

## Features

- Fine-tunes LoRA adapters for Stable Diffusion v1.5 using custom image datasets.
- Compares three dataset sources: web-crawled images, real photographs, and AI-generated images.
- Uses Hugging Face `imagefolder` format with caption metadata in `metadata.csv`.
- Generates before/after image grids comparing the base model and LoRA-adapted outputs.
- Runs ablation experiments for LoRA `rank` and `alpha`.
- Exports comparison images and summary CSV files for reporting.

## Repository Structure

```text
.
|-- 00_Customizing_LoRA.ipynb
|-- 01_dataset_training.ipynb
|-- 02_ablation_rank.ipynb
|-- 02_ablation_alpha.ipynb
|-- 03_test.ipynb
|-- 04_monitor.ipynb
|-- experiment_utils.py
|-- environment.yml
|-- README.md
|-- data/
|   |-- web_crawled_custom_dataset/
|   |   `-- train/
|   |       |-- metadata.csv
|   |       `-- 000001.jpg ... 000050.jpg
|   |-- real_custom_dataset/
|   |   `-- train/
|   |       |-- metadata.csv
|   |       `-- DSC_*.JPG
|   `-- generated_custom_dataset/
|       `-- train/
|           |-- metadata.csv
|           `-- ChatGPT Image *.png
|-- lora_experiments/
|   |-- dataset_training/
|   |-- comparisons/
|   |-- ablation/
|   |-- experiment_summary_rank.csv
|   `-- experiment_summary_alpha.csv
|-- report_assets/
|   `-- *_labeled.png
`-- sd_lora/
    |-- pytorch_lora_weights.safetensors
    |-- loss_log.csv
    `-- checkpoint-*/
```

## Environment

Create the conda environment from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate genai-assignment2
python -m ipykernel install --user --name genai-assignment2 --display-name "genai-assignment2"
```

Main dependencies:

| Package | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.1.2 + CUDA 12.1 |
| diffusers | 0.26.0 |
| transformers | 4.37.0 |
| accelerate | 0.26.1 |
| datasets | 2.16.1 |
| peft | 0.7.1 |

## Dataset Format

Each dataset follows the Hugging Face `imagefolder` layout. Images and `metadata.csv` are stored under `data/{dataset_name}/train/`.

```text
data/{dataset_name}/
`-- train/
    |-- metadata.csv
    |-- image_01.jpg
    `-- image_02.png
```

`metadata.csv` uses two columns:

```csv
file_name,caption
image_01.jpg,a sks building in pixel art style
```

The current dataset configuration is defined in `experiment_utils.py`.

| Name | Folder | Images | Prompt | Style token |
| --- | --- | ---: | --- | --- |
| `web_crawled` | `./data/web_crawled_custom_dataset` | 50 | `a building` | `sks` |
| `real` | `./data/real_custom_dataset` | 10 | `a city street in winter` | `sks` |
| `generated` | `./data/generated_custom_dataset` | 20 | `a house in a flower field` | `sks` |

## Training Configuration

Shared training and inference settings are centralized in `experiment_utils.py`.

| Setting | Value |
| --- | --- |
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

The final LoRA weights are saved as `pytorch_lora_weights.safetensors` in each experiment output directory.

## How to Run

Run the notebooks in order.

1. Open `00_Customizing_LoRA.ipynb` to validate the basic LoRA workflow, data preparation, caption generation, and single-run training.
2. Run `01_dataset_training.ipynb` to train the default `rank4_alpha4` LoRA on each dataset and generate before/after comparisons.
3. Run `02_ablation_rank.ipynb` to compare rank `8` and `16` on the web-crawled dataset.
4. Run `02_ablation_alpha.ipynb` to compare alpha `8` and `16` on the web-crawled dataset.
5. Use `03_test.ipynb` to load trained LoRA weights, test prompts, and export additional checkpoint results.
6. Use `04_monitor.ipynb` to inspect experiment state and generated artifacts.

## Outputs

Default dataset training outputs:

```text
lora_experiments/dataset_training/
|-- web_crawled/rank4_alpha4/
|-- real/rank4_alpha4/
`-- generated/rank4_alpha4/
```

Before/after comparison images:

```text
lora_experiments/comparisons/
|-- web_crawled_before_after.png
|-- real_before_after.png
`-- generated_before_after.png
```

Ablation outputs:

```text
lora_experiments/ablation/
|-- rank/
|   |-- rank8_alpha4/
|   `-- rank16_alpha4/
|-- alpha/
|   |-- rank4_alpha8/
|   `-- rank4_alpha16/
|-- rank_comparison.png
|-- alpha_comparison.png
`-- all_checkpoints/
```

Experiment summaries:

- `lora_experiments/experiment_summary_rank.csv`
- `lora_experiments/experiment_summary_alpha.csv`

## Notebook Guide

| Notebook | Purpose |
| --- | --- |
| `00_Customizing_LoRA.ipynb` | End-to-end single LoRA workflow: data setup, captioning, metadata, training, and inference |
| `01_dataset_training.ipynb` | Dataset-wise LoRA training and before/after result generation |
| `02_ablation_rank.ipynb` | LoRA rank ablation experiment |
| `02_ablation_alpha.ipynb` | LoRA alpha ablation experiment |
| `03_test.ipynb` | Additional inference tests with trained LoRA weights and checkpoints |
| `04_monitor.ipynb` | Experiment artifact and progress monitoring |

## Tips

- If GPU memory is limited, reduce `train_batch_size`, `max_train_steps`, or the number of inference images.
- If outputs look too similar to the base model, check whether captions consistently include the `sks` style token.
- If LoRA overfits, compare earlier checkpoints such as `checkpoint-500`, `checkpoint-1000`, and `checkpoint-1500`.
- For reports, use the labeled assets in `report_assets/` instead of the raw grids in `lora_experiments/`.
