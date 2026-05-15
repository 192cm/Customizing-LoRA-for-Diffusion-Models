# Customizing LoRA

Stable Diffusion v1.5에 세 가지 custom image dataset을 적용해 LoRA를 학습하고, 데이터셋별 before/after 결과와 LoRA rank/alpha ablation 결과를 비교하는 프로젝트입니다.

## 개요

프로젝트의 실행 흐름은 노트북별로 분리되어 있습니다.

1. `00_Customizing_LoRA.ipynb`: custom dataset 준비, 이미지 크롤링, caption 생성, metadata 작성, 단일 LoRA 학습 기본 흐름
2. `01_dataset_training.ipynb`: web-crawled, generated, real 데이터셋별 LoRA 학습 및 before/after 비교 이미지 생성
3. `02_ablation_rank.ipynb`: LoRA rank 값에 따른 ablation 실험
4. `02_ablation_alpha.ipynb`: LoRA alpha 값에 따른 ablation 실험
5. `03_test.ipynb`: 학습된 LoRA 로드, prompt 변경 테스트, checkpoint별 결과 이미지 export
6. `04_monitor.ipynb`: 학습/실험 상태 확인용 노트북

공통 학습, metadata, inference, 비교 이미지 생성 로직은 `experiment_utils.py`에 모아 두었습니다.

## 저장소 구조

```text
.
├── 00_Customizing_LoRA.ipynb
├── 01_dataset_training.ipynb
├── 02_ablation_rank.ipynb
├── 02_ablation_alpha.ipynb
├── 03_test.ipynb
├── 04_monitor.ipynb
├── experiment_utils.py
├── environment.yml
├── README.md
├── data/
│   ├── web_crawled_custom_dataset/
│   │   └── train/
│   │       ├── metadata.csv
│   │       └── 000001.jpg ... 000050.jpg
│   ├── generated_custom_dataset/
│   │   └── train/
│   │       ├── metadata.csv
│   │       └── ChatGPT Image 1.png ... ChatGPT Image 10.png
│   └── real_custom_dataset/
│       └── train/
│           ├── metadata.csv
│           └── DSC_*.JPG
├── lora_experiments/
│   ├── dataset_training/
│   ├── comparisons/
│   ├── ablation/
│   ├── experiment_summary_rank.csv
│   └── experiment_summary_alpha.csv
└── sd_lora/
    ├── pytorch_lora_weights.safetensors
    ├── loss_log.csv
    └── checkpoint-*/
```

## 환경 설정

`environment.yml`로 conda 환경을 생성합니다.

```bash
conda env create -f environment.yml
conda activate genai-assignment2
python -m ipykernel install --user --name genai-assignment2 --display-name "genai-assignment2"
```

주요 환경은 다음과 같습니다.

- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- diffusers 0.26.0
- transformers 4.37.0
- accelerate 0.26.1
- datasets 2.16.1
- peft 0.7.1

## 데이터셋

데이터셋은 Hugging Face `datasets.load_dataset("imagefolder", ...)`에서 읽을 수 있는 `imagefolder` 형식을 사용합니다. 각 데이터셋은 `data/{dataset_name}/train/` 아래에 이미지 파일과 `metadata.csv`를 둡니다.

```text
data/{dataset_name}/
└── train/
    ├── metadata.csv
    ├── image_01.jpg
    └── image_02.png
```

현재 사용하는 데이터셋 구성은 `experiment_utils.py`의 `DATASET_CONFIGS`에 정의되어 있습니다.

| Name | Folder | Images | Prompt |
| --- | --- | ---: | --- |
| `web_crawled` | `./data/web_crawled_custom_dataset` | 50 | `a building` |
| `real` | `./data/real_custom_dataset` | 10 | `a city street in winter` |
| `generated` | `./data/generated_custom_dataset` | 10 | `a street with buildings` |

`metadata.csv`는 `file_name,caption` 컬럼을 가지며, metadata가 없을 때는 BLIP captioning으로 생성합니다.

## 학습 설정

기본 설정은 `experiment_utils.py`에 정의되어 있습니다.

```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
SEED = 2015
train_batch_size = 8
resolution = 512
max_train_steps = 2000
checkpointing_steps = 500
learning_rate = 1e-4
DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 4
```

LoRA는 UNet attention 모듈의 `to_k`, `to_q`, `to_v`, `to_out.0`에 적용됩니다. 학습 중에는 500 step마다 checkpoint를 저장하고, 최종 LoRA weight는 각 실험 output directory의 `pytorch_lora_weights.safetensors`로 저장됩니다.

## 실행 순서

1. `00_Customizing_LoRA.ipynb`에서 기본 데이터 준비와 단일 학습 흐름을 확인합니다.
2. `01_dataset_training.ipynb`를 실행해 세 데이터셋별 기본 LoRA를 학습합니다.
3. `02_ablation_rank.ipynb`를 실행해 rank 8, 16 실험을 비교합니다.
4. `02_ablation_alpha.ipynb`를 실행해 alpha 8, 16 실험을 비교합니다.
5. `03_test.ipynb`에서 학습된 LoRA와 checkpoint 결과를 추가로 확인합니다.

GPU memory가 부족하면 `train_batch_size`, `max_train_steps`, inference image 수, 또는 ablation 실행 범위를 줄여서 실행합니다.

## 산출물

`01_dataset_training.ipynb`의 기본 데이터셋 학습 결과는 `lora_experiments/dataset_training/` 아래에 저장됩니다.

```text
lora_experiments/dataset_training/
├── web_crawled/rank4_alpha4/
├── real/rank4_alpha4/
└── generated/rank4_alpha4/
```

데이터셋별 before/after 비교 이미지는 `lora_experiments/comparisons/`에 저장됩니다.

```text
lora_experiments/comparisons/
├── web_crawled_before_after.png
├── real_before_after.png
└── generated_before_after.png
```

Ablation 결과는 `lora_experiments/ablation/`에 저장됩니다.

```text
lora_experiments/ablation/
├── rank/rank8_alpha4/
├── rank/rank16_alpha4/
├── alpha/rank4_alpha8/
├── alpha/rank4_alpha16/
├── rank_comparison.png
├── alpha_comparison.png
└── all_checkpoints/
```

실험 요약 CSV는 다음 파일에 기록됩니다.

- `lora_experiments/experiment_summary_rank.csv`
- `lora_experiments/experiment_summary_alpha.csv`

`sd_lora/`는 `00_Customizing_LoRA.ipynb`에서 사용하는 단일 LoRA 학습 산출물과 checkpoint를 담습니다.
