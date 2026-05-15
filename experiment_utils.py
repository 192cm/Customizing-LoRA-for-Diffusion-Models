# Standard library imports
import gc
import math
import os
import random
import re

# Third-party imports
import accelerate
import numpy as np
import pandas
import glob
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image, ImageDraw
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import make_image_grid, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig

# ── Model & training constants ─────────────────────────────────────────────────
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
VARIANT = "fp16"

SEED = 2015
train_batch_size = 8
resolution = 512
max_train_steps = 2000
checkpointing_steps = 500
learning_rate = 1e-4

# Populated by load_base_components(); read by training_function via module globals
tokenizer_one = None
noise_scheduler = None

logger = get_logger(__name__)

# ── Experiment configuration ────────────────────────────────────────────────────
BASE_OUTPUT_DIR = "./lora_experiments"
BASE_COMPARISON_DIR = os.path.join(BASE_OUTPUT_DIR, "comparisons")
BASE_DATASET_TRAINING_DIR = os.path.join(BASE_OUTPUT_DIR, "dataset_training")
BASE_ABLATION_DIR = os.path.join(BASE_OUTPUT_DIR, "ablation")

for _d in [BASE_OUTPUT_DIR, BASE_COMPARISON_DIR, BASE_DATASET_TRAINING_DIR, BASE_ABLATION_DIR]:
    os.makedirs(_d, exist_ok=True)

DATASET_CONFIGS = [
    {
        # 8-bit pixel art 게임 캐릭터/스프라이트 이미지
        "name": "web_crawled",
        "data_dir": "./data/web_crawled_custom_dataset",
        "prompt": "a building",
        "style_token": "sks",
        "style_words": sorted([
            "pixel art", "pixel-art", "pixelated", "pixels", "pixel",
            "8-bit", "8bit", "8 bit", "16-bit", "16bit",
            "retro", "sprite", "sprite sheet", "blocky", "mosaic",
            "video game", "game over", "game", "low resolution", "low-res",
        ], key=len, reverse=True),
    },
    {
        # 일본 홋카이도 겨울 거리 실제 사진
        # BLIP은 색온도/보정을 텍스트로 표현하지 않으므로 style_words 없이
        # style_token만 붙이면 LoRA가 일관된 색감/계조를 sks에 귀속시킴
        "name": "real",
        "data_dir": "./data/real_custom_dataset",
        "prompt": "a city street in winter",
        "style_token": "sks",
        "style_words": [],
    },
    {
        # 종이 공예(paper craft) 스타일 AI 생성 일러스트
        "name": "generated",
        "data_dir": "./data/generated_custom_dataset",
        "prompt": "a house in a flower field",
        "style_token": "sks",
        "style_words": sorted([
            "paper cut art", "paper cut", "paper craft", "paper-cut", "papercut",
            "layered paper", "cut paper", "paper collage", "paper art",
            "paper", "craft", "illustration", "digital art", "concept art",
        ], key=len, reverse=True),
    },
]

DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 4
DEFAULT_EXPERIMENT_STEPS = 2000
NUM_IMAGES_PER_PROMPT = 3
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, text, watermark"
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5

ABLATION_DATASET = DATASET_CONFIGS[0]
LORA_RANK_EXPERIMENTS = [8, 16]
LORA_ALPHA_EXPERIMENTS = [8, 16]

experiment_records = []


# ── Prompt helpers (mirrored from Section 2 for standalone use) ────────────────
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            try:
                text_input_ids = text_input_ids_list[i]
            except IndexError:
                pass

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel(text_encoder_config).from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder
        )
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection(text_encoder_config).from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder
        )
    else:
        raise ValueError(f"{model_class} is not supported.")


# ── Training function (mirrored from Section 2 for standalone use) ─────────────
# tokenizer_one and noise_scheduler are read from this module's globals,
# which are set by load_base_components() before notebook_launcher is called.
def training_function(
    dataset,
    unet,
    vae,
    text_encoder_one,
    text_encoder_two=None,
    lora_rank=DEFAULT_LORA_RANK,
    lora_alpha=DEFAULT_LORA_ALPHA,
    run_output_dir=None,
    run_max_train_steps=None,
):
    current_output_dir = run_output_dir or os.path.join(BASE_OUTPUT_DIR, "sd_lora")
    current_max_train_steps = run_max_train_steps or max_train_steps
    os.makedirs(current_output_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=VARIANT)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_two is not None:
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    cast_training_params([unet], dtype=torch.float32)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            if text_encoder_two is not None:
                text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif text_encoder_two is not None and isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                if weights:
                    weights.pop()

            if text_encoder_two is not None:
                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )
            else:
                StableDiffusionPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif text_encoder_two is not None and isinstance(
                model, type(unwrap_model(text_encoder_two))
            ):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )
        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_one_,
        )
        if text_encoder_two_ is not None:
            text_encoder_2_state_dict = {
                k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
            }
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=text_encoder_two_,
            )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer_class = torch.optim.AdamW
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        # lr=learning_rate * train_batch_size,
        lr=learning_rate,  
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    column_names = dataset["train"].column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        # tokenizer_one is the module-level global set by load_base_components()
        tokens_one = tokenize_prompt(tokenizer_one, captions)
        tokens_two = None
        return tokens_one, tokens_two

    train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if random.random() < 0.5:
                image = train_flip(image)
            y1 = max(0, int(round((image.height - resolution) / 2.0)))
            x1 = max(0, int(round((image.width - resolution) / 2.0)))
            image = train_crop(image)
            crop_top_lefts.append((y1, x1))
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        if text_encoder_two is not None:
            examples["input_ids_two"] = tokens_two
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        if text_encoder_two is not None:
            input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
            return {
                "pixel_values": pixel_values,
                "input_ids_one": input_ids_one,
                "input_ids_two": input_ids_two,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
            }
        else:
            return {
                "pixel_values": pixel_values,
                "input_ids_one": input_ids_one,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
            }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=0,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=current_max_train_steps * 0.1 * 1,
        num_training_steps=current_max_train_steps * 1,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_train_epochs = math.ceil(current_max_train_steps / len(train_dataloader))
    total_batch_size = train_batch_size * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {current_max_train_steps}")

    resume_from_checkpoint = os.path.join(current_output_dir, "checkpoint-1500")
    if not os.path.isdir(resume_from_checkpoint):
        resume_from_checkpoint = None

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if resume_from_checkpoint is not None:
        accelerator.load_state(resume_from_checkpoint)
        cast_training_params([unet], dtype=torch.float32)

        base_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
        for group in base_optimizer.param_groups:
            for param in group["params"]:
                state = base_optimizer.state.get(param, {})
                for key, value in state.items():
                    if torch.is_tensor(value):
                        if key == "step":
                            state[key] = value.to(device=param.device, dtype=torch.float32)
                        else:
                            state[key] = value.to(device=param.device, dtype=param.dtype)

        checkpoint_name = os.path.basename(resume_from_checkpoint)
        global_step = int(checkpoint_name.split("-")[-1])

        if accelerator.is_main_process:
            loss_log_path = os.path.join(current_output_dir, "loss_log.csv")
            if os.path.exists(loss_log_path):
                loss_log = pandas.read_csv(loss_log_path)
                loss_log = loss_log[loss_log["step"] <= global_step]
                loss_log.to_csv(loss_log_path, index=False)

        first_epoch = global_step // len(train_dataloader)
        resume_step = global_step % len(train_dataloader)

        print(f"Resumed from {resume_from_checkpoint} at step {global_step}")

    progress_bar = tqdm(
        range(0, current_max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if resume_from_checkpoint is not None and epoch == first_epoch and step < resume_step:
                continue

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                timesteps = timesteps.long()
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                def compute_time_ids(original_size, crops_coords_top_left):
                    target_size = (resolution, resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [
                        compute_time_ids(s, c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )

                unet_added_conditions = {"time_ids": add_time_ids}

                if text_encoder_two is not None:
                    text_input_ids_list = [batch["input_ids_one"], batch["input_ids_two"]]
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=text_input_ids_list,
                    )
                else:
                    text_input_ids_list = [batch["input_ids_one"]]
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=text_input_ids_list,
                    )

                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        save_path = os.path.join(current_output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= current_max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=current_output_dir,
            unet_lora_layers=unet_lora_state_dict,
        )

    accelerator.free_memory()
    del unet, vae, text_encoder_one, optimizer, train_dataloader, lr_scheduler
    if text_encoder_two is not None:
        del text_encoder_two
    gc.collect()
    torch.cuda.empty_cache()


# ── Section 3: Dataset utilities ────────────────────────────────────────────────
def list_image_files(train_dir):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in glob.glob(os.path.join(train_dir, "*"))
        if os.path.splitext(path)[1].lower() in image_extensions
    )


def generate_metadata_with_blip(data_dir, processor, blip_model, device, style_token=None, style_words=None):
    train_dir = os.path.join(data_dir, "train")
    image_paths = list_image_files(train_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No train images found in {train_dir}")

    caption_rows = []
    for image_path in tqdm(image_paths, desc=f"Captioning {os.path.basename(data_dir)}"):
        image = Image.open(image_path).convert("RGB")

        # conditional captioning으로 content만 추출
        inputs = processor(image, "a photo of", return_tensors="pt").to(device)
        outputs = blip_model.generate(**inputs, max_new_tokens=75, num_beams=5)
        blip_caption = processor.decode(outputs[0], skip_special_tokens=True)

        if style_token:
            # 스타일 단어 제거 후 style token 삽입
            content = blip_caption.replace("a photo of", "").strip()
            for word in (style_words or []):
                content = re.sub(rf'\b{re.escape(word)}\b', '', content, flags=re.IGNORECASE).strip()
            content = re.sub(r'\s+', ' ', content).strip()
            caption = f"{style_token} {content}"
        else:
            caption = blip_caption

        caption_rows.append({"file_name": os.path.basename(image_path), "caption": caption})

    metadata_df = pandas.DataFrame(caption_rows).sort_values("file_name").reset_index(drop=True)
    metadata_path = os.path.join(train_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    return metadata_path


def prepare_dataset(dataset_config, processor, blip_model, device):
    data_dir = dataset_config["data_dir"]
    metadata_path = os.path.join(data_dir, "train", "metadata.csv")

    if not os.path.exists(metadata_path):
        metadata_path = generate_metadata_with_blip(
            data_dir,
            processor,
            blip_model,
            device,
            style_token=dataset_config.get("style_token"),
            style_words=dataset_config.get("style_words", []),
        )

    dataset = load_dataset("imagefolder", data_dir=data_dir)
    print(f"Loaded {dataset_config['name']} from {data_dir}; metadata={metadata_path}")
    return dataset


def load_base_components():
    """Load SD v1.5 sub-models and update module-level tokenizer_one / noise_scheduler."""
    global tokenizer_one, noise_scheduler

    tokenizer_one = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer",
        use_fast=True,
    )
    text_encoder = import_model_class_from_model_name_or_path(MODEL_NAME, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    vae_model = AutoencoderKL.from_pretrained(VAE_NAME)
    unet_model = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet", variant=VARIANT)

    vae_model.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_model.requires_grad_(False)
    vae_model.eval()
    text_encoder.eval()

    return unet_model, vae_model, text_encoder


# ── Section 3: Training & inference helpers ─────────────────────────────────────
def run_lora_training_experiment(
    dataset_config,
    processor,
    blip_model,
    device,
    lora_rank=DEFAULT_LORA_RANK,
    lora_alpha=DEFAULT_LORA_ALPHA,
    max_steps=DEFAULT_EXPERIMENT_STEPS,
    experiment_type="dataset_training",
):
    dataset = prepare_dataset(dataset_config, processor, blip_model, device)
    unet_model, vae_model, text_encoder = load_base_components()

    output_subdir = f"rank{lora_rank}_alpha{lora_alpha}"
    if experiment_type == "dataset_training":
        run_output_dir = os.path.join(BASE_DATASET_TRAINING_DIR, dataset_config["name"], output_subdir)
    else:
        run_output_dir = os.path.join(BASE_ABLATION_DIR, experiment_type, output_subdir)
    os.makedirs(run_output_dir, exist_ok=True)

    try:
        accelerate.notebook_launcher(
            training_function,
            args=(
                dataset,
                unet_model,
                vae_model,
                text_encoder,
                None,
                lora_rank,
                lora_alpha,
                run_output_dir,
                max_steps,
            ),
            num_processes=1,
        )
    finally:
        del unet_model, vae_model, text_encoder, dataset
        gc.collect()
        torch.cuda.empty_cache()

    experiment_records.append({
        "experiment_type": experiment_type,
        "dataset": dataset_config["name"],
        "prompt": dataset_config["prompt"],
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_train_steps": max_steps,
        "output_dir": run_output_dir,
        "result_image": "",
    })
    return run_output_dir


def load_inference_pipeline(device):
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant=VARIANT if torch.cuda.is_available() else None,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def label_image(image, label, strip_height=36):
    labeled = Image.new("RGB", (image.width, image.height + strip_height), "white")
    labeled.paste(image.convert("RGB"), (0, strip_height))
    draw = ImageDraw.Draw(labeled)
    draw.text((10, 10), label, fill="black")
    return labeled


def generate_images(
    pipe,
    prompt,
    device,
    seed=SEED,
    num_images=NUM_IMAGES_PER_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images


def generate_before_after_grid(dataset_config, lora_dir, save_path, device, seed=SEED):
    prompt = dataset_config["prompt"]

    base_pipe = load_inference_pipeline(device)
    before_images = generate_images(base_pipe, prompt, device, seed)
    del base_pipe
    torch.cuda.empty_cache()

    lora_pipe = load_inference_pipeline(device)
    lora_pipe.load_lora_weights(lora_dir)
    after_images = generate_images(lora_pipe, prompt, device, seed)
    del lora_pipe
    torch.cuda.empty_cache()

    labeled_before = [label_image(img, f"Before {i + 1}") for i, img in enumerate(before_images)]
    labeled_after = [label_image(img, f"After {i + 1}") for i, img in enumerate(after_images)]
    grid = make_image_grid(labeled_before + labeled_after, rows=2, cols=NUM_IMAGES_PER_PROMPT)
    grid.save(save_path)
    return grid


def generate_lora_comparison_grid(prompt, lora_dirs, labels, save_path, device, seed=SEED):
    images = []
    for lora_dir, label in zip(lora_dirs, labels):
        pipe = load_inference_pipeline(device)
        pipe.load_lora_weights(lora_dir)
        image = generate_images(pipe, prompt, device, seed=seed, num_images=1)[0]
        images.append(label_image(image, label))
        del pipe
        torch.cuda.empty_cache()

    grid = make_image_grid(images, rows=1, cols=len(images))
    grid.save(save_path)
    print("Comparison order:", labels)
    return grid
