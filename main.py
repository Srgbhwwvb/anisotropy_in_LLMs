import gc
import json
import os
import random
import re
import sys
import importlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import login
from mpl_toolkits.mplot3d import Axes3D
from peft import LoraConfig, TaskType, get_peft_model
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel,
    GPT2Tokenizer, get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
import layerwise_cosine_anisotropy
import componentwise_anisotropy_analysis
import gride
import intrinsic
import mmap_load_for_train
import model
import remove_ffn
import save
import singular


from mmap_load_for_train import (
    load_dataset, load_data_for_training, prepare_batch_from_dataset,
    compute_val_loss_mmap, get_random_batch_from_dataset
)
importlib.reload(layerwise_cosine_anisotropy)
from layerwise_cosine_anisotropy import analyze_cosine_distribution
from model import load_gpt2, load_qwen, load_pythia, compute_val_loss, check_gradient_health

from intrinsic import (
    intrinsic_dimension_levina_bickel_vectorized, compute_intrinsic_dim_per_sample,
    measure_intrinsic_dimension_per_layer, plot_layerwise_intrinsic_dim_distribution
)
from remove_ffn import (
    zero_out_ffn_weights_gpt2, verify_ffn_disabled, zero_out_ffn_qwen,
    ResidualScaling, apply_residual_scaling
)
from save import setup_plot_saving, save_checkpoint_simple, manage_last_checkpoints
from singular import analyze_svd_anisotropy_by_layer

importlib.reload(componentwise_anisotropy_analysis)
from componentwise_anisotropy_analysis import (
    analyze_componentwise_anisotropy_qwen,
    analyze_intrinsic_dimension_pythia,
    compute_correlations_attention_weights,
    analyze_componentwise_anisotropy_gpt2,
    analyze_intrinsic_dimension_distribution,
    analyze_componentwise_anisotropy_pythia
)

from gride import measure_intrinsic_dimension_per_layer_gride

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



def hard_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
def clear_gpu_memory() -> None:
    """Очистка кэша GPU и сборка мусора."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def set_seed(seed: int = 42) -> None:
    """Фиксация всех генераторов случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
                
@dataclass
class TrainingConfig:
    """Конфигурация эксперимента по дообучению и анализу анизотропии."""
    model_name: str = "GPT2"   # "GPT2", "Qwen", "Pythia"
    use_large: bool = False    # использовать большую версию модели
    remove_ffn: bool = False   # занулять FFN слои
    num_iterations: int = 1  # количество прогонов 
    finetune: bool = False    # выполнять дообучение или только анализ
    load_checkpoint: bool = False    # загружать чекпоинт

    # Параметры данных
    text_amount: int = 50000   # количество текстов для загрузки
    wiki: bool = True  # использовать WikiText или realnewslike
    block_size: int = 512  # длина последовательности
    train_size: int = 8000  # размер обучающей выборки
    val_size: int = 1000  # размер валидационной выборки
    num_samples: int = 10000  # число векторов для анализа (подвыборка)

    # Параметры обучения
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_epochs: int = 30
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    num_cycles: float = 0.5
    eval_interval: int = 100               
    val_split: float = 0.2
    gradient_accumulation_steps: int = 4

    # LoRA
    lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Анализ
    intrinsic_dim: bool = True             
    texts_for_cosine: int = 8              # число текстов для анализа косинусов

    # Сохранение
    save_dir: str = "finetune_checkpoints"
    save_best: bool = True

    # Дополнительные флаги
    remove_norm: bool = False
    pretrained: bool = True
    max_train_samples: int = 800000

    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
    
config = TrainingConfig()
os.makedirs(config.save_dir, exist_ok=True)


def initialize_experiment(config: TrainingConfig):
    """Загружает данные, модель, ускоритель."""
    set_seed(42)
    clear_gpu_memory()
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Загрузка данных
    train_loader, val_dataset, tokenizer = load_dataset(config)
    # Модель
    if config.model_name == "GPT2":
        model = load_gpt2(config)
    elif config.model_name == "Qwen":
        model = load_qwen(config)
    else:
        model = load_pythia(config)

    # Подготовка одного батча для анализа
    batch = get_random_batch_from_dataset(
        dataset=val_dataset,
        batch_size=min(config.texts_for_cosine, len(val_dataset)),
        device="cpu"
    )
    model, batch = accelerator.prepare(model, batch)

    return accelerator, model, train_loader, val_dataset, batch


def run_full_analysis(model, batch, config, step_label=""):
    """Выполняет весь набор анализов: косинусный, SVD, компонентный, intrinsic dimensions."""
    logging.info(f"Running full analysis: {step_label}")
    analyze_cosine_distribution(model, batch, config)
    analyze_svd_anisotropy_by_layer(model, batch, config)

    if config.model_name == "GPT2":
        analyze_componentwise_anisotropy_gpt2(model, batch, config)
    elif config.model_name == "Qwen":
        analyze_componentwise_anisotropy_qwen(model, batch, config)
    else:
        analyze_componentwise_anisotropy_pythia(model, batch, config)

    if config.intrinsic_dim:
        measure_intrinsic_dimension_per_layer(model, batch)
        # MLE анализ (если есть)
        # GRIDE с разными k
        for k in [10, 20, 30]:
            measure_intrinsic_dimension_per_layer_gride(
                model, batch, maxk=k, scale_method='last',
                plot=True, include_embedding=True, max_points=10000
            )
            
def finetune_with_anisotropy_measurement(config: TrainingConfig):
    """Основной цикл эксперимента."""
    accelerator, model, train_loader, val_dataset, batch = initialize_experiment(config)

    # Начальный анализ
    run_full_analysis(model, batch, config, "initial")

    # Валидация перед обучением
    val_loss = compute_val_loss_mmap(model, val_dataset, accelerator.device, config)
    logging.info(f"Initial validation loss: {val_loss:.4f}")

    if config.finetune:
        # Подготовка оптимизатора и планировщика
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup_steps,
            num_training_steps=len(train_loader) * config.max_epochs,
            num_cycles=config.num_cycles
        )
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

        best_val_loss = float('inf')
        for epoch in range(config.max_epochs):
            logging.info(f"Epoch {epoch+1}/{config.max_epochs}")
            model.train()
            total_loss = 0.0
            step_loss = 0.0
            step_count = 0

            progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False)
            for batch_idx, batch_data in enumerate(progress_bar):
                if isinstance(batch_data, dict):
                    batch = batch_data
                else:
                    batch = prepare_batch_from_dataset(batch_data)

                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"],
                        use_cache=False
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # логирование
                step_loss += loss.item()
                total_loss += loss.item()
                step_count += 1
                if step_count % 10 == 0:
                    progress_bar.set_postfix({'avg_loss': total_loss / step_count})

                # Периодический анализ и валидация
                if step_count % config.eval_interval == 0:
                    val_loss = compute_val_loss_mmap(model, val_dataset, accelerator.device, config)
                    logging.info(f"Step {step_count}: val_loss = {val_loss:.4f}")
                    # Анализ на валидационном батче
                    val_batch = get_random_batch_from_dataset(
                        dataset=val_dataset,
                        batch_size=min(config.texts_for_cosine, len(val_dataset)),
                        device=accelerator.device
                    )
                    run_full_analysis(model, val_batch, config, f"step_{step_count}")
                    # Сохранение лучшей модели
                    if config.save_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint_simple(model, config, epoch, val_loss, is_best=True)

            # Конец эпохи – валидация
            val_loss = compute_val_loss_mmap(model, val_dataset, accelerator.device, config)
            logging.info(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")
            # Сохранение чекпоинта по эпохам
            save_checkpoint_simple(model, config, epoch, val_loss)

    else:
        logging.info("Finetuning disabled, only initial analysis performed.")



if __name__ == "__main__":
    
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    set_seed(42)
    
    config = TrainingConfig(
        remove_ffn=False,
        text_amount=15000,  
        block_size=512,       
        model_name="Pythia", # or Qwen or GPT2
        max_epochs=3,
        intrinsic_dim=True,
        wiki=True,          
        lora=True,
        num_samples=18000,
        finetune=False,
        batch_size=1,
        load_checkpoint=False,
        remove_norm=False
    )
    finetune_with_anisotropy_measurement(config)


    
