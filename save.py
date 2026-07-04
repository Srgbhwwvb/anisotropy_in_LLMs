import os
import json
import shutil
import re
from datetime import datetime


STEP_CHECKPOINTS_TO_KEEP = 2  # Хранить 2 последних чекпоинта по шагам
LAST_CHECKPOINTS_DIR = "last_checkpoints"  # Папка для последних чекпоинтов
BEST_CHECKPOINT_DIR = "best_checkpoint"    # Папка для лучшего чекпоинта


def setup_plot_saving(config, suffix=""):
    """Создает папку для сохранения графиков с временной меткой"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.model_name == "GPT2":
        model_size = "large" if config.use_large else "small"
    else:
        model_size = "Qwen"
    
    plot_dir = f"plots_{timestamp}_{model_size}_{suffix}"
    if config.remove_ffn:
        plot_dir += "_noffn"
    if config.lora:
        plot_dir += "_lora"
    
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def save_checkpoint_simple(accelerator, epoch, val_loss, config, is_best=False):
    """Простая функция сохранения чекпоинта"""
    # Определяем папку и имя чекпоинта
    if is_best:
        checkpoint_dir = os.path.join(config.save_dir, BEST_CHECKPOINT_DIR)
        checkpoint_name = f"best_loss_{val_loss:.4f}"
    else:
        checkpoint_dir = os.path.join(config.save_dir, LAST_CHECKPOINTS_DIR)
        checkpoint_name = f"loss_{val_loss:.4f}"
    
    # Создаем папку
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Сохраняем через Accelerator
    accelerator.save_state(checkpoint_path)
    
    print(f" Чекпоинт сохранен: {checkpoint_name}")
    
    # Для обычных чекпоинтов - ограничиваем количество
    if not is_best:
        manage_last_checkpoints(checkpoint_dir)

def manage_last_checkpoints(checkpoint_dir):
    """Управляет последними чекпоинтами (оставляет только STEP_CHECKPOINTS_TO_KEEP)"""
    if not os.path.exists(checkpoint_dir):
        return
    
    # Получаем список чекпоинтов
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            # Сортируем по времени создания
            ctime = os.path.getctime(item_path)
            checkpoints.append((item_path, ctime))
    
    # Сортируем по времени (старые первые)
    checkpoints.sort(key=lambda x: x[1])
    
    # Удаляем лишние
    while len(checkpoints) > STEP_CHECKPOINTS_TO_KEEP:
        oldest_path, _ = checkpoints.pop(0)
        shutil.rmtree(oldest_path, ignore_errors=True)
        print(f"  Удален старый чекпоинт: {os.path.basename(oldest_path)}")

