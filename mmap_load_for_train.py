import os
import numpy as np
import torch
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmap_loader import load_mmap_data, MemoryMappedDatasetLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Tokenizer

def load_dataset(config):
    """Загрузка и подготовка данных"""

    if config.wiki:
        print("Загрузка Wiki")
        dataset = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en", 
            split=f"train[:{config.text_amount}]",
            trust_remote_code=True
        )
    else:
        print("Загрузка RealNews")
        dataset = load_dataset(
            "allenai/c4",
            "realnewslike",
            split=f"train[:{config.text_amount}]",
            # trust_remote_code не нужен
        )
    
    # Очистка текста
    dataset = dataset.map(
        lambda x: {'text': clean_wikipedia_text(x['text'])},
        desc="Очистка текста"
    )
    
    # Фильтрация пустых текстов
    dataset = dataset.filter(lambda x: len(x['text']) > 2048)
    print("len(dataset)", len(dataset))

    # for i in range(3):
    #     print(dataset[i]["text"])
    
    # Загрузка токенизатора
    if config.model_name=="GPT2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif config.model_name=="Qwen": 
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-32B", 
            trust_remote_code=True
        )
    else: 
        model_name = "EleutherAI/pythia-1.4b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Разделение на train/val
    train_size = min(config.train_size, len(dataset))
    val_size = min(config.val_size, len(dataset) - train_size)
    
    train_dataset = dataset.select(range(train_size-1))
    val_dataset = dataset.select(range(train_size, train_size + val_size-1))
    
    # Токенизация
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=config.block_size
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

    # Конвертируем в формат PyTorch
    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask'])
    tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    # Создаем DataLoader'
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, tokenized_val, tokenizer


    
def load_data_for_training(config):
    """
    Универсальная загрузка данных с поддержкой memory-mapped формата
    Возвращает: train_loader, val_dataset, tokenizer
    """
    
    # Путь к предобработанным данным
    if config.model_name=="GPT2":
        DATA_PATH = "./wiki_gpt_memory_mapped"
    else:
        if config.wiki:
            DATA_PATH = "./wiki_qwen_memory_mapped"
        else:
            DATA_PATH = "./qwen_memory_mapped"  # Измените при необходимости
    
    # Проверяем существование данных
    if not os.path.exists(DATA_PATH):
        print(f"Memory-mapped данные не найдены в {DATA_PATH}")
        print("Сначала запустите create_mmap_dataset.py")
        raise FileNotFoundError(f"Директория {DATA_PATH} не найдена")
    
    # Загружаем memory-mapped данные
    train_dataset, val_dataset, tokenizer = load_mmap_data(DATA_PATH)
    
    # Создаем DataLoader для тренировочных данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Можно увеличить для скорости
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train DataLoader создан: {len(train_loader)} батчей")
    print(f"Val dataset: {len(val_dataset)} примеров")
    
    return train_loader, val_dataset, tokenizer


def prepare_batch_from_dataset(batch_samples):
    """Подготовка батча из memory-mapped датасета"""
    # Если batch_samples уже словарь (от DataLoader), возвращаем как есть
    if isinstance(batch_samples, dict):
        return batch_samples
    
    # Если это список элементов от датасета
    input_ids = torch.stack([item["input_ids"] for item in batch_samples])
    attention_mask = torch.stack([item["attention_mask"] for item in batch_samples])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def compute_val_loss_mmap(model, val_dataset, device, config, accelerator=None, 
                         max_samples=1000, use_random_sampling=True):
    """Вычисление loss на подмножестве валидационного набора"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Определяем сколько примеров будем использовать
    total_size = len(val_dataset)
    if max_samples > total_size:
        max_samples = total_size
    
    # Определяем индексы для выборки
    if use_random_sampling:
        # Случайная выборка
        indices = np.random.choice(total_size, size=max_samples, replace=False)
    else:
        # Первые N примеров
        indices = np.arange(min(max_samples, total_size))
    
    print(f"Вычисление валидационного loss на {len(indices)} примерах из {total_size}")
    
    with torch.no_grad():
        # Разбиваем индексы на батчи
        num_batches_total = (len(indices) + config.batch_size - 1) // config.batch_size
        
        # Прогресс-бар
        pbar = tqdm(total=num_batches_total, desc="Валидация", leave=False, mininterval=4.5)
        
        for batch_start in range(0, len(indices), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # Подготовка батча
            input_ids_list = []
            attention_mask_list = []
            
            for idx in batch_indices:
                item = val_dataset[int(idx)]
                input_ids_list.append(item['input_ids'])
                attention_mask_list.append(item['attention_mask'])
            
            if not input_ids_list:
                continue
                
            max_len = max(len(ids) for ids in input_ids_list)
            
            input_ids = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
            attention_mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
            
            for j, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
                input_ids[j, :len(ids)] = torch.tensor(ids)
                attention_mask[j, :len(mask)] = torch.tensor(mask)
            
            # Перемещаем на устройство
            if accelerator is not None:
                input_ids = input_ids.to(accelerator.device)
                attention_mask = attention_mask.to(accelerator.device)
            else:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Обновляем прогресс-бар
            pbar.update(1)
            # pbar.set_postfix({
            #     'loss': f'{outputs.loss.item():.4f}',
            #     'avg_loss': f'{total_loss / num_batches:.4f}'
            # })
        
        pbar.close()
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Средний валидационный loss: {avg_loss:.4f}")
    
    return avg_loss


def get_random_batch_from_dataset(dataset, batch_size, device='cpu'):
    """Получение случайного батча из датасета"""
    indices = np.random.choice(len(dataset), size=batch_size, replace=False)
    
    # Преобразуем numpy.int64 в Python int
    indices = [int(idx) for idx in indices]
    
    # Получаем данные
    input_ids_list = []
    attention_mask_list = []
    
    for idx in indices:
        item = dataset[idx]
        input_ids_list.append(item['input_ids'])
        attention_mask_list.append(item['attention_mask'])
    
    # Преобразуем в тензоры
    max_len = max(len(ids) for ids in input_ids_list)
    
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        input_ids[i, :len(ids)] = ids.clone() if isinstance(ids, torch.Tensor) else torch.tensor(ids)
        attention_mask[i, :len(mask)] = mask.clone() if isinstance(mask, torch.Tensor) else torch.tensor(mask)

    # Перемещаем на указанное устройство
    if device != 'cpu':
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device)
        }
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }





def clean_wikipedia_text(text):
    """Очистка текста Wikipedia"""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Удаляем не-ASCII символы
    text = re.sub(r'\[\[.*?\]\]', '', text)     # Удаляем [[...]]
    text = re.sub(r'\{\{.*?\}\}', '', text)     # Удаляем {{...}}
    text = re.sub(r'<[^>]+>', '', text)         # Удаляем HTML теги
    text = re.sub(r'\s+', ' ', text)            # Удаляем лишние пробелы
    return text.strip()



