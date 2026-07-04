import torch
import importlib
#from prepare_data import load_and_preprocess_data, prepare_batch
import remove_ffn
from remove_ffn import zero_out_ffn_weights_gpt2, verify_ffn_disabled, zero_out_ffn_qwen
from transformers import GPT2LMHeadModel
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from datetime import datetime
from transformers import GPT2Config
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file


def check_gradient_health(model):
    """Проверяет и корректирует проблемные градиенты"""
    total_grad_norm = 0
    vanishing_count = 0
    exploding_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            
            # Проверка затухающих градиентов
            if grad_norm < 1e-7:
                vanishing_count += 1
                #print(f" Затухающий градиент в {name}: {grad_norm:.2e}")
            
            # Проверка взрывающихся градиентов
            if grad_norm > 10.0:
                exploding_count += 1
                #print(f" Взрывающийся градиент в {name}: {grad_norm:.2e}")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Общая норма градиентов: {total_grad_norm:.4f}")
    print(f"Затухающих: {vanishing_count}, Взрывающихся: {exploding_count}")
    
    return total_grad_norm

    
def remove_model_layers(model, layer_indices):
    """Удаляет указанные слои с сохранением совместимости"""

    # Получаем слои
    layers = getattr(model.model, 'layers')
    
    # Создаем новые слои без удаляемых
    new_layers = torch.nn.ModuleList([
        layer for i, layer in enumerate(layers) 
        if i not in layer_indices
    ])
    
    # Заменяем слои
    setattr(model.model, 'layers', new_layers)
    
    model.config.num_hidden_layers = len(new_layers)
    
    print(f"Удалено слоев: {len(layer_indices)}")
    print(f"Осталось слоев: {len(new_layers)}")
    print(f"Удаленные индексы: {layer_indices}")
    
    return model



def load_gpt2(config):
    """Загрузка предобученной GPT-2 small или large с хуками"""
    if config.use_large==1:
        model_name = "gpt2-large" 
    elif config.use_large==2:
        model_name = "gpt2-medium" 
    elif config.use_large==3:
        model_name="gpt2-xl"
    else:
        model_name = "gpt2" 

    print(f"Загрузка предобученной {model_name} из Hugging Face...")

    if config.pretrained:
        model = GPT2LMHeadModel.from_pretrained(
            model_name
        )
    else:
        if config.use_large==1:
            hf_config = GPT2Config.from_pretrained("gpt2-large", torch_dtype=torch.bfloat16)
        elif config.use_large==2:
            hf_config = GPT2Config.from_pretrained("gpt2-xl",torch_dtype=torch.bfloat16)
        else:
            hf_config = GPT2Config.from_pretrained("gpt2",torch_dtype=torch.bfloat16)
        
        model=GPT2LMHeadModel(hf_config)

    
    model.config.pad_token_id = model.config.eos_token_id
    print(f"Конфигурация {model_name}:")
    print(f"  - Слоев (n_layer): {model.config.n_layer}")
    print(f"  - Голов внимания (n_head): {model.config.n_head}")
    print(f"  - Размер эмбеддинга (n_embd): {model.config.n_embd}")
    print(f"  - Размер словаря (vocab_size): {model.config.vocab_size}")
    print(f"  - Длина контекста (n_positions): {model.config.n_positions}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {num_params:,}")
    
    # Если нужно удалить FFN
    if config.remove_ffn:
        model = zero_out_ffn_weights_gpt2(model)
        print(f"\nМодель {model_name} загружена с отключенными FFN слоями!")
    
    return model


import torch.nn as nn
import types

def disable_qwen_normalization(model):
    """
    Отключает нормализацию в модели Qwen2.5
    """
    disabled_count = 0
    
    # Импортируем нужный класс, если доступен
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        NormClass = Qwen2RMSNorm
        print("Используется Qwen2RMSNorm из transformers")
    except ImportError:
        # Альтернативный способ - находим класс по первому экземпляру
        for _, module in model.named_modules():
            if 'Qwen2RMSNorm' in str(type(module)):
                NormClass = type(module)
                print(f"Используется NormClass из модели: {NormClass}")
                break
        else:
            NormClass = None
            print("Не удалось определить класс нормализации")
            return model
    
    def identity_forward(self, x, *args, **kwargs):
        """Тождественное преобразование вместо нормализации"""
        return x
    
    for name, module in model.named_modules():
        if isinstance(module, NormClass):
            print(f"Отключаю: {name}")
            
            # Заменяем forward метод
            module.forward = types.MethodType(identity_forward, module)
            
            # Отключаем градиенты для параметров
            if hasattr(module, 'weight'):
                module.weight.requires_grad = False
                # Также можно установить вес в 1.0 для стабильности
                with torch.no_grad():
                    module.weight.fill_(1.0)
            
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad = False
                with torch.no_grad():
                    module.bias.fill_(0.0)
            
            disabled_count += 1
    
    print(f"\nВсего отключено слоев нормализации: {disabled_count}")
    return model



def load_pythia(config):
    model_name = "EleutherAI/pythia-1.4b"
    from transformers import GPTNeoXForCausalLM

    if config.pretrained:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     attn_implementation="eager",  # Явно используем стандартную реализацию
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
    else:
        config_model = AutoConfig.from_pretrained(model_name)
        model = GPTNeoXForCausalLM(config_model)
        model = model.to(dtype=torch.bfloat16)  # при необходимости
    # if config.pretrained:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16,
    #         trust_remote_code=True
    #     )
    # else:
    #     config_model = AutoConfig.from_pretrained(model_name)          
    #     model = AutoModelForCausalLM.from_config(
    #         config_model,
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True
    #     )
    
    
    return model



def load_qwen(config):
    """Загрузка предобученной Qwen с хуками для Accelerate"""
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Загрузка предобученной {model_name}")
    print(f"Доступно GPU: {torch.cuda.device_count()}")
    
    if config.load_checkpoint:
        # return model
        accelerator = Accelerator(
            mixed_precision='bf16'
        )
    
        # Сначала загружаем модель
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Подготавливаем модель с accelerator
        model = accelerator.prepare(model)
        
        # Загружаем полное состояние
        
        # checkpoint_path = "finetune_checkpoints_large/best_checkpoint/best_loss_3.0712"
        checkpoint_path = "finetune_checkpoints_FFN/best_checkpoint/best_loss_3.5091"
        
        accelerator.load_state(checkpoint_path)
        
        print(f"Чекпоинт загружен из {checkpoint_path}")
        
        # Получаем неподготовленную модель для использования
        model = accelerator.unwrap_model(model)

        if config.remove_ffn:
            model = zero_out_ffn_qwen(model)

        trainable_params = sum(p.numel() for p in model.parameters() 
                              if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Обучаемые параметры с LoRA: {trainable_params:,} "
              f"({100*trainable_params/total_params:.2f}% от {total_params:,})")
        
        return model
        
    if config.pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        if config.remove_ffn:
            model = zero_out_ffn_qwen(model)
            print(f"\nМодель {model_name} загружена с отключенными FFN слоями!")

        if config.remove_norm:
            model = disable_qwen_normalization(model)
            print(f"\nМодель {model_name} загружена с отключенными нормализациями!")
            
    else:
        if config.remove_ffn: # Attention-only
            config_model = AutoConfig.from_pretrained(
                "Qwen/Qwen2.5-1.5B",
                trust_remote_code=True
            )
            #config.initializer_range = 0.05
            # Создаем модель с архитектурой из конфигурации, но без предобученных весов
            model = AutoModelForCausalLM.from_config(
                config_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model = zero_out_ffn_qwen(model)
            print(f"\nМодель {model_name} загружена с отключенными FFN слоями!")

        else: # обычная модель
            config_model = AutoConfig.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True
            )
            # Создаем модель с архитектурой из конфигурации, но без предобученных весов
            model = AutoModelForCausalLM.from_config(
                config_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            if config.remove_norm:
                model = disable_qwen_normalization(model)
                print(f"\nМодель {model_name} загружена с отключенными нормализациями!")
    
    if config.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() 
                              if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Обучаемые параметры с LoRA: {trainable_params:,} "
              f"({100*trainable_params/total_params:.2f}% от {total_params:,})")
    
    return model
# ===========================================



def compute_val_loss(model, config, val_texts, device):
    """Вычисление loss на валидационной выборке"""
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_texts), config.batch_size):
            if i + config.batch_size > len(val_texts):
                break

            batch_samples = val_texts[i:i + config.batch_size]
            batch = prepare_batch(batch_samples)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Прямой проход
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_val_loss += loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
    return avg_val_loss