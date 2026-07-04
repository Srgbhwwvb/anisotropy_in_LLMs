import torch

def zero_out_ffn_weights_gpt2(model):
    """Обнуляет и замораживает веса FFN слоев GPT2"""
    print("Обнуление и заморозка весов FFN слоев GPT2...")
    
    for layer_idx, layer in enumerate(model.transformer.h):
        # Проверяем наличие MLP атрибута
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # Первый линейный слой (расширение)
            if hasattr(mlp, 'c_fc'):
                mlp.c_fc.weight.data.zero_()
                mlp.c_fc.weight.requires_grad = False  # ЗАМОРОЗКА
                
                if hasattr(mlp.c_fc, 'bias') and mlp.c_fc.bias is not None:
                    mlp.c_fc.bias.data.zero_()
                    mlp.c_fc.bias.requires_grad = False  # ЗАМОРОЗКА
            
            # Второй линейный слой (проекция)
            if hasattr(mlp, 'c_proj'):
                mlp.c_proj.weight.data.zero_()
                mlp.c_proj.weight.requires_grad = False  # ЗАМОРОЗКА
                
                if hasattr(mlp.c_proj, 'bias') and mlp.c_proj.bias is not None:
                    mlp.c_proj.bias.data.zero_()
                    mlp.c_proj.bias.requires_grad = False  # ЗАМОРОЗКА
            
            #print(f"  Обнулены и заморожены веса FFN в слое {layer_idx}")
        else:
            print(f"  ВНИМАНИЕ: В слое {layer_idx} нет атрибута 'mlp'")
    
    return model

# def zero_out_ffn_qwen(model):
#     """Наиболее эффективное удаление FFN - замена на пустой модуль"""
#     print("Эффективное удаление FFN слоев")
    
#     class EmptyModule(nn.Module):
#         def __init__(self):
#             super().__init__()
            
#         def forward(self, x):
#             # Возвращаем нули, чтобы не нарушать residual connection
#             # Но с учетом, что в Qwen обычно: x = x + attention(x) + mlp(x) (если параллельный)
#             # или x = x + mlp(ln(x)) (если последовательный)
#             # Возвращаем нули той же формы
#             return torch.zeros_like(x)
    
#     for layer in model.model.layers:
#         if hasattr(layer, 'mlp'):
#             # Создаем пустой модуль без параметров
#             empty_mlp = EmptyModule()
            
#             # Заменяем оригинальный mlp
#             layer.mlp = empty_mlp
            
#             # Также можно удалить ссылки на параметры
#             if hasattr(layer, 'post_attention_layernorm'):
#                 # В Qwen2.5 FFN обычно идет после layernorm
#                 # Сохраняем оригинальный forward
#                 original_forward = layer.forward
                
#                 def new_forward(hidden_states, *args, **kwargs):
#                     # Пропускаем FFN, но выполняем layernorm если нужно
#                     residual = hidden_states
#                     hidden_states = layer.post_attention_layernorm(hidden_states)
#                     # Вместо mlp возвращаем нули
#                     mlp_output = torch.zeros_like(hidden_states)
#                     hidden_states = residual + mlp_output
#                     return (hidden_states,)
                
#                 layer.forward = new_forward.__get__(layer, type(layer))
    
#     # Принудительно очищаем память
#     import gc
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Обучаемые параметры: {trainable_params:,} из {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
#     return model


    
def zero_out_ffn_qwen(model):
    """Обнуляет и замораживает веса FFN слоев для Qwen2.5"""
    print("Обнуление и заморозка FFN слоев")
    layers = model.model.layers
    
    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # Для Qwen2.5 - обнуляем и замораживаем
            if hasattr(mlp, 'gate_proj'):
                mlp.gate_proj.weight.data.zero_()
                mlp.gate_proj.weight.requires_grad = False
                if mlp.gate_proj.bias is not None:
                    mlp.gate_proj.bias.data.zero_()
                    mlp.gate_proj.bias.requires_grad = False
            
            if hasattr(mlp, 'up_proj'):
                mlp.up_proj.weight.data.zero_()
                mlp.up_proj.weight.requires_grad = False
                if mlp.up_proj.bias is not None:
                    mlp.up_proj.bias.data.zero_()
                    mlp.up_proj.bias.requires_grad = False
            
            if hasattr(mlp, 'down_proj'):
                mlp.down_proj.weight.data.zero_()
                mlp.down_proj.weight.requires_grad = False
                if mlp.down_proj.bias is not None:
                    mlp.down_proj.bias.data.zero_()
                    mlp.down_proj.bias.requires_grad = False
    
    # Проверяем какие параметры обучаются
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Обучаемые параметры: {trainable_params:,} из {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model


import torch
import torch.nn as nn

class ResidualScaling(nn.Module):
    """Масштабирование residual connections (только для обучаемых слоев)"""
    def __init__(self, dim, scale=0.1):
        super().__init__()
        # Создаем обучаемый параметр
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        
    def forward(self, x, residual):
        return x + self.scale * residual


def apply_residual_scaling(model):
    """
    Применяет residual scaling ТОЛЬКО к обучаемым слоям
    
    Args:
        model: модель для модификации
    """
    applied_count = 0
    
    for name, module in model.named_modules():
        # Пропускаем полностью замороженные модули
        has_trainable_params = any(p.requires_grad for p in module.parameters())
        if not has_trainable_params:
            continue
        
        # Добавляем scaling к attention и MLP слоям
        if 'attention' in name and hasattr(module, 'out_proj'):
            dim = module.out_proj.out_features
            setattr(module, 'residual_scaling', ResidualScaling(dim))
            applied_count += 1
            print(f"✓ ResidualScaling добавлен к {name}")
            
        elif 'mlp' in name and hasattr(module, 'fc2'):
            dim = module.fc2.out_features
            setattr(module, 'residual_scaling', ResidualScaling(dim))
            applied_count += 1
            print(f"✓ ResidualScaling добавлен к {name}")
    
    print(f"\nИтог: ResidualScaling применен к {applied_count} обучаемым слоям")
    return model

# def check_model_weights(model):
#     """Проверяет, действительно ли обнулены веса FFN"""
#     total_zero_weights = 0
#     total_weights = 0
    
#     for layer_idx, layer in enumerate(model.model.layers):
#         if hasattr(layer, 'mlp'):
#             mlp = layer.mlp
#             for name, module in mlp.named_children():
#                 if isinstance(module, torch.nn.Linear):
#                     zero_count = (module.weight == 0).sum().item()
#                     total_zero_weights += zero_count
#                     total_weights += module.weight.numel()
                    
#                     if module.bias is not None:
#                         zero_bias = (module.bias == 0).sum().item()
#                         total_zero_weights += zero_bias
#                         total_weights += module.bias.numel()
                    
#                     print(f"Слой {layer_idx}, {name}: {zero_count}/{module.weight.numel()} нулевых весов")
    
#     print(f"\nИтого: {total_zero_weights}/{total_weights} нулевых параметров "
#           f"({100*total_zero_weights/total_weights:.1f}%)")
    
# def zero_out_ffn_weights_qwen(model):
#     """Обнуляет веса FFN слоев для Qwen1.5"""
    
#     layers = model.model.layers
    
#     for layer_idx, layer in enumerate(layers):
#         if hasattr(layer, 'mlp'):
#             mlp = layer.mlp
            
#             # Первый линейный слой (расширение)
#             if hasattr(mlp, 'c_fc'):
#                 mlp.c_fc.weight.data.zero_()
#                 if hasattr(mlp.c_fc, 'bias') and mlp.c_fc.bias is not None:
#                     mlp.c_fc.bias.data.zero_()
#             elif hasattr(mlp, 'gate_proj'):
#                 mlp.gate_proj.weight.data.zero_()
#                 if hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
#                     mlp.gate_proj.bias.data.zero_()
#             elif hasattr(mlp, 'w1'):
#                 mlp.w1.weight.data.zero_()
#                 if hasattr(mlp.w1, 'bias') and mlp.w1.bias is not None:
#                     mlp.w1.bias.data.zero_()
            
#             # Второй линейный слой (проекция)
#             if hasattr(mlp, 'c_proj'):
#                 mlp.c_proj.weight.data.zero_()
#                 if hasattr(mlp.c_proj, 'bias') and mlp.c_proj.bias is not None:
#                     mlp.c_proj.bias.data.zero_()
#             elif hasattr(mlp, 'up_proj'):
#                 mlp.up_proj.weight.data.zero_()
#                 if hasattr(mlp.up_proj, 'bias') and mlp.up_proj.bias is not None:
#                     mlp.up_proj.bias.data.zero_()
#             elif hasattr(mlp, 'w2'):
#                 mlp.w2.weight.data.zero_()
#                 if hasattr(mlp.w2, 'bias') and mlp.w2.bias is not None:
#                     mlp.w2.bias.data.zero_()
    
#     return model



def verify_ffn_disabled(model):
    """Проверяет, что FFN слои отключены (веса обнулены)"""
    total_params = 0
    active_ffn_params = 0
    
    for layer_idx, layer in enumerate(model.transformer.h):
        if hasattr(layer, 'mlp'):
            for name, param in layer.mlp.named_parameters():
                total_params += param.numel()
                # Проверяем, не все ли веса нулевые
                if torch.norm(param).item() > 1e-6:
                    active_ffn_params += param.numel()
                    print(f"  Предупреждение: ненулевые веса в слое {layer_idx}, {name}")
    
    print(f"\nПроверка FFN слоев:")
    print(f"  Всего параметров в FFN: {total_params:,}")
    print(f"  Активных параметров в FFN: {active_ffn_params:,}")
    print(f"  Процент отключенных FFN: {(1 - active_ffn_params/max(total_params,1)) * 100:.2f}%")
    
    return active_ffn_params == 0
