import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from save import setup_plot_saving, save_checkpoint_simple, manage_last_checkpoints
from intrinsic import intrinsic_dimension_levina_bickel_vectorized, compute_intrinsic_dim_per_sample, measure_intrinsic_dimension_per_layer, plot_layerwise_intrinsic_dim_distribution


# Генерация LaTeX-таблицы со статистиками по компонентам
def generate_latex_table(metrics, num_layers, model, save_plots=False, plot_dir=None):
    # Определим соответствие точек компонентам
    component_points = {
        'Attention': ['attn'],
        'MLP': ['mlp'],
        'LayerNorm': ['ln1', 'ln2'],
        'Residuals': ['res1', 'res2']
    }
    
    # Собираем для каждого слоя (0..num_layers-1) значения
    layer_data = {comp: [] for comp in component_points}
    
    for layer in range(num_layers):
        for comp, points in component_points.items():
            values_cos = []
            values_aniso = []
            for point in points:
                key = (layer, point)
                if key in metrics:
                    values_cos.append(metrics[key]['cosine_mean'])
                    values_aniso.append(metrics[key]['anisotropy'])
            if values_cos and values_aniso:
                # Усредняем по точкам внутри слоя (для LayerNorm и Residuals)
                layer_data[comp].append({
                    'cos': sum(values_cos)/len(values_cos),
                    'aniso': sum(values_aniso)/len(values_aniso)
                })
    
    # Вычисляем статистики для каждой компоненты
    stats = {}
    for comp, data in layer_data.items():
        if not data:
            continue
        cos_vals = [d['cos'] for d in data]
        aniso_vals = [d['aniso'] for d in data]
        stats[comp] = {
            'max_cos': max(cos_vals), 'min_cos': min(cos_vals), 'mean_cos': sum(cos_vals)/len(cos_vals),
            'max_aniso': max(aniso_vals), 'min_aniso': min(aniso_vals), 'mean_aniso': sum(aniso_vals)/len(aniso_vals),
        }
        stats[comp]['combined'] = (stats[comp]['mean_cos'] + stats[comp]['mean_aniso']) / 2
    
    # Определяем, какие значения выделять жирным
    # Для каждого столбца вычисляем глобальный максимум (кроме min_aniso – минимум)
    all_max_cos = max(s['max_cos'] for s in stats.values())
    all_min_cos = max(s['min_cos'] for s in stats.values())   # как в примере – максимум среди минимумов
    all_mean_cos = max(s['mean_cos'] for s in stats.values())
    all_max_aniso = max(s['max_aniso'] for s in stats.values())
    all_min_aniso = min(s['min_aniso'] for s in stats.values())  # минимум среди минимумов
    all_mean_aniso = max(s['mean_aniso'] for s in stats.values())
    all_combined = max(s['combined'] for s in stats.values())
    
    # Формируем строки таблицы
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Anisotropy statistics for " + model.config._name_or_path + "}")
    lines.append("\\label{tab:metrics_" + model.config._name_or_path.replace('-', '_') + "}")
    lines.append("\\begin{tabular}{lccc|ccc|c}")
    lines.append("\\toprule")
    lines.append("& max $A_{c}$ & min $A_{c}$ & $\\bar{A}_{c}$ & max $A_{s}$& min $A_{s}$& $\\bar{A}_{s}$& $\\frac{\\bar{A}_{c}+\\bar{A}_{s}}{2}$ \\\\")
    lines.append("\\midrule")
    
    for comp in ['Attention', 'MLP', 'LayerNorm', 'Residuals']:
        if comp not in stats:
            continue
        s = stats[comp]
        row = comp + " & "
        # max_cos
        if s['max_cos'] == all_max_cos:
            row += f"$\\mathbf{{{s['max_cos']:.2f}}}$ & "
        else:
            row += f"{s['max_cos']:.2f} & "
        # min_cos
        if s['min_cos'] == all_min_cos:
            row += f"$\\mathbf{{{s['min_cos']:.2f}}}$ & "
        else:
            row += f"{s['min_cos']:.2f} & "
        # mean_cos
        if s['mean_cos'] == all_mean_cos:
            row += f"$\\mathbf{{{s['mean_cos']:.2f}}}$ & "
        else:
            row += f"{s['mean_cos']:.2f} & "
        # max_aniso
        if s['max_aniso'] == all_max_aniso:
            row += f"$\\mathbf{{{s['max_aniso']:.2f}}}$ & "
        else:
            row += f"{s['max_aniso']:.2f} & "
        # min_aniso
        if s['min_aniso'] == all_min_aniso:
            row += f"$\\mathbf{{{s['min_aniso']:.2f}}}$ & "
        else:
            row += f"{s['min_aniso']:.2f} & "
        # mean_aniso
        if s['mean_aniso'] == all_mean_aniso:
            row += f"$\\mathbf{{{s['mean_aniso']:.2f}}}$ & "
        else:
            row += f"{s['mean_aniso']:.2f} & "
        # combined
        if s['combined'] == all_combined:
            row += f"$\\mathbf{{{s['combined']:.2f}}}$ \\\\"
        else:
            row += f"{s['combined']:.2f} \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    table_str = "\n".join(lines)
    print("\n" + "="*60)
    print("LaTeX-таблица со статистиками сингулярной анизотропии и косинусного сходства:")
    print("="*60)
    print(table_str)
    
    if save_plots and plot_dir:
        from pathlib import Path
        with open(Path(plot_dir) / "anisotropy_table.tex", "w") as f:
            f.write(table_str)
        print(f"\nТаблица сохранена в {plot_dir}/anisotropy_table.tex")





def compute_isoscore(embeddings):
    """
    Вычисляет IsoScore для центрированного облака точек.
    Parameters
    -
    embeddings : np.ndarray, shape (N, n)
        Центрированные данные.
    Returns
    -
    isoscore : float
        Значение IsoScore в [0, 1].
    """
    N, n = embeddings.shape
    U, s, Vt = np.linalg.svd(embeddings.cpu().float().numpy(), full_matrices=False)
    # Дисперсии вдоль главных компонент (несмещённая оценка)
    k = len(s)
    variances = (s ** 2) / (N - 1)
    # Дополняем вектор дисперсий нулями до размерности n
    sigma_D = np.zeros(n)
    sigma_D[:k] = variances

    norm_sd = np.linalg.norm(sigma_D)
    sigma_D_hat = np.sqrt(n) * sigma_D / norm_sd

    # isotropy defect
    one_vector = np.ones(n)
    numerator = np.linalg.norm(sigma_D_hat - one_vector)
    denominator = np.sqrt(2 * (n - np.sqrt(n)))
    delta = numerator / denominator

    # доля равномерно используемых размерностей
    phi = ((n - delta**2 * (n - np.sqrt(n))) ** 2) / (n ** 2)

    # преобразование в IsoScore
    isoscore = (n * phi - 1) / (n - 1)
    return isoscore



def analyze_componentwise_anisotropy_qwen(model, batch, config, layers_to_plot=None, save_plots=False):
    """
    Расширенный анализ косинусной близости и сингулярной анизотропии для моделей Qwen2.5.
    Собирает скрытые состояния на 7 этапах внутри каждого decoder layer:
        - вход в слой (input)
        - после input_layernorm (ln1)
        - после self_attention (attn)
        - после первого residual (res1)
        - после post_attention_layernorm (ln2)
        - после MLP (mlp)
        - после второго residual / выход слоя (res2)
    Для каждого этапа вычисляет:
        - среднее попарное косинусное сходство (и std)
        - анизотропию = (σ₁²) / (Σ σᵢ²), где σᵢ – сингулярные числа матрицы состояний
    Строит два графика: среднее косинусное сходство и анизотропия по всем этапам всех слоёв.
    Дополнительно добавляет два финальных этапа:
        - после final_layernorm (перед lm_head)
        - после lm_head (логиты)
    
    Также вычисляются метрики на основе норм токенов:
        - Доля примеров, в которых норма первого токена максимальна.
        - Отношение нормы первого токена к сумме норм всех токенов.
        - Отношение максимальной нормы к сумме остальных норм.
    Для корректной работы требуется, чтобы в данных присутствовал BOS токен и его id был указан в config.
    """
   
    # Определяем устройство и перемещаем batch
    model_device = next(model.parameters()).device
    print("model_device", model_device)
    if batch['input_ids'].device != model_device:
        print(f"Перемещаем данные с {batch['input_ids'].device} на {model_device}")
        batch = {k: v.to(model_device) for k, v in batch.items()}
        print(f"Модель на устройстве: {next(model.parameters()).device}")
        print(f"Batch на устройстве: {batch['input_ids'].device}")

    # Pad token id для Qwen2.5 
    if config.model_name == "Qwen":
        pad_token_id = 151643  
    elif config.model_name == "GPT2":
        pad_token_id = 50256
    else:
        pad_token_id = 0

    # BOS token id 
    bos_token_id = getattr(config, 'bos_token_id', None)
    if bos_token_id is None and hasattr(model, 'config'):
        bos_token_id = getattr(model.config, 'bos_token_id', None)
    if bos_token_id is None:
        print("Предупреждение: bos_token_id не задан. Метрики, связанные с первым токеном, будут считаться для первого токена как есть.")
        # В этом случае будем считать метрики для первого токена без проверки на BOS
    else:
        print(f"Используется bos_token_id = {bos_token_id}")

    # Определяем список анализируемых слоёв
    if layers_to_plot is None:
        layers_to_plot = list(range(model.config.num_hidden_layers+1))  

    # Директория для сохранения графиков
    plot_dir = setup_plot_saving(config, suffix="cosine_analysis")
    print(f"Графики будут сохранены в: {plot_dir}")

    # Определяем точки внутри слоя и их сокращения
    POINTS = ['input','ln1', 'attn', 'res1', 'ln2', 'mlp', 'res2']
    POINT_ABBR = {
        'input': 'input',
        'ln1': 'rmsnorm 1',
        'attn': 'attention',
        'res1': 'residual 1',
        'ln2': 'rmsnorm 2',
        'mlp': 'fnn',
        'res2': 'residual 2'
    }

    # Контейнер для результатов 
    # Ключ: (layer_idx, point_name), значение: dict с метриками
    metrics = {}
    norm_metrics = {}   # для метрик на основе норм токенов

    # Функция обработки одного тензора (на GPU) 
    def compute_metrics(tensor_hidden, input_ids_flat, pad_token_id, num_samples):
        """
        tensor_hidden: (batch*seq_len, hidden_dim) на GPU
        input_ids_flat: (batch*seq_len,) на GPU
        Возвращает dict с метриками: cosine_mean, cosine_std, anisotropy, participation_ratio
        """
        # Маска валидных токенов
        valid_mask = (input_ids_flat != pad_token_id)  # на GPU
        embeddings = tensor_hidden[valid_mask]         # [N_valid, hidden_dim]
    
        if embeddings.shape[0] == 0:
            return None
    
        # Подвыборка
        if embeddings.shape[0] > num_samples:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:num_samples]
            embeddings = embeddings[idx]
    
        # косинусное сходство
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / torch.clamp(norms, min=1e-8)
        cos_matrix = normalized @ normalized.T
        n = cos_matrix.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1, device=cos_matrix.device)
        cos_values = cos_matrix[triu_indices[0], triu_indices[1]]
        cosine_mean = cos_values.mean().item()
        cosine_std = cos_values.std().item()
    
        # анизотропия и participation ratio через SVD 
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(embeddings.float())   # сингулярные числа
        s_sq = s ** 2
        sum_sq = s_sq.sum()
        sum_sq_sq = (s_sq ** 2).sum()
        anisotropy = (s_sq[0] / sum_sq).item()
        participation_ratio = (sum_sq ** 2 / sum_sq_sq).item() if sum_sq_sq > 0 else 0.0
        rank = (s > 1e-7).sum().item()
    
        return {
            'cosine_mean': cosine_mean,
            'cosine_std': cosine_std,
            'anisotropy': anisotropy,
            'participation_ratio': participation_ratio,
            'rank': rank,
            'max_sing': s[0].cpu().item() if s.numel() > 0 else 0.0
        }

    # функция для вычисления метрик на основе норм токенов 
    def compute_norm_metrics(hidden_states, attention_mask, input_ids, bos_token_id):
        """
        Возвращает словарь с ключами:
            - 'bos_is_max_ratio'  (доля примеров, где норма первого токена максимальна)
            - 'bos_norm_ratio'    (отношение нормы первого токена к сумме всех норм)
            - 'max_norm_to_sum_others_ratio'
            - 'first_norm_abs'    (абсолютная норма первого токена)
            - 'mean_norm'         (средняя норма всех токенов)
        """
        norms = torch.norm(hidden_states, dim=-1)
        mask = attention_mask.bool()
        batch_size = hidden_states.size(0)
    
        first_is_max_list = []
        first_norm_ratio_list = []
        first_norm_abs_list = []
        max_to_sum_ratio_list = []
        mean_norm_list = []
    
        for i in range(batch_size):
            sample_norms = norms[i]
            sample_mask = mask[i]
            valid_norms = sample_norms[sample_mask]
            if valid_norms.numel() == 0:
                continue
    
            mean_norm_list.append(valid_norms.mean().item())
    
            if sample_mask[0].item():
                first_norm = sample_norms[0].item()
                first_norm_abs_list.append(first_norm)
                first_is_max = (first_norm == valid_norms.max().item())
                first_is_max_list.append(float(first_is_max))
                total_sum = valid_norms.sum().item()
                first_norm_ratio = first_norm / total_sum if total_sum > 0 else 0.0
                first_norm_ratio_list.append(first_norm_ratio)
    
            max_norm = valid_norms.max().item()
            sum_others = valid_norms.sum().item() - max_norm
            max_to_sum_ratio = max_norm / sum_others if sum_others > 0 else 0.0
            max_to_sum_ratio_list.append(max_to_sum_ratio)
    
        avg_first_is_max = sum(first_is_max_list) / len(first_is_max_list) if first_is_max_list else 0.0
        avg_first_norm_ratio = sum(first_norm_ratio_list) / len(first_norm_ratio_list) if first_norm_ratio_list else 0.0
        avg_first_norm_abs = sum(first_norm_abs_list) / len(first_norm_abs_list) if first_norm_abs_list else 0.0
        avg_max_to_sum_ratio = sum(max_to_sum_ratio_list) / len(max_to_sum_ratio_list) if max_to_sum_ratio_list else 0.0
        avg_mean_norm = sum(mean_norm_list) / len(mean_norm_list) if mean_norm_list else 0.0
    
        return {
            'bos_is_max_ratio': avg_first_is_max,
            'bos_norm_ratio': avg_first_norm_ratio,
            'max_norm_to_sum_others_ratio': avg_max_to_sum_ratio,
            'first_norm_abs': avg_first_norm_abs,
            'mean_norm': avg_mean_norm
        }
    # Регистрация хуков для каждого слоя 
    handles = []

    # Определяем, где находятся decoder layers
    if hasattr(model, 'layers') and isinstance(model.layers, torch.nn.ModuleList):
        layers_container = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_container = model.model.layers
    else:
        raise AttributeError("Не удалось найти decoder layers модели. Ожидается model.layers или model.model.layers")

    print(f"Найдено слоёв: {len(layers_container)}")
    print("=" * 60)

    # Плоский тензор input_ids для быстрой фильтрации (для compute_metrics)
    input_ids_flat = batch['input_ids'].view(-1)  # [batch*seq_len] на устройстве

    model.eval()
    with torch.no_grad():
        # Для каждого выбранного слоя регистрируем хуки
        for layer_idx in layers_to_plot:
            if layer_idx >= len(layers_container):
                print(f"Пропускаем слой {layer_idx}, так как его нет в модели")
                continue

            layer = layers_container[layer_idx]

            # Переменные для сохранения residual между хуками (замыкание)
            residual = None
            res1_state = None

            # Pre-hook на самом слое: сохраняем вход 
            def create_pre_hook(l_idx, point):
                def pre_hook(module, args):
                    nonlocal residual
                    residual = args[0]  # hidden_states
                    # Обработка и сохранение метрик для 'input'
                    hidden_flat = residual.view(-1, residual.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # нормные метрики 
                    norm_met = compute_norm_metrics(residual, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                    return args
                return pre_hook

            pre_handle = layer.register_forward_pre_hook(create_pre_hook(layer_idx, 'input'))
            handles.append(pre_handle)

            # Hook на input_layernorm: выход нормализации 
            def create_ln1_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # нормные метрики 
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook

            ln1_handle = layer.input_layernorm.register_forward_hook(create_ln1_hook(layer_idx, 'ln1'))
            handles.append(ln1_handle)

            # Hook на self_attn: выход attention и вычисление res1 
            def create_attn_hook(l_idx, point_attn, point_res1):
                def hook(module, input, output):
                    nonlocal residual, res1_state
                    # output[0] – скрытые состояния после attention
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # Сохраняем метрики для 'attn'
                    hidden_flat = attn_out.view(-1, attn_out.shape[-1])
                    met_attn = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met_attn is not None:
                        metrics[(l_idx, point_attn)] = met_attn
                    # нормные метрики для attention 
                    norm_met_attn = compute_norm_metrics(attn_out, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point_attn)] = norm_met_attn

                    # Вычисляем res1 = residual + attn_out
                    if residual is not None:
                        res1 = residual + attn_out
                        # Сохраняем метрики для 'res1'
                        res1_flat = res1.view(-1, res1.shape[-1])
                        met_res1 = compute_metrics(res1_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met_res1 is not None:
                            metrics[(l_idx, point_res1)] = met_res1
                        # нормные метрики для residual 1
                        norm_met_res1 = compute_norm_metrics(res1, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point_res1)] = norm_met_res1
                        # Запоминаем для следующего residual
                        res1_state = res1
                return hook

            attn_handle = layer.self_attn.register_forward_hook(
                create_attn_hook(layer_idx, 'attn', 'res1')
            )
            handles.append(attn_handle)

            # Hook на post_attention_layernorm: выход второй нормализации
            def create_ln2_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # нормные метрики 
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook

            ln2_handle = layer.post_attention_layernorm.register_forward_hook(
                create_ln2_hook(layer_idx, 'ln2')
            )
            handles.append(ln2_handle)

            # Hook на mlp: выход MLP и вычисление res2 
            def create_mlp_hook(l_idx, point_mlp, point_res2):
                def hook(module, input, output):
                    nonlocal res1_state
                    # output – скрытые состояния после MLP
                    mlp_out = output
                    # Сохраняем метрики для 'mlp'
                    hidden_flat = mlp_out.view(-1, mlp_out.shape[-1])
                    met_mlp = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met_mlp is not None:
                        metrics[(l_idx, point_mlp)] = met_mlp
                    # нормные метрики для MLP 
                    norm_met_mlp = compute_norm_metrics(mlp_out, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point_mlp)] = norm_met_mlp

                    # Вычисляем res2 = res1_state + mlp_out
                    if res1_state is not None:
                        res2 = res1_state + mlp_out
                        res2_flat = res2.view(-1, res2.shape[-1])
                        met_res2 = compute_metrics(res2_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met_res2 is not None:
                            metrics[(l_idx, point_res2)] = met_res2
                        # нормные метрики для residual 2
                        norm_met_res2 = compute_norm_metrics(res2, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point_res2)] = norm_met_res2
                return hook

            mlp_handle = layer.mlp.register_forward_hook(
                create_mlp_hook(layer_idx, 'mlp', 'res2')
            )
            handles.append(mlp_handle)

        # ДОБАВЛЯЕМ ХУКИ НА ФИНАЛЬНЫЕ СЛОИ (norm и lm_head) 
        # Ищем final_layernorm 
        final_norm = None
        if layers_to_plot[-1] == model.config.num_hidden_layers:
            if hasattr(model, 'norm'):
                final_norm = model.norm
            elif hasattr(model, 'model') and hasattr(model.model, 'norm'):
                final_norm = model.model.norm

        if final_norm is not None:
            def create_final_norm_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # нормные метрики для final norm
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook
            final_layer_idx = len(layers_container)
            norm_handle = final_norm.register_forward_hook(create_final_norm_hook(final_layer_idx, 'final_norm'))
            handles.append(norm_handle)
            print("Хук на final_layernorm зарегистрирован.")
        else:
            print("Предупреждение: не найден final_layernorm")

        # lm_head
        lm_head = None
        if layers_to_plot[-1] == model.config.num_hidden_layers:
            if hasattr(model, 'lm_head'):
                lm_head = model.lm_head
            elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
                lm_head = model.model.lm_head

        if lm_head is not None:
            def create_lm_head_hook(l_idx, point):
                def hook(module, input, output):
                    # output может быть tuple? обычно просто тензор логитов
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    hidden_flat = logits.view(-1, logits.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # нормные метрики для логитов 
                    norm_met = compute_norm_metrics(logits, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook
            # Для lm_head используем номер len(layers_container) + 1
            lm_head_idx = len(layers_container) + 1
            lm_head_handle = lm_head.register_forward_hook(create_lm_head_hook(lm_head_idx, 'lm_head'))
            handles.append(lm_head_handle)
            print("Хук на lm_head зарегистрирован.")
        else:
            print("Предупреждение: не найден lm_head")

        # Запуск одного forward pass 
        print("Запуск forward pass для сбора промежуточных состояний...")
        _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=False  # нам не нужны полные hidden states, работают хуки
        )
        print("Forward pass завершён.")

    # Удаление всех хуков 
    for h in handles:
        h.remove()
    print(f"Удалено {len(handles)} хуков.")

    # Построение графиков
    if not metrics:
        print("Не удалось собрать метрики. Проверьте модель и входные данные.")
        return {}

    # Подготовка данных для графиков:
    # Собираем все ключи и сортируем:
    #   - сначала все обычные слои (числовые индексы слоёв) в порядке возрастания
    #   - для каждого слоя добавляем точки в порядке POINTS
    #   - затем добавляем финальные точки (final_norm, lm_head) в заданном порядке
    # Получаем список ключей в нужном порядке отображения.
    layer_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] < len(layers_container)))
    # Добавим финальные слои, если они есть (они имеют индексы >= len(layers_container))
    final_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] >= len(layers_container)))

    all_points = []
    for layer in layer_numbers:
        for point in POINTS:
            # Добавляем input только для нулевого слоя
            if point == 'input' and layer != 0:
                continue
            key = (layer, point)
            if key in metrics:
                all_points.append(key)

    final_desired_order = ['final_norm', 'lm_head']
    for point_name in final_desired_order:
        for layer in final_numbers:
            key = (layer, point_name)
            if key in metrics:
                all_points.append(key)
                break

    
    # Подписи для оси X
    x_labels_all = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            if point == 'res2':
                x_labels_all.append(f"{layer}")
            else:
                x_labels_all.append("")
        else:
            if point == 'final_norm':
                x_labels_all.append("")
            elif point == 'lm_head':
                x_labels_all.append("")
            else:
                x_labels_all.append(f"Final {point}")
                        
    x_labels = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            x_labels.append(f"{layer} {POINT_ABBR.get(point, point)}")
        else:
            if point == 'final_norm':
                x_labels.append("Final LayerNorm")
            elif point == 'lm_head':
                x_labels.append("LM Head")
            else:
                x_labels.append(f"Final {point}")
    x_pos = np.arange(len(all_points))

    dashed_indices = []
    for i, (layer, point) in enumerate(all_points):
        if (layer == 0 and point == 'input') or point == 'res2' or point == 'lm_head':
            dashed_indices.append(i)

    # Построение исходных графиков (косинус, анизотропия и т.д.) 
    cosine_means = [metrics[k]['cosine_mean'] for k in all_points]
    #cosine_means_centered = [metrics[k]['cosine_mean_centered'] for k in all_points]
    anisotropies = [metrics[k]['anisotropy'] for k in all_points]
    participation_ratios = [metrics[k]['participation_ratio'] for k in all_points]
    ranks = [metrics[k]['rank'] for k in all_points]
    #isoscores = [metrics[k]['isoscore'] for k in all_points]
    #max_sings = [metrics[k]['max_sing'] for k in all_points]


    # Вычисление корреляций Пирсона 
    from scipy.stats import pearsonr

    # Находим общие точки, для которых есть оба типа метрик
    common_keys = sorted(set(metrics.keys()) & set(norm_metrics.keys()))
    print(f"Найдено общих точек для корреляций: {len(common_keys)}")

    if len(common_keys) > 1:
        # Формируем списки значений в одном порядке
        anisotropy_vals = []
        cosine_vals = []
        ratio_vals = []
        abs_vals = []
        for key in common_keys:
            anisotropy_vals.append(metrics[key]['anisotropy'])
            cosine_vals.append(metrics[key]['cosine_mean'])
            ratio_vals.append(norm_metrics[key]['bos_norm_ratio'])
            abs_vals.append(norm_metrics[key]['first_norm_abs'])

        # Анизотропия vs относительная норма первого токена
        corr_ratio_aniso, p_ratio_aniso = pearsonr(anisotropy_vals, ratio_vals)
        print(f"Корреляция анизотропии и относительной нормы первого токена: "
              f"r = {corr_ratio_aniso:.4f}, p-value = {p_ratio_aniso:.4e}")

        # Анизотропия vs абсолютная норма первого токена
        corr_abs_aniso, p_abs_aniso = pearsonr(anisotropy_vals, abs_vals)
        print(f"Корреляция анизотропии и абсолютной нормы первого токена: "
              f"r = {corr_abs_aniso:.4f}, p-value = {p_abs_aniso:.4e}")

        # Среднее косинусное сходство vs относительная норма первого токена
        corr_ratio_cos, p_ratio_cos = pearsonr(cosine_vals, ratio_vals)
        print(f"Корреляция среднего косинусного сходства и относительной нормы первого токена: "
              f"r = {corr_ratio_cos:.4f}, p-value = {p_ratio_cos:.4e}")

        # Среднее косинусное сходство vs абсолютная норма первого токена
        corr_abs_cos, p_abs_cos = pearsonr(cosine_vals, abs_vals)
        print(f"Корреляция среднего косинусного сходства и абсолютной нормы первого токена: "
              f"r = {corr_abs_cos:.4f}, p-value = {p_abs_cos:.4e}")
    else:
        print("Недостаточно общих точек для вычисления корреляций.")
        
  
    # Находим общие точки, для которых есть оба типа метрик
    common_keys = set(metrics.keys()) & set(norm_metrics.keys())
    print(f"Найдено общих точек для корреляции: {len(common_keys)}")

    if len(common_keys) > 1:
        anisotropy_vals = []
        first_norm_ratio_vals = []
        first_norm_abs_vals = []

        for key in sorted(common_keys):  # сортировка для порядка
            anisotropy_vals.append(metrics[key]['anisotropy'])
            first_norm_ratio_vals.append(norm_metrics[key]['bos_norm_ratio'])
            first_norm_abs_vals.append(norm_metrics[key]['first_norm_abs'])

        # Корреляция между анизотропией и относительной нормой первого токена
        corr_ratio, p_ratio = pearsonr(anisotropy_vals, first_norm_ratio_vals)
        print(f"Корреляция Пирсона между анизотропией и относительной нормой первого токена: "
              f"r = {corr_ratio:.4f}, p-value = {p_ratio:.4e}")

        # Корреляция между анизотропией и абсолютной нормой первого токена
        corr_abs, p_abs = pearsonr(anisotropy_vals, first_norm_abs_vals)
        print(f"Корреляция Пирсона между анизотропией и абсолютной нормой первого токена: "
              f"r = {corr_abs:.4f}, p-value = {p_abs:.4e}")
    else:
        print("Недостаточно общих точек для вычисления корреляции.")

    
    # График 1.1: среднее косинусное сходство
    print("cosine component-wise", cosine_means)
    
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, cosine_means, marker='o', linestyle='-', color='b')
    dashed_vals = [cosine_means[i] for i in dashed_indices]
    plt.plot(dashed_indices, dashed_vals, 'r--', marker='o', markersize=6, color="blue", linewidth=2, label='layer-wise cosine anisotropy')
    plt.xticks(x_pos, x_labels_all, fontsize=20)#, rotation=90
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity", fontsize=20)
    plt.title(f"Cosine similarity of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
    plt.show()


    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[61:61+54], cosine_means[61:61+54], marker='o', linestyle='-', color='b')
    plt.xticks(x_pos[61:61+54], x_labels[61:61+54], rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity", fontsize=20)
    plt.title(f"Cosine similarity of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
    plt.show()

    # График 1.2: среднее косинусное сходство centered
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, cosine_means_centered, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity centered", fontsize=20)
    plt.title(f"Cosine similarity of hidden states centered – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage_centered.png", dpi=150)
    plt.show()

    # IsoScore
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, isoscores, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("IsoScore", fontsize=20)
    plt.title(f"IsoScore of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "isoscore.png", dpi=150)
    plt.show()

    
    # Plot 2: anisotropy
    print("component-wise singular", anisotropies)
    
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, anisotropies, marker='s', linestyle='-', color='r')
    dashed_vals = [anisotropies[i] for i in dashed_indices]
    plt.plot(dashed_indices, dashed_vals, 'r--', marker='s', markersize=6, linewidth=2, label='layer-wise singular anisotropy')
    plt.xticks(x_pos, x_labels_all, fontsize=20)#, rotation=90
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Anisotropy (σ₁² / Σ σᵢ²)", fontsize=20)
    plt.title(f"Anisotropy of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
    plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:55], anisotropies[:55], marker='s', linestyle='-', color='r')
    plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Anisotropy (σ₁² / Σ σᵢ²)", fontsize=20)
    plt.title(f"Anisotropy of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
    plt.show()
        
    # Plot 3: effective dimensionality
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, participation_ratios, marker='^', linestyle='-', color='g')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.ylabel("Effective dimensionality", fontsize=20)
    plt.title(f"Effective dimensionality – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "participation_ratio_per_stage.png", dpi=150)
    plt.show()
    
    # Plot 4: rank
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, ranks, marker='d', linestyle='-', color='purple')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.ylabel("Rank (number of singular values > 1e-7)")
    plt.title(f"Rank of hidden states – {model.config._name_or_path}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "rank_per_stage.png", dpi=150)
    plt.show()
    
    # Plot 5: maximum singular value
    # plt.figure(figsize=(20, 8))
    # plt.plot(x_pos, max_sings, marker='d', linestyle='-', color='orange')
    # plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    # plt.ylabel("Maximum singular value", fontsize=20)
    # plt.title(f"Maximum singular value – {model.config._name_or_path}", fontsize=20)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # if save_plots:
    #     plt.savefig(Path(plot_dir) / "max_singular_per_stage.png", dpi=150)
    # plt.show()
    
    # plots for norm-based metrics 
    # Extract values for the same points as in all_points (check presence in norm_metrics)
    bos_is_max_values = [norm_metrics[k]['bos_is_max_ratio'] for k in all_points if k in norm_metrics]
    bos_norm_ratio_values = [norm_metrics[k]['bos_norm_ratio'] for k in all_points if k in norm_metrics]
    max_to_sum_values = [norm_metrics[k]['max_norm_to_sum_others_ratio'] for k in all_points if k in norm_metrics]
    
    # If some points are missing from norm_metrics, issue a warning but plot the available ones
    if len(bos_is_max_values) != len(all_points):
        print("Warning: norm metrics not collected for all points.")
    
    # Plot 6: fraction of examples where the BOS norm is maximum
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:len(bos_is_max_values)], bos_is_max_values, marker='o', linestyle='-', color='blue')
    plt.xticks(x_pos[:len(bos_is_max_values)], x_labels[:len(bos_is_max_values)], rotation=90)
    plt.ylabel("Fraction of examples where the first token norm is maximum")
    plt.title("Fraction of examples with maximum norm at BOS")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "bos_is_max_ratio.png", dpi=150)
    plt.show()
    
    # Plot 7: ratio of BOS norm to sum of all norms
    first_norm_abs_values = [norm_metrics[k]['first_norm_abs'] for k in all_points if k in norm_metrics]
    mean_norm_values = [norm_metrics[k]['mean_norm'] for k in all_points if k in norm_metrics]
    
    if len(first_norm_abs_values) > 0 and len(mean_norm_values) > 0:
        plt.figure(figsize=(20, 8))
        
        # Use the common length (assume lists have the same length)
        n = len(first_norm_abs_values)
        x_vals = x_pos[:n]
        x_labels_trim = x_labels[:n]
        
        plt.plot(x_vals, first_norm_abs_values, 
                 marker='s', linestyle='-', color='green', label='First token norm')
        plt.plot(x_vals, mean_norm_values, 
                 marker='.', linestyle='--', color='blue', alpha=0.7, label='Mean token norm')
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel("Token norm", fontsize=20)
        plt.xticks(x_vals, x_labels_trim, rotation=90, fontsize=20)
        plt.legend(fontsize=20)  
        plt.title("Absolute norm of the first token and mean token norm except first token", fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(plot_dir) / "first_norm_abs_with_mean.png", dpi=150)
        plt.show()
    else:
        print("Warning: insufficient data to plot first token norm and mean norm.")

        # Plot: отношение нормы первого токена к сумме всех норм 
    if len(bos_norm_ratio_values) > 0:
        print("relative norm of FT", bos_norm_ratio_values)
        n = len(bos_norm_ratio_values)
        plt.figure(figsize=(20, 8))
        plt.plot(x_pos[:n], bos_norm_ratio_values, marker='o', linestyle='-', color='purple', linewidth=2)
        plt.xticks(x_pos[:n], x_labels[:n], rotation=90, fontsize=20)
        plt.ylabel("First token norm / sum of all norms", fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.title("Relative norm of the first token", fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(plot_dir) / "first_norm_to_sum_ratio.png", dpi=150)
        plt.show()
    
    # Plot 8: ratio of maximum norm to the sum of the others
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:len(max_to_sum_values)], max_to_sum_values, marker='^', linestyle='-', color='red')
    plt.xticks(x_pos[:len(max_to_sum_values)], x_labels[:len(max_to_sum_values)], rotation=90)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("max(norm) / sum(other norms)")
    plt.title("Maximum norm dominance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "max_norm_to_sum_others.png", dpi=150)
    plt.show()

    # Вывод сводки в консоль 
    print("\n" + "=" * 60)
    print("Сводка метрик по этапам:")
    for layer, point in all_points:
        m = metrics[(layer, point)]
        layer_label = f"{layer}" if layer < len(layers_container) else "Final"
        print(f"{layer_label:<4} {point:10s}: cosine={m['cosine_mean']:.4f}±{m['cosine_std']:.4f}, "
              f"anisotropy={m['anisotropy']:.4f}")

    generate_latex_table(metrics, len(layers_container), model, save_plots=save_plots, plot_dir=plot_dir if save_plots else None)
    # Возвращаем собранные метрики для возможного дальнейшего использования
    return {'metrics': metrics, 'norm_metrics': norm_metrics}

    # Сохранение таблиц с результатами 
    try:
        import pandas as pd
        from pathlib import Path

        # Получаем общее число параметров модели
        total_params = sum(p.numel() for p in model.parameters())
        # Формируем читаемое представление (в миллиардах/миллионах)
        if total_params >= 1e9:
            params_str = f"{total_params/1e9:.1f}B"
        elif total_params >= 1e6:
            params_str = f"{total_params/1e6:.1f}M"
        else:
            params_str = str(total_params)

        base_filename = f"{config.model_name}_{params_str}_cosine_analysis"

        # Подробная таблица по точкам 
        detailed_data = []
        for key in all_points:
            layer, point = key
            if layer < len(layers_container):
                layer_label = str(layer)
            else:
                if point == 'final_norm':
                    layer_label = 'final_norm'
                elif point == 'lm_head':
                    layer_label = 'lm_head'
                else:
                    layer_label = 'final'
            m = metrics[key]
            detailed_data.append({
                'Stage': x_labels[len(detailed_data)],  # используем уже готовую подпись
                'Layer': layer_label,
                'Point': point,
                'Cosine_mean': m['cosine_mean'],
                'Cosine_std': m['cosine_std'],
                'Anisotropy': m['anisotropy'],
                'Participation_ratio': m['participation_ratio'],
                'Rank': m['rank'],
                'Max_singular': m['max_sing']
            })

        df_detailed = pd.DataFrame(detailed_data)

        # Сохраняем в Excel (два листа) и CSV
        excel_path = Path(plot_dir) / f"{base_filename}.xlsx"
        csv_path = Path(plot_dir) / f"{base_filename}.csv"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_detailed.to_excel(writer, sheet_name='Detailed', index=False)

            # Сводная таблица по типам операций 
            # Определяем категорию для каждой точки
            def get_category(point):
                if point == 'attn':
                    return 'attention'
                elif point == 'mlp':
                    return 'fnn'
                elif point in ['ln1', 'ln2', 'final_norm']:
                    return 'normalization'
                elif point in ['input', 'res1', 'res2']:
                    return 'residuals'
                else:
                    return 'other'  # например, lm_head

            # Добавим колонку категории в детальный DataFrame
            df_detailed['Category'] = df_detailed['Point'].apply(get_category)

            # Для сводной таблицы возьмём только основные категории (исключим 'other')
            categories_of_interest = ['attention', 'fnn', 'normalization', 'residuals']
            df_cat = df_detailed[df_detailed['Category'].isin(categories_of_interest)]

            # Метрики, для которых считаем статистики
            metrics_for_summary = ['Cosine_mean', 'Anisotropy', 'Participation_ratio', 'Rank', 'Max_singular']

            # Собираем строки сводной таблицы
            summary_rows = []
            for cat in categories_of_interest:
                cat_data = df_cat[df_cat['Category'] == cat]
                row = {'Category': cat}
                for met in metrics_for_summary:
                    values = cat_data[met].dropna()
                    if len(values) > 0:
                        row[f'{met}_max'] = values.max()
                        row[f'{met}_min'] = values.min()
                        row[f'{met}_mean'] = values.mean()
                    else:
                        row[f'{met}_max'] = row[f'{met}_min'] = row[f'{met}_mean'] = None
                summary_rows.append(row)

            df_summary = pd.DataFrame(summary_rows)

            # Сохраняем сводную таблицу на отдельном листе
            df_summary.to_excel(writer, sheet_name='Summary_by_type', index=False)

        # Также сохраняем CSV с подробной таблицей
        df_detailed.to_csv(csv_path, index=False)

        print(f"\nТаблицы сохранены:")
        print(f"  - Excel: {excel_path}")
        print(f"  - CSV: {csv_path}")

    except ImportError:
        print("\nБиблиотека pandas не установлена. Таблицы не сохранены.")
    except Exception as e:
        print(f"\nОшибка при сохранении таблиц: {e}")

    return metrics

  


def analyze_componentwise_anisotropy_pythia(model, batch, config, compute_isoscore=False, layers_to_plot=None, save_plots=False, ALL=False):
    """
    Расширенный анализ косинусной близости и анизотропии для моделей Pythia (GPT-NeoX).
    Собирает скрытые состояния на 7 этапах внутри каждого decoder layer:
        - вход в слой (input)
        - после input_layernorm (ln1)
        - после attention (attn)
        - после первого residual (res1)
        - после post_attention_layernorm (ln2)
        - после MLP (mlp)
        - после второго residual / выход слоя (res2)
    Для каждого этапа вычисляет:
        - среднее попарное косинусное сходство (и std)
        - анизотропию = (σ₁²) / (Σ σᵢ²), где σᵢ – сингулярные числа
    Строит два графика: среднее косинусное сходство и анизотропия по всем этапам.
    Дополнительно добавляет два финальных этапа:
        - после final_layer_norm (перед lm_head)
        - после lm_head (логиты)
    
    Также вычисляются метрики на основе норм токенов:
        - Доля примеров, в которых норма BOS максимальна среди всех не‑паддинг токенов.
        - Отношение нормы BOS к сумме норм всех токенов (усреднённое по примерам).
        - Отношение максимальной нормы к сумме остальных норм (усреднённое по примерам).
    Для корректности проверяется, что первый токен действительно является BOS (используется bos_token_id из config).
    """

    # Устройство и данные 
    model_device = next(model.parameters()).device
    print("model_device", model_device)
    if batch['input_ids'].device != model_device:
        print(f"Перемещаем данные с {batch['input_ids'].device} на {model_device}")
        batch = {k: v.to(model_device) for k, v in batch.items()}

    #  Pad token id для Pythia (обычно eos_token_id = 0) 
    pad_token_id = getattr(config, 'eos_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0
    print(f"Используется pad_token_id = {pad_token_id}")

    #  BOS token id для проверки первого токена 
    bos_token_id = getattr(config, 'bos_token_id', getattr(config, 'eos_token_id', 0))
    if bos_token_id is None:
        bos_token_id = 0
    print(f"Используется bos_token_id = {bos_token_id}")

    #  Определяем список слоёв для анализа 
    if layers_to_plot is None:
        layers_to_plot = list(range(0, model.config.num_hidden_layers+1,1))  # все слои + финальные

    # Директория для сохранения графиков
    if save_plots:
        plot_dir = setup_plot_saving(config, suffix="cosine_analysis_pythia")
        print(f"Графики будут сохранены в: {plot_dir}")

    #  Точки внутри слоя и их подписи 
    POINTS = ['input','ln1', 'attn', 'res1', 'ln2', 'mlp', 'res2']
    POINT_ABBR = {
        'input': 'input',
        'ln1': 'LayerNorm 1',
        'attn': 'attention',
        'res1': 'residual 1',
        'ln2': 'LayerNorm 2',
        'mlp': 'MLP',
        'res2': 'residual 2'
    }

    #  Контейнеры для результатов 
    metrics = {}          # для косинусных и SVD метрик
    norm_metrics = {}     # для метрик на основе норм токенов

    #  Вспомогательная функция вычисления исходных метрик (косинус, анизотропия) 
    def compute_metrics(tensor_hidden, input_ids_flat, pad_token_id, num_samples):
        """
        tensor_hidden: (batch*seq_len, hidden_dim) на GPU
        input_ids_flat: (batch*seq_len,) на GPU
        """
        
        valid_mask = (input_ids_flat != pad_token_id)
        embeddings = tensor_hidden[valid_mask]

        if embeddings.shape[0] == 0:
            return None

        # Подвыборка, если слишком много токенов
        if embeddings.shape[0] > num_samples:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:num_samples]
            embeddings = embeddings[idx]

        #  Косинусное сходство 
        mean = embeddings.mean(dim=0, keepdim=True)   #           !!!!!
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / torch.clamp(norms, min=1e-8)
        cos_matrix = normalized @ normalized.T
        n = cos_matrix.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1, device=cos_matrix.device)
        cos_values = cos_matrix[triu_indices[0], triu_indices[1]]
        cosine_mean = cos_values.mean().item()
        cosine_std = cos_values.std().item()

        if compute_isoscore:
            isoscore_val = compute_isoscore(embeddings)
        else:
            isoscore_val = 0

        #  Косинусное сходство centered
        mean = embeddings.mean(dim=0, keepdim=True)   
        embeddings = embeddings - mean 
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / torch.clamp(norms, min=1e-8)
        cos_matrix = normalized @ normalized.T
        n = cos_matrix.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1, device=cos_matrix.device)
        cos_values = cos_matrix[triu_indices[0], triu_indices[1]]
        cosine_mean_centered = cos_values.mean().item()
        cosine_std_centered = cos_values.std().item()

        #  Анизотропия и участие (participation ratio) через SVD 
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True) 
        s = torch.linalg.svdvals(embeddings.float())   # сингулярные числа
        s_sq = s ** 2
        sum_sq = s_sq.sum()
        sum_sq_sq = (s_sq ** 2).sum()
        anisotropy = (s_sq[0] / sum_sq).item() if sum_sq > 0 else 0.0
        participation_ratio = (sum_sq ** 2 / sum_sq_sq).item() if sum_sq_sq > 0 else 0.0
        rank = (s > 1e-7).sum().item()

        return {
            'cosine_mean': cosine_mean,
            'cosine_std': cosine_std,
            'cosine_mean_centered': cosine_mean_centered,
            'cosine_std_centered': cosine_std_centered,
            'isoscore': isoscore_val,
            'anisotropy': anisotropy,
            'participation_ratio': participation_ratio,
            'rank': rank,
            'max_sing': s[0].cpu().item() if s.numel() > 0 else 0.0
        }

    #  Новая функция для вычисления метрик на основе норм токенов 
    def compute_norm_metrics(hidden_states, attention_mask, input_ids, bos_token_id):
        """
        Возвращает словарь с ключами:
            - 'bos_is_max_ratio'  (доля примеров, где норма первого токена максимальна)
            - 'bos_norm_ratio'    (отношение нормы первого токена к сумме всех норм)
            - 'max_norm_to_sum_others_ratio'
            - 'first_norm_abs'    (абсолютная норма первого токена)
            - 'mean_norm'         (средняя норма всех токенов)
        """
        norms = torch.norm(hidden_states, dim=-1)
        mask = attention_mask.bool()
        batch_size = hidden_states.size(0)
    
        first_is_max_list = []
        first_norm_ratio_list = []
        first_norm_abs_list = []
        max_to_sum_ratio_list = []
        mean_norm_list = []
    
        for i in range(batch_size):
            sample_norms = norms[i]
            sample_mask = mask[i]
            valid_norms = sample_norms[sample_mask]
            if valid_norms.numel() == 0:
                continue
    
            mean_norm_list.append(valid_norms.mean().item())
    
            if sample_mask[0].item():
                first_norm = sample_norms[0].item()
                first_norm_abs_list.append(first_norm)
                first_is_max = (first_norm == valid_norms.max().item())
                first_is_max_list.append(float(first_is_max))
                total_sum = valid_norms.sum().item()
                first_norm_ratio = first_norm / total_sum if total_sum > 0 else 0.0
                first_norm_ratio_list.append(first_norm_ratio)
    
            max_norm = valid_norms.max().item()
            sum_others = valid_norms.sum().item() - max_norm
            max_to_sum_ratio = max_norm / sum_others if sum_others > 0 else 0.0
            max_to_sum_ratio_list.append(max_to_sum_ratio)
    
        avg_first_is_max = sum(first_is_max_list) / len(first_is_max_list) if first_is_max_list else 0.0
        avg_first_norm_ratio = sum(first_norm_ratio_list) / len(first_norm_ratio_list) if first_norm_ratio_list else 0.0
        avg_first_norm_abs = sum(first_norm_abs_list) / len(first_norm_abs_list) if first_norm_abs_list else 0.0
        avg_max_to_sum_ratio = sum(max_to_sum_ratio_list) / len(max_to_sum_ratio_list) if max_to_sum_ratio_list else 0.0
        avg_mean_norm = sum(mean_norm_list) / len(mean_norm_list) if mean_norm_list else 0.0
    
        return {
            # старые имена (для совместимости)
            'bos_is_max_ratio': avg_first_is_max,
            'bos_norm_ratio': avg_first_norm_ratio,
            'max_norm_to_sum_others_ratio': avg_max_to_sum_ratio,
            # новые имена
            'first_norm_abs': avg_first_norm_abs,
            'mean_norm': avg_mean_norm
        }
    
    #  Регистрация хуков 
    handles = []

    # Доступ к списку слоёв: для Pythia это model.gpt_neox.layers
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        layers_container = model.gpt_neox.layers
    elif hasattr(model, 'layers'):
        layers_container = model.layers
    else:
        raise AttributeError("Не удалось найти decoder layers. Ожидается model.gpt_neox.layers или model.layers")

    print(f"Найдено слоёв: {len(layers_container)}")
    print("=" * 60)

    # Плоский тензор input_ids для фильтрации паддингов в compute_metrics
    input_ids_flat = batch['input_ids'].view(-1)

    model.eval()
    with torch.no_grad():
        for layer_idx in layers_to_plot:
            if layer_idx >= len(layers_container):
                print(f"Пропускаем слой {layer_idx} (нет в модели)")
                continue

            layer = layers_container[layer_idx]

            # Переменные для сохранения residual между хуками (замыкание)
            residual = None
            res1_state = None

            # Pre-hook на самом слое: сохраняем вход (точка 'input') 
            def create_pre_hook(l_idx, point):
                def pre_hook(module, args):
                    nonlocal residual
                    residual = args[0]  # hidden_states
                    hidden_flat = residual.view(-1, residual.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # Нормные метрики считаем на полном тензоре
                    norm_met = compute_norm_metrics(residual, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                    return args
                return pre_hook

            pre_handle = layer.register_forward_pre_hook(create_pre_hook(layer_idx, 'input'))
            handles.append(pre_handle)

            # - Hook на input_layernorm -
            def create_ln1_hook(l_idx, point):
                def hook(module, input, output):
                    # Исходные метрики
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # Нормные метрики
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook

            if hasattr(layer, 'input_layernorm'):
                ln1_handle = layer.input_layernorm.register_forward_hook(create_ln1_hook(layer_idx, 'ln1'))
                handles.append(ln1_handle)
            else:
                print(f"Слой {layer_idx} не имеет input_layernorm")

            # Hook на attention
            def create_attn_hook(l_idx, point_attn, point_res1):
                def hook(module, input, output):
                    nonlocal residual, res1_state
                    # У attention output может быть кортежем (hidden_states, ...)
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # Метрики для attention
                    hidden_flat = attn_out.view(-1, attn_out.shape[-1])
                    met_attn = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met_attn is not None:
                        metrics[(l_idx, point_attn)] = met_attn
                    # Нормные метрики для attention
                    norm_met_attn = compute_norm_metrics(attn_out, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point_attn)] = norm_met_attn

                    # Вычисляем residual 1 = residual + attn_out
                    if residual is not None:
                        res1 = residual + attn_out
                        res1_flat = res1.view(-1, res1.shape[-1])
                        met_res1 = compute_metrics(res1_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met_res1 is not None:
                            metrics[(l_idx, point_res1)] = met_res1
                        # Нормные метрики для residual 1
                        norm_met_res1 = compute_norm_metrics(res1, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point_res1)] = norm_met_res1
                        res1_state = res1
                return hook

            if hasattr(layer, 'attention'):
                attn_handle = layer.attention.register_forward_hook(
                    create_attn_hook(layer_idx, 'attn', 'res1')
                )
                handles.append(attn_handle)
            else:
                print(f"Слой {layer_idx} не имеет attention")

            # - Hook на post_attention_layernorm -
            def create_ln2_hook(l_idx, point):
                def hook(module, input, output):
                    # Исходные метрики
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # Нормные метрики
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook

            if hasattr(layer, 'post_attention_layernorm'):
                ln2_handle = layer.post_attention_layernorm.register_forward_hook(create_ln2_hook(layer_idx, 'ln2'))
                handles.append(ln2_handle)
            else:
                print(f"Слой {layer_idx} не имеет post_attention_layernorm")

            # - Hook на MLP -
            def create_mlp_hook(l_idx, point_mlp, point_res2):
                def hook(module, input, output):
                    nonlocal res1_state
                    mlp_out = output  # тензор
                    # Метрики для MLP
                    hidden_flat = mlp_out.view(-1, mlp_out.shape[-1])
                    met_mlp = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met_mlp is not None:
                        metrics[(l_idx, point_mlp)] = met_mlp
                    # Нормные метрики для MLP
                    norm_met_mlp = compute_norm_metrics(mlp_out, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point_mlp)] = norm_met_mlp

                    # Вычисляем residual 2 = res1_state + mlp_out
                    if res1_state is not None:
                        res2 = res1_state + mlp_out
                        res2_flat = res2.view(-1, res2.shape[-1])
                        met_res2 = compute_metrics(res2_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met_res2 is not None:
                            metrics[(l_idx, point_res2)] = met_res2
                        # Нормные метрики для residual 2
                        norm_met_res2 = compute_norm_metrics(res2, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point_res2)] = norm_met_res2
                return hook

            if hasattr(layer, 'mlp'):
                mlp_handle = layer.mlp.register_forward_hook(
                    create_mlp_hook(layer_idx, 'mlp', 'res2')
                )
                handles.append(mlp_handle)
            else:
                print(f"Слой {layer_idx} не имеет mlp")

        #  Хуки на финальные слои 
        # Final LayerNorm: обычно model.gpt_neox.final_layer_norm
        if len(layers_to_plot)>model.config.num_hidden_layers-1:
            final_norm = None
            if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
                final_norm = model.gpt_neox.final_layer_norm
            elif hasattr(model, 'final_layer_norm'):
                final_norm = model.final_layer_norm
            
            if final_norm is not None:
                def create_final_norm_hook(l_idx, point):
                    def hook(module, input, output):
                        # Исходные метрики
                        hidden_flat = output.view(-1, output.shape[-1])
                        met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met is not None:
                            metrics[(l_idx, point)] = met
                        # Нормные метрики
                        norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point)] = norm_met
                    return hook
                final_layer_idx = len(layers_container)  # следующий индекс за последним слоем
                norm_handle = final_norm.register_forward_hook(create_final_norm_hook(final_layer_idx, 'final_norm'))
                handles.append(norm_handle)
                print("Хук на final_layer_norm зарегистрирован.")
            else:
                print("Предупреждение: не найден final_layer_norm")
    
            # LM Head: обычно model.lm_head
            lm_head = None
            if hasattr(model, 'lm_head'):
                lm_head = model.lm_head
            elif hasattr(model, 'embed_out'):  # иногда называется embed_out
                lm_head = model.embed_out
    
            if lm_head is not None:
                def create_lm_head_hook(l_idx, point):
                    def hook(module, input, output):
                        logits = output[0] if isinstance(output, tuple) else output
                        # Исходные метрики
                        hidden_flat = logits.view(-1, logits.shape[-1])
                        met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met is not None:
                            metrics[(l_idx, point)] = met
                        # Нормные метрики (логиты имеют ту же форму)
                        norm_met = compute_norm_metrics(logits, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point)] = norm_met
                    return hook
                lm_head_idx = len(layers_container) + 1  # следующий после final_norm
                lm_head_handle = lm_head.register_forward_hook(create_lm_head_hook(lm_head_idx, 'lm_head'))
                handles.append(lm_head_handle)
                print("Хук на lm_head зарегистрирован.")
            else:
                print("Предупреждение: не найден lm_head")

        #  Один forward pass 
        print("Запуск forward pass для сбора промежуточных состояний...")
        _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=False  # используем только хуки
        )
        print("Forward pass завершён.")

    #  Удаление хуков 
    for h in handles:
        h.remove()
    print(f"Удалено {len(handles)} хуков.")

    #  Проверка наличия метрик 
    if not metrics:
        print("Не удалось собрать метрики. Проверьте модель и входные данные.")
        return {}

    #  Подготовка данных для графиков (общая для всех метрик) 
    # Сортируем ключи: сначала обычные слои (0..N-1) в порядке возрастания,
    # для каждого слоя точки в порядке POINTS, затем финальные точки.
    layer_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] < len(layers_container)))
    final_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] >= len(layers_container)))

    all_points = []
    for layer in layer_numbers:
        for point in POINTS:
            # Добавляем input только для нулевого слоя
            if point == 'input' and layer != 0:
                continue
            key = (layer, point)
            if key in metrics:
                all_points.append(key)

    final_desired_order = ['final_norm', 'lm_head']
    for point_name in final_desired_order:
        for layer in final_numbers:
            key = (layer, point_name)
            if key in metrics:
                all_points.append(key)
                break

    
    # Подписи для оси X

    x_labels_all = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            if point == 'res2':
                x_labels_all.append(f"{layer}")
            else:
                x_labels_all.append("")
        else:
            if point == 'final_norm':
                x_labels_all.append("")
            elif point == 'lm_head':
                x_labels_all.append("")
            else:
                x_labels_all.append(f"Final {point}")
                        
    x_labels = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            x_labels.append(f"{layer} {POINT_ABBR.get(point, point)}")
        else:
            if point == 'final_norm':
                x_labels.append("Final LayerNorm")
            elif point == 'lm_head':
                x_labels.append("LM Head")
            else:
                x_labels.append(f"Final {point}")
    x_pos = np.arange(len(all_points))

    dashed_indices = []
    for i, (layer, point) in enumerate(all_points):
        if (layer == 0 and point == 'input') or point == 'res2' or point == 'lm_head':
            dashed_indices.append(i)

    #  Построение исходных графиков (косинус, анизотропия и т.д.) 
    # (код такой же, как в оригинале, оставляем без изменений)
    cosine_means = [metrics[k]['cosine_mean'] for k in all_points]
    cosine_means_centered = [metrics[k]['cosine_mean_centered'] for k in all_points]
    anisotropies = [metrics[k]['anisotropy'] for k in all_points]
    participation_ratios = [metrics[k]['participation_ratio'] for k in all_points]
    ranks = [metrics[k]['rank'] for k in all_points]
    isoscores = [metrics[k]['isoscore'] for k in all_points]
    max_sings = [metrics[k]['max_sing'] for k in all_points]


    #  Вычисление корреляций Пирсона 
    from scipy.stats import pearsonr

    # Находим общие точки, для которых есть оба типа метрик
    common_keys = sorted(set(metrics.keys()) & set(norm_metrics.keys()))
    print(f"Найдено общих точек для корреляций: {len(common_keys)}")

    if len(common_keys) > 1:
        # Формируем списки значений в одном порядке
        anisotropy_vals = []
        cosine_vals = []
        ratio_vals = []
        abs_vals = []
        for key in common_keys:
            anisotropy_vals.append(metrics[key]['anisotropy'])
            cosine_vals.append(metrics[key]['cosine_mean'])
            ratio_vals.append(norm_metrics[key]['bos_norm_ratio'])
            abs_vals.append(norm_metrics[key]['first_norm_abs'])

        # 1. Анизотропия vs относительная норма первого токена
        corr_ratio_aniso, p_ratio_aniso = pearsonr(anisotropy_vals, ratio_vals)
        print(f"Корреляция анизотропии и относительной нормы первого токена: "
              f"r = {corr_ratio_aniso:.4f}, p-value = {p_ratio_aniso:.4e}")

        # 2. Анизотропия vs абсолютная норма первого токена
        corr_abs_aniso, p_abs_aniso = pearsonr(anisotropy_vals, abs_vals)
        print(f"Корреляция анизотропии и абсолютной нормы первого токена: "
              f"r = {corr_abs_aniso:.4f}, p-value = {p_abs_aniso:.4e}")

        # 3. Среднее косинусное сходство vs относительная норма первого токена
        corr_ratio_cos, p_ratio_cos = pearsonr(cosine_vals, ratio_vals)
        print(f"Корреляция среднего косинусного сходства и относительной нормы первого токена: "
              f"r = {corr_ratio_cos:.4f}, p-value = {p_ratio_cos:.4e}")

        # 4. Среднее косинусное сходство vs абсолютная норма первого токена
        corr_abs_cos, p_abs_cos = pearsonr(cosine_vals, abs_vals)
        print(f"Корреляция среднего косинусного сходства и абсолютной нормы первого токена: "
              f"r = {corr_abs_cos:.4f}, p-value = {p_abs_cos:.4e}")
    else:
        print("Недостаточно общих точек для вычисления корреляций.")
        #  Вычисление корреляции Пирсона 
    from scipy.stats import pearsonr  # можно использовать numpy, но scipy даёт p-value

    # Находим общие точки, для которых есть оба типа метрик
    common_keys = set(metrics.keys()) & set(norm_metrics.keys())
    print(f"Найдено общих точек для корреляции: {len(common_keys)}")

    if len(common_keys) > 1:
        anisotropy_vals = []
        first_norm_ratio_vals = []
        first_norm_abs_vals = []

        for key in sorted(common_keys):  # сортировка для порядка
            anisotropy_vals.append(metrics[key]['anisotropy'])
            first_norm_ratio_vals.append(norm_metrics[key]['bos_norm_ratio'])
            first_norm_abs_vals.append(norm_metrics[key]['first_norm_abs'])

        # Корреляция между анизотропией и относительной нормой первого токена
        corr_ratio, p_ratio = pearsonr(anisotropy_vals, first_norm_ratio_vals)
        print(f"Корреляция Пирсона между анизотропией и относительной нормой первого токена: "
              f"r = {corr_ratio:.4f}, p-value = {p_ratio:.4e}")

        # Корреляция между анизотропией и абсолютной нормой первого токена
        corr_abs, p_abs = pearsonr(anisotropy_vals, first_norm_abs_vals)
        print(f"Корреляция Пирсона между анизотропией и абсолютной нормой первого токена: "
              f"r = {corr_abs:.4f}, p-value = {p_abs:.4e}")
    else:
        print("Недостаточно общих точек для вычисления корреляции.")

    # График 1.1: среднее косинусное сходство
    print("cosine component-wise", cosine_means)
    
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, cosine_means, marker='o', linestyle='-', color='b')
    dashed_vals = [cosine_means[i] for i in dashed_indices]
    plt.plot(dashed_indices, dashed_vals, 'r--', marker='o', markersize=6, color="blue", linewidth=2, label='layer-wise cosine anisotropy')
    plt.xticks(x_pos, x_labels_all, fontsize=20)#, rotation=90
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity", fontsize=20)
    plt.xlabel("Layer", fontsize=20)
    plt.title(f"Cosine similarity of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
    plt.show()


    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:55], cosine_means[:55], marker='o', linestyle='-', color='b')
    plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    dashed_indices_55 = [i for i in dashed_indices if i < 55]
    dashed_vals_55 = [cosine_means[i] for i in dashed_indices_55]
    plt.plot(dashed_indices_55, dashed_vals_55, 'b--', marker='s', markersize=6, linewidth=2,label='layer-wise singular anisotropy')
    plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity", fontsize=20)
    plt.title(f"Cosine similarity of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
    plt.show()
        

     # График 1.2: среднее косинусное сходство centered
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, cosine_means_centered, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity centered", fontsize=20)
    plt.title(f"Cosine similarity of hidden states centered – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage_centered.png", dpi=150)
    plt.show()

    # IsoScore
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, isoscores, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("IsoScore", fontsize=20)
    plt.title(f"IsoScore of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "isoscore.png", dpi=150)
    plt.show()
    
    # Plot 2: anisotropy
    print("component-wise singular", anisotropies)

    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, anisotropies, marker='s', linestyle='-', color='r')
    dashed_vals = [anisotropies[i] for i in dashed_indices]
    plt.plot(dashed_indices, dashed_vals, 'r--', marker='s', markersize=6, linewidth=2, label='layer-wise singular anisotropy')
    plt.xticks(x_pos, x_labels_all, fontsize=20)#, rotation=90
    plt.tick_params(axis='y', labelsize=20)
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.ylabel("Anisotropy (σ₁² / Σ σᵢ²)", fontsize=20)
    plt.xlabel("Layer", fontsize=20)
    plt.title(f"Anisotropy of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
    ax1 = plt.gca()
    pos1 = ax1.get_position()
    height1 = pos1.height   
    plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:55], anisotropies[:55], marker='s', linestyle='-', color='r')
    dashed_indices_55 = [i for i in dashed_indices if i < 55]
    dashed_vals_55 = [anisotropies[i] for i in dashed_indices_55]
    plt.plot(dashed_indices_55, dashed_vals_55, 'r--', marker='s', markersize=6, linewidth=2,label='layer-wise singular anisotropy')
    plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    # dashed_vals = [anisotropies[i] for i in dashed_indices[:55]]
    # plt.plot(dashed_indices, dashed_vals, 'r--', marker='s', markersize=6, linewidth=2, label='layer-wise singular anisotropy')
    # plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Anisotropy (σ₁² / Σ σᵢ²)", fontsize=20)
    plt.title(f"Anisotropy of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.tight_layout()
    ax2 = plt.gca()
    pos2 = ax2.get_position()
    # Принудительно задаём ту же высоту, что у первого графика
    ax2.set_position([pos2.x0, pos2.y0, pos2.width, height1])
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
    plt.show()
        
    # Plot 3: effective dimensionality
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, participation_ratios, marker='^', linestyle='-', color='g')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.ylabel("Effective dimensionality", fontsize=20)
    plt.title(f"Effective dimensionality – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "participation_ratio_per_stage.png", dpi=150)
    plt.show()
    
    # Plot 4: rank
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, ranks, marker='d', linestyle='-', color='purple')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.ylabel("Rank (number of singular values > 1e-7)")
    plt.title(f"Rank of hidden states – {model.config._name_or_path}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "rank_per_stage.png", dpi=150)
    plt.show()
    
    # Plot 5: maximum singular value
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, max_sings, marker='d', linestyle='-', color='orange')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.ylabel("Maximum singular value", fontsize=20)
    plt.title(f"Maximum singular value – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "max_singular_per_stage.png", dpi=150)
    plt.show()
    
    #  plots for norm-based metrics 
    # Extract values for the same points as in all_points
    bos_is_max_values = [norm_metrics[k]['bos_is_max_ratio'] for k in all_points if k in norm_metrics]
    bos_norm_ratio_values = [norm_metrics[k]['bos_norm_ratio'] for k in all_points if k in norm_metrics]
    max_to_sum_values = [norm_metrics[k]['max_norm_to_sum_others_ratio'] for k in all_points if k in norm_metrics]
    
    # If some points are missing from norm_metrics, issue a warning but plot the available ones
    if len(bos_is_max_values) != len(all_points):
        print("Warning: norm metrics not collected for all points.")
    
    # Plot 6: fraction of examples where the BOS norm is maximum
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:len(bos_is_max_values)], bos_is_max_values, marker='o', linestyle='-', color='blue')
    plt.xticks(x_pos[:len(bos_is_max_values)], x_labels[:len(bos_is_max_values)], rotation=90)
    plt.ylabel("Fraction of examples where the first token norm is maximum")
    plt.title("Fraction of examples with maximum norm at BOS")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "bos_is_max_ratio.png", dpi=150)
    plt.show()
    
    # Plot 7: ratio of BOS norm to sum of all norms
    first_norm_abs_values = [norm_metrics[k]['first_norm_abs'] for k in all_points if k in norm_metrics]
    mean_norm_values = [norm_metrics[k]['mean_norm'] for k in all_points if k in norm_metrics]
    
    if len(first_norm_abs_values) > 0 and len(mean_norm_values) > 0:
        plt.figure(figsize=(20, 8))
        n = len(first_norm_abs_values)
        x_vals = x_pos[:n]
        x_labels_trim = x_labels[:n]
        
        plt.plot(x_vals, first_norm_abs_values, 
                 marker='s', linestyle='-', color='green', label='First token norm')
        plt.plot(x_vals, mean_norm_values, 
                 marker='.', linestyle='--', color='blue', alpha=0.7, label='Mean token norm')
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel("Token norm", fontsize=20)
        plt.xticks(x_vals, x_labels_trim, rotation=90, fontsize=20)
        plt.legend(fontsize=20)  
        plt.title("Absolute norm of the first token and mean token norm except first token", fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(plot_dir) / "first_norm_abs_with_mean.png", dpi=150)
        plt.show()
    else:
        print("Warning: insufficient data to plot first token norm and mean norm.")

    #  Plot: отношение нормы первого токена к сумме всех норм 
    if len(bos_norm_ratio_values) > 0:
        print("relative norm of FT", bos_norm_ratio_values)
        n = len(bos_norm_ratio_values)
        plt.figure(figsize=(20, 8))
        plt.plot(x_pos[:n], bos_norm_ratio_values, marker='o', linestyle='-', color='purple', linewidth=2)
        plt.xticks(x_pos[:n], x_labels[:n], rotation=90, fontsize=20)
        plt.ylabel("First token norm / sum of all norms", fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.title("Relative norm of the first token", fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(plot_dir) / "first_norm_to_sum_ratio.png", dpi=150)
        plt.show()
    
    # Plot 8: ratio of maximum norm to the sum of the others
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:len(max_to_sum_values)], max_to_sum_values, marker='^', linestyle='-', color='red')
    plt.xticks(x_pos[:len(max_to_sum_values)], x_labels[:len(max_to_sum_values)], rotation=90)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("max(norm) / sum(other norms)")
    plt.title("Maximum norm dominance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "max_norm_to_sum_others.png", dpi=150)
    plt.show()

    #  Вывод сводки в консоль 
    print("\n" + "=" * 60)
    print("Сводка метрик по этапам:")
    for layer, point in all_points:
        m = metrics[(layer, point)]
        layer_label = f"{layer}" if layer < len(layers_container) else "Final"
        print(f"{layer_label:<4} {point:10s}: cosine={m['cosine_mean']:.4f}±{m['cosine_std']:.4f}, "
              f"anisotropy={m['anisotropy']:.4f}")

    # В конце функции, перед return:
    generate_latex_table(metrics, len(layers_container), model, save_plots=save_plots, plot_dir=plot_dir if save_plots else None)
    # Возвращаем собранные метрики для возможного дальнейшего использования
    return {'metrics': metrics, 'norm_metrics': norm_metrics}

    # 10. Сохранение таблиц с результатами 
    try:
        import pandas as pd
        from pathlib import Path

        # Получаем общее число параметров модели
        total_params = sum(p.numel() for p in model.parameters())
        # Формируем читаемое представление 
        if total_params >= 1e9:
            params_str = f"{total_params/1e9:.1f}B"
        elif total_params >= 1e6:
            params_str = f"{total_params/1e6:.1f}M"
        else:
            params_str = str(total_params)

        base_filename = f"{config.model_name}_{params_str}_cosine_analysis"

        # Подробная таблица по точкам 
        detailed_data = []
        for key in all_points:
            layer, point = key
            if layer < len(layers_container):
                layer_label = str(layer)
            else:
                if point == 'final_norm':
                    layer_label = 'final_norm'
                elif point == 'lm_head':
                    layer_label = 'lm_head'
                else:
                    layer_label = 'final'
            m = metrics[key]
            detailed_data.append({
                'Stage': x_labels[len(detailed_data)],  # используем уже готовую подпись
                'Layer': layer_label,
                'Point': point,
                'Cosine_mean': m['cosine_mean'],
                'Cosine_std': m['cosine_std'],
                'Anisotropy': m['anisotropy'],
                'Participation_ratio': m['participation_ratio'],
                'Rank': m['rank'],
                'Max_singular': m['max_sing']
            })

        df_detailed = pd.DataFrame(detailed_data)

        # Сохраняем в Excel и CSV
        excel_path = Path(plot_dir) / f"{base_filename}.xlsx"
        csv_path = Path(plot_dir) / f"{base_filename}.csv"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_detailed.to_excel(writer, sheet_name='Detailed', index=False)

            #  Сводная таблица по типам операций 
            # Определяем категорию для каждой точки
            def get_category(point):
                if point == 'attn':
                    return 'attention'
                elif point == 'mlp':
                    return 'fnn'
                elif point in ['ln1', 'ln2', 'final_norm']:
                    return 'normalization'
                elif point in ['input', 'res1', 'res2']:
                    return 'residuals'
                else:
                    return 'other'  

            # Добавим колонку категории в детальный DataFrame
            df_detailed['Category'] = df_detailed['Point'].apply(get_category)

            # Для сводной таблицы возьмём только основные категории (исключим 'other')
            categories_of_interest = ['attention', 'fnn', 'normalization', 'residuals']
            df_cat = df_detailed[df_detailed['Category'].isin(categories_of_interest)]

            # Метрики, для которых считаем статистики
            metrics_for_summary = ['Cosine_mean', 'Anisotropy', 'Participation_ratio', 'Rank', 'Max_singular']

            # Собираем строки сводной таблицы
            summary_rows = []
            for cat in categories_of_interest:
                cat_data = df_cat[df_cat['Category'] == cat]
                row = {'Category': cat}
                for met in metrics_for_summary:
                    values = cat_data[met].dropna()
                    if len(values) > 0:
                        row[f'{met}_max'] = values.max()
                        row[f'{met}_min'] = values.min()
                        row[f'{met}_mean'] = values.mean()
                    else:
                        row[f'{met}_max'] = row[f'{met}_min'] = row[f'{met}_mean'] = None
                summary_rows.append(row)

            df_summary = pd.DataFrame(summary_rows)

            # Сохраняем сводную таблицу на отдельном листе
            df_summary.to_excel(writer, sheet_name='Summary_by_type', index=False)

        # Также сохраняем CSV с подробной таблицей
        df_detailed.to_csv(csv_path, index=False)

        print(f"\nТаблицы сохранены:")
        print(f"  - Excel: {excel_path}")
        print(f"  - CSV: {csv_path}")

    except ImportError:
        print("\nБиблиотека pandas не установлена. Таблицы не сохранены.")
    except Exception as e:
        print(f"\nОшибка при сохранении таблиц: {e}")

    return metrics


def analyze_componentwise_anisotropy_gpt2(model, batch, config, compute_isoscore=False, layers_to_plot=None, save_plots=False, ALL=False):
    """
    Расширенный анализ косинусной близости и анизотропии для моделей GPT-2.
    Собирает скрытые состояния на 7 этапах внутри каждого decoder layer (блока Transformer):
        - вход в слой (input)
        - после ln_1 (первый layer norm)
        - после attention (attn)
        - после первого residual (res1 = input + attn_out)
        - после ln_2 (второй layer norm)
        - после MLP (mlp)
        - после второго residual (res2 = res1 + mlp_out)
    Для каждого этапа вычисляет:
        - среднее попарное косинусное сходство (и std)
        - анизотропию = (σ₁²) / (Σ σᵢ²), где σᵢ – сингулярные числа
    Строит два графика: среднее косинусное сходство и анизотропия по всем этапам.
    Дополнительно добавляет два финальных этапа:
        - после final_layer_norm (model.transformer.ln_f, перед lm_head)
        - после lm_head (логиты)
    
    Также вычисляются метрики на основе норм токенов:
        - Доля примеров, в которых норма BOS максимальна среди всех не‑паддинг токенов.
        - Отношение нормы BOS к сумме норм всех токенов (усреднённое по примерам).
        - Отношение максимальной нормы к сумме остальных норм (усреднённое по примерам).
    Для корректности проверяется, что первый токен действительно является BOS (используется bos_token_id из config).
    """

    #  Устройство и данные 
    model_device = next(model.parameters()).device
    print("model_device", model_device)
    if batch['input_ids'].device != model_device:
        print(f"Перемещаем данные с {batch['input_ids'].device} на {model_device}")
        batch = {k: v.to(model_device) for k, v in batch.items()}

    #  Pad token id для GPT-2 (обычно eos_token_id = 50256) 
    pad_token_id = getattr(config, 'eos_token_id', 50256)
    if pad_token_id is None:
        pad_token_id = 50256
    print(f"Используется pad_token_id = {pad_token_id}")

    #  BOS token id для проверки первого токена 
    bos_token_id = getattr(config, 'bos_token_id', getattr(config, 'eos_token_id', 50256))
    if bos_token_id is None:
        bos_token_id = 50256
    print(f"Используется bos_token_id = {bos_token_id}")

    #  Определяем список слоёв для анализа 
    if layers_to_plot is None:
        layers_to_plot = list(range(0, model.config.num_hidden_layers + 1, 1))  # все слои + финальные

    # Директория для сохранения графиков
    if save_plots:
        plot_dir = setup_plot_saving(config, suffix="cosine_analysis_gpt2")
        print(f"Графики будут сохранены в: {plot_dir}")

    #  Точки внутри слоя и их подписи 
    POINTS = ['input', 'ln1', 'attn', 'res1', 'ln2', 'mlp', 'res2']
    POINT_ABBR = {
        'input': 'input',
        'ln1': 'LayerNorm 1',
        'attn': 'attention',
        'res1': 'residual 1',
        'ln2': 'LayerNorm 2',
        'mlp': 'MLP',
        'res2': 'residual 2'
    }

    #  Контейнеры для результатов 
    metrics = {}          # для косинусных и SVD метрик
    norm_metrics = {}     # для метрик на основе норм токенов

    #  Вспомогательная функция вычисления исходных метрик (косинус, анизотропия) 
    # (оставляем без изменений, как в оригинале)
    def compute_metrics(tensor_hidden, input_ids_flat, pad_token_id, num_samples):
        """
        tensor_hidden: (batch*seq_len, hidden_dim) на GPU
        input_ids_flat: (batch*seq_len,) на GPU
        """
        valid_mask = (input_ids_flat != pad_token_id)
        embeddings = tensor_hidden[valid_mask]

        if embeddings.shape[0] == 0:
            return None

        # Подвыборка, если слишком много токенов
        if embeddings.shape[0] > num_samples:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:num_samples]
            embeddings = embeddings[idx]

        #  Косинусное сходство 
        mean = embeddings.mean(dim=0, keepdim=True)
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / torch.clamp(norms, min=1e-8)
        cos_matrix = normalized @ normalized.T
        n = cos_matrix.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1, device=cos_matrix.device)
        cos_values = cos_matrix[triu_indices[0], triu_indices[1]]
        cosine_mean = cos_values.mean().item()
        cosine_std = cos_values.std().item()

        if compute_isoscore:
            isoscore_val = compute_isoscore(embeddings)  # предполагается, что функция определена вне
        else:
            isoscore_val = 0

        #  Косинусное сходство centered 
        embeddings_centered = embeddings - mean
        norms_centered = torch.norm(embeddings_centered, dim=1, keepdim=True)
        normalized_centered = embeddings_centered / torch.clamp(norms_centered, min=1e-8)
        cos_matrix_centered = normalized_centered @ normalized_centered.T
        cos_values_centered = cos_matrix_centered[triu_indices[0], triu_indices[1]]
        cosine_mean_centered = cos_values_centered.mean().item()
        cosine_std_centered = cos_values_centered.std().item()

        #  Анизотропия и участие (participation ratio) через SVD 
        embeddings_svd = embeddings_centered - embeddings_centered.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(embeddings_svd.float())
        s_sq = s ** 2
        sum_sq = s_sq.sum()
        sum_sq_sq = (s_sq ** 2).sum()
        anisotropy = (s_sq[0] / sum_sq).item() if sum_sq > 0 else 0.0
        participation_ratio = (sum_sq ** 2 / sum_sq_sq).item() if sum_sq_sq > 0 else 0.0
        rank = (s > 1e-7).sum().item()

        return {
            'cosine_mean': cosine_mean,
            'cosine_std': cosine_std,
            'cosine_mean_centered': cosine_mean_centered,
            'cosine_std_centered': cosine_std_centered,
            'isoscore': isoscore_val,
            'anisotropy': anisotropy,
            'participation_ratio': participation_ratio,
            'rank': rank,
            'max_sing': s[0].cpu().item() if s.numel() > 0 else 0.0
        }

    #  Функция для вычисления метрик на основе норм токенов (без изменений) 
    def compute_norm_metrics(hidden_states, attention_mask, input_ids, bos_token_id):
        norms = torch.norm(hidden_states, dim=-1)
        mask = attention_mask.bool()
        batch_size = hidden_states.size(0)

        first_is_max_list = []
        first_norm_ratio_list = []
        first_norm_abs_list = []
        max_to_sum_ratio_list = []
        mean_norm_list = []

        for i in range(batch_size):
            sample_norms = norms[i]
            sample_mask = mask[i]
            valid_norms = sample_norms[sample_mask]
            if valid_norms.numel() == 0:
                continue

            mean_norm_list.append(valid_norms.mean().item())

            if sample_mask[0].item():
                first_norm = sample_norms[0].item()
                first_norm_abs_list.append(first_norm)
                first_is_max = (first_norm == valid_norms.max().item())
                first_is_max_list.append(float(first_is_max))
                total_sum = valid_norms.sum().item()
                first_norm_ratio = first_norm / total_sum if total_sum > 0 else 0.0
                first_norm_ratio_list.append(first_norm_ratio)

            max_norm = valid_norms.max().item()
            sum_others = valid_norms.sum().item() - max_norm
            max_to_sum_ratio = max_norm / sum_others if sum_others > 0 else 0.0
            max_to_sum_ratio_list.append(max_to_sum_ratio)

        avg_first_is_max = sum(first_is_max_list) / len(first_is_max_list) if first_is_max_list else 0.0
        avg_first_norm_ratio = sum(first_norm_ratio_list) / len(first_norm_ratio_list) if first_norm_ratio_list else 0.0
        avg_first_norm_abs = sum(first_norm_abs_list) / len(first_norm_abs_list) if first_norm_abs_list else 0.0
        avg_max_to_sum_ratio = sum(max_to_sum_ratio_list) / len(max_to_sum_ratio_list) if max_to_sum_ratio_list else 0.0
        avg_mean_norm = sum(mean_norm_list) / len(mean_norm_list) if mean_norm_list else 0.0

        return {
            'bos_is_max_ratio': avg_first_is_max,
            'bos_norm_ratio': avg_first_norm_ratio,
            'max_norm_to_sum_others_ratio': avg_max_to_sum_ratio,
            'first_norm_abs': avg_first_norm_abs,
            'mean_norm': avg_mean_norm
        }

    #  Регистрация хуков для GPT-2 
    handles = []

    # Доступ к списку слоёв GPT-2: model.transformer.h
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers_container = model.transformer.h
    else:
        raise AttributeError("Не удалось найти decoder layers. Ожидается model.transformer.h")

    print(f"Найдено слоёв: {len(layers_container)}")
    print("=" * 60)

    # Плоский тензор input_ids для фильтрации паддингов в compute_metrics
    input_ids_flat = batch['input_ids'].view(-1)

    model.eval()
    with torch.no_grad():
        for layer_idx in layers_to_plot:
            if layer_idx >= len(layers_container):
                print(f"Пропускаем слой {layer_idx} (нет в модели)")
                continue

            layer = layers_container[layer_idx]

            # Переменные для сохранения residual между хуками
            residual = None          # вход в блок (до ln1)
            res1_state = None        # результат первого residual (input + attn)

            # - Pre-hook на самом блоке: сохраняем вход (точка 'input') -
            def create_pre_hook(l_idx, point):
                def pre_hook(module, args):
                    nonlocal residual
                    residual = args[0]  # hidden_states
                    hidden_flat = residual.view(-1, residual.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    # Нормные метрики
                    norm_met = compute_norm_metrics(residual, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                    return args
                return pre_hook

            pre_handle = layer.register_forward_pre_hook(create_pre_hook(layer_idx, 'input'))
            handles.append(pre_handle)

            # - Hook на ln_1 (первый layer norm) -
            def create_ln1_hook(l_idx, point):
                def hook(module, input, output):
                    # output = после ln_1
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook

            if hasattr(layer, 'ln_1'):
                ln1_handle = layer.ln_1.register_forward_hook(create_ln1_hook(layer_idx, 'ln1'))
                handles.append(ln1_handle)
            else:
                print(f"Слой {layer_idx} не имеет ln_1")

            # - Hook на attention (attn) -
            def create_attn_hook(l_idx, point_attn, point_res1):
                def hook(module, input, output):
                    nonlocal residual, res1_state
                    # У attention output обычно тензор (hidden_states)
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # Метрики для attention
                    hidden_flat = attn_out.view(-1, attn_out.shape[-1])
                    met_attn = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met_attn is not None:
                        metrics[(l_idx, point_attn)] = met_attn
                    norm_met_attn = compute_norm_metrics(attn_out, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point_attn)] = norm_met_attn

                    # Вычисляем residual 1 = residual + attn_out
                    if residual is not None:
                        res1 = residual + attn_out
                        res1_flat = res1.view(-1, res1.shape[-1])
                        met_res1 = compute_metrics(res1_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met_res1 is not None:
                            metrics[(l_idx, point_res1)] = met_res1
                        norm_met_res1 = compute_norm_metrics(res1, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point_res1)] = norm_met_res1
                        res1_state = res1
                return hook

            if hasattr(layer, 'attn'):
                attn_handle = layer.attn.register_forward_hook(create_attn_hook(layer_idx, 'attn', 'res1'))
                handles.append(attn_handle)
            else:
                print(f"Слой {layer_idx} не имеет attn")

            # - Hook на ln_2 (второй layer norm) -
            def create_ln2_hook(l_idx, point):
                def hook(module, input, output):
                    # output = после ln_2
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point)] = norm_met
                return hook

            if hasattr(layer, 'ln_2'):
                ln2_handle = layer.ln_2.register_forward_hook(create_ln2_hook(layer_idx, 'ln2'))
                handles.append(ln2_handle)
            else:
                print(f"Слой {layer_idx} не имеет ln_2")

            # - Hook на MLP -
            def create_mlp_hook(l_idx, point_mlp, point_res2):
                def hook(module, input, output):
                    nonlocal res1_state
                    mlp_out = output  # тензор
                    # Метрики для MLP
                    hidden_flat = mlp_out.view(-1, mlp_out.shape[-1])
                    met_mlp = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met_mlp is not None:
                        metrics[(l_idx, point_mlp)] = met_mlp
                    norm_met_mlp = compute_norm_metrics(mlp_out, batch['attention_mask'], batch['input_ids'], bos_token_id)
                    norm_metrics[(l_idx, point_mlp)] = norm_met_mlp

                    # Вычисляем residual 2 = res1_state + mlp_out
                    if res1_state is not None:
                        res2 = res1_state + mlp_out
                        res2_flat = res2.view(-1, res2.shape[-1])
                        met_res2 = compute_metrics(res2_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met_res2 is not None:
                            metrics[(l_idx, point_res2)] = met_res2
                        norm_met_res2 = compute_norm_metrics(res2, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point_res2)] = norm_met_res2
                return hook

            if hasattr(layer, 'mlp'):
                mlp_handle = layer.mlp.register_forward_hook(create_mlp_hook(layer_idx, 'mlp', 'res2'))
                handles.append(mlp_handle)
            else:
                print(f"Слой {layer_idx} не имеет mlp")

        #  Хуки на финальные слои (после всех блоков) 
        # Final LayerNorm: model.transformer.ln_f
        if len(layers_to_plot) > model.config.num_hidden_layers - 1:
            final_norm = None
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
                final_norm = model.transformer.ln_f

            if final_norm is not None:
                def create_final_norm_hook(l_idx, point):
                    def hook(module, input, output):
                        hidden_flat = output.view(-1, output.shape[-1])
                        met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met is not None:
                            metrics[(l_idx, point)] = met
                        norm_met = compute_norm_metrics(output, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point)] = norm_met
                    return hook
                final_layer_idx = len(layers_container)  # следующий индекс
                norm_handle = final_norm.register_forward_hook(create_final_norm_hook(final_layer_idx, 'final_norm'))
                handles.append(norm_handle)
                print("Хук на final_layer_norm зарегистрирован.")
            else:
                print("Предупреждение: не найден final_layer_norm (model.transformer.ln_f)")

            # LM Head: model.lm_head
            lm_head = None
            if hasattr(model, 'lm_head'):
                lm_head = model.lm_head

            if lm_head is not None:
                def create_lm_head_hook(l_idx, point):
                    def hook(module, input, output):
                        logits = output[0] if isinstance(output, tuple) else output
                        hidden_flat = logits.view(-1, logits.shape[-1])
                        met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                        if met is not None:
                            metrics[(l_idx, point)] = met
                        norm_met = compute_norm_metrics(logits, batch['attention_mask'], batch['input_ids'], bos_token_id)
                        norm_metrics[(l_idx, point)] = norm_met
                    return hook
                lm_head_idx = len(layers_container) + 1  # следующий после final_norm
                lm_head_handle = lm_head.register_forward_hook(create_lm_head_hook(lm_head_idx, 'lm_head'))
                handles.append(lm_head_handle)
                print("Хук на lm_head зарегистрирован.")
            else:
                print("Предупреждение: не найден lm_head")

        #  Один forward pass 
        print("Запуск forward pass для сбора промежуточных состояний...")
        _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=False
        )
        print("Forward pass завершён.")

    #  Удаление хуков 
    for h in handles:
        h.remove()
    print(f"Удалено {len(handles)} хуков.")

    #  Проверка наличия метрик 
    if not metrics:
        print("Не удалось собрать метрики. Проверьте модель и входные данные.")
        return {}

    #  Подготовка данных для графиков (аналогично оригиналу) 
    layer_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] < len(layers_container)))
    final_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] >= len(layers_container)))

    all_points = []
    for layer in layer_numbers:
        for point in POINTS:
            if point == 'input' and layer != 0:
                continue
            key = (layer, point)
            if key in metrics:
                all_points.append(key)

    final_desired_order = ['final_norm', 'lm_head']
    for point_name in final_desired_order:
        for layer in final_numbers:
            key = (layer, point_name)
            if key in metrics:
                all_points.append(key)
                break

    # Подписи для оси X (упрощённые)
    x_labels_all = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            if point == 'res2':
                x_labels_all.append(f"{layer}")
            else:
                x_labels_all.append("")
        else:
            if point == 'final_norm':
                x_labels_all.append("")
            elif point == 'lm_head':
                x_labels_all.append("")
            else:
                x_labels_all.append(f"Final {point}")

    x_labels = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            x_labels.append(f"{layer} {POINT_ABBR.get(point, point)}")
        else:
            if point == 'final_norm':
                x_labels.append("Final LayerNorm")
            elif point == 'lm_head':
                x_labels.append("LM Head")
            else:
                x_labels.append(f"Final {point}")

    x_pos = np.arange(len(all_points))

    dashed_indices = []
    for i, (layer, point) in enumerate(all_points):
        if (layer == 0 and point == 'input') or point == 'res2' or point == 'lm_head':
            dashed_indices.append(i)

    #  Извлечение значений для графиков 
    cosine_means = [metrics[k]['cosine_mean'] for k in all_points]
    cosine_means_centered = [metrics[k]['cosine_mean_centered'] for k in all_points]
    anisotropies = [metrics[k]['anisotropy'] for k in all_points]
    participation_ratios = [metrics[k]['participation_ratio'] for k in all_points]
    ranks = [metrics[k]['rank'] for k in all_points]
    isoscores = [metrics[k]['isoscore'] for k in all_points]
    max_sings = [metrics[k]['max_sing'] for k in all_points]

    #  Вычисление корреляций Пирсона (как в оригинале) 
    from scipy.stats import pearsonr
    common_keys = sorted(set(metrics.keys()) & set(norm_metrics.keys()))
    print(f"Найдено общих точек для корреляций: {len(common_keys)}")
    if len(common_keys) > 1:
        anisotropy_vals = []
        cosine_vals = []
        ratio_vals = []
        abs_vals = []
        for key in common_keys:
            anisotropy_vals.append(metrics[key]['anisotropy'])
            cosine_vals.append(metrics[key]['cosine_mean'])
            ratio_vals.append(norm_metrics[key]['bos_norm_ratio'])
            abs_vals.append(norm_metrics[key]['first_norm_abs'])

        corr_ratio_aniso, p_ratio_aniso = pearsonr(anisotropy_vals, ratio_vals)
        print(f"Корреляция анизотропии и относительной нормы первого токена: r = {corr_ratio_aniso:.4f}, p = {p_ratio_aniso:.4e}")
        corr_abs_aniso, p_abs_aniso = pearsonr(anisotropy_vals, abs_vals)
        print(f"Корреляция анизотропии и абсолютной нормы первого токена: r = {corr_abs_aniso:.4f}, p = {p_abs_aniso:.4e}")
        corr_ratio_cos, p_ratio_cos = pearsonr(cosine_vals, ratio_vals)
        print(f"Корреляция среднего косинусного сходства и относительной нормы первого токена: r = {corr_ratio_cos:.4f}, p = {p_ratio_cos:.4e}")
        corr_abs_cos, p_abs_cos = pearsonr(cosine_vals, abs_vals)
        print(f"Корреляция среднего косинусного сходства и абсолютной нормы первого токена: r = {corr_abs_cos:.4f}, p = {p_abs_cos:.4e}")
    else:
        print("Недостаточно общих точек для вычисления корреляций.")

    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, cosine_means, marker='o', linestyle='-', color='b')
    dashed_vals = [cosine_means[i] for i in dashed_indices]
    plt.plot(dashed_indices, dashed_vals, 'r--', marker='o', markersize=6, color="blue", linewidth=2, label='layer-wise cosine anisotropy')
    plt.xticks(x_pos, x_labels_all, fontsize=20)#, rotation=90
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity", fontsize=20)
    plt.title(f"Cosine similarity of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
    plt.show()


    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:55], cosine_means[:55], marker='o', linestyle='-', color='b')
    plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity", fontsize=20)
    plt.title(f"Cosine similarity of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
    plt.show()
        


    

     # График 1.2: среднее косинусное сходство centered
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, cosine_means_centered, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Mean pairwise cosine similarity centered", fontsize=20)
    plt.title(f"Cosine similarity of hidden states centered – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage_centered.png", dpi=150)
    plt.show()

    # IsoScore
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, isoscores, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("IsoScore", fontsize=20)
    plt.title(f"IsoScore of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "isoscore.png", dpi=150)
    plt.show()

    
    # Plot 2: anisotropy
    print("component-wise singular", anisotropies)
    


    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, anisotropies, marker='s', linestyle='-', color='r')
    dashed_vals = [anisotropies[i] for i in dashed_indices]
    plt.plot(dashed_indices, dashed_vals, 'r--', marker='s', markersize=6, linewidth=2, label='layer-wise singular anisotropy')
    plt.xticks(x_pos, x_labels_all, fontsize=20)#, rotation=90
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Anisotropy (σ₁² / Σ σᵢ²)", fontsize=20)
    plt.title(f"Anisotropy of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
    plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:55], anisotropies[:55], marker='s', linestyle='-', color='r')
    plt.xticks(x_pos[:55], x_labels[:55], rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Anisotropy (σ₁² / Σ σᵢ²)", fontsize=20)
    plt.title(f"Anisotropy of hidden states – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
    plt.show()
        
    
    
    # Plot 3: effective dimensionality
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, participation_ratios, marker='^', linestyle='-', color='g')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.ylabel("Effective dimensionality", fontsize=20)
    plt.title(f"Effective dimensionality – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "participation_ratio_per_stage.png", dpi=150)
    plt.show()
    
    # Plot 4: rank
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, ranks, marker='d', linestyle='-', color='purple')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.ylabel("Rank (number of singular values > 1e-7)")
    plt.title(f"Rank of hidden states – {model.config._name_or_path}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "rank_per_stage.png", dpi=150)
    plt.show()
    
    # Plot 5: maximum singular value
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, max_sings, marker='d', linestyle='-', color='orange')
    plt.xticks(x_pos, x_labels, rotation=90, fontsize=20)
    plt.ylabel("Maximum singular value", fontsize=20)
    plt.title(f"Maximum singular value – {model.config._name_or_path}", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "max_singular_per_stage.png", dpi=150)
    plt.show()
    
    #  New plots for norm-based metrics 
    # Extract values for the same points as in all_points (check presence in norm_metrics)
    bos_is_max_values = [norm_metrics[k]['bos_is_max_ratio'] for k in all_points if k in norm_metrics]
    bos_norm_ratio_values = [norm_metrics[k]['bos_norm_ratio'] for k in all_points if k in norm_metrics]
    max_to_sum_values = [norm_metrics[k]['max_norm_to_sum_others_ratio'] for k in all_points if k in norm_metrics]
    
    # If some points are missing from norm_metrics, issue a warning but plot the available ones
    if len(bos_is_max_values) != len(all_points):
        print("Warning: norm metrics not collected for all points.")
    
    # Plot 6: fraction of examples where the BOS norm is maximum
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:len(bos_is_max_values)], bos_is_max_values, marker='o', linestyle='-', color='blue')
    plt.xticks(x_pos[:len(bos_is_max_values)], x_labels[:len(bos_is_max_values)], rotation=90)
    plt.ylabel("Fraction of examples where the first token norm is maximum")
    plt.title("Fraction of examples with maximum norm at BOS")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "bos_is_max_ratio.png", dpi=150)
    plt.show()
    
    # Plot 7: ratio of BOS norm to sum of all norms
    first_norm_abs_values = [norm_metrics[k]['first_norm_abs'] for k in all_points if k in norm_metrics]
    mean_norm_values = [norm_metrics[k]['mean_norm'] for k in all_points if k in norm_metrics]
    
    if len(first_norm_abs_values) > 0 and len(mean_norm_values) > 0:
        plt.figure(figsize=(20, 8))
        
        # Use the common length (assume lists have the same length)
        n = len(first_norm_abs_values)
        x_vals = x_pos[:n]
        x_labels_trim = x_labels[:n]
        
        plt.plot(x_vals, first_norm_abs_values, 
                 marker='s', linestyle='-', color='green', label='First token norm')
        plt.plot(x_vals, mean_norm_values, 
                 marker='.', linestyle='--', color='blue', alpha=0.7, label='Mean token norm')
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel("Token norm", fontsize=20)
        plt.xticks(x_vals, x_labels_trim, rotation=90, fontsize=20)
        plt.legend(fontsize=20)  
        plt.title("Absolute norm of the first token and mean token norm except first token", fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(plot_dir) / "first_norm_abs_with_mean.png", dpi=150)
        plt.show()
    else:
        print("Warning: insufficient data to plot first token norm and mean norm.")

        #  Plot: отношение нормы первого токена к сумме всех норм 
    if len(bos_norm_ratio_values) > 0:
        print("relative norm of FT", bos_norm_ratio_values)
        n = len(bos_norm_ratio_values)
        plt.figure(figsize=(20, 8))
        plt.plot(x_pos[:n], bos_norm_ratio_values, marker='o', linestyle='-', color='purple', linewidth=2)
        plt.xticks(x_pos[:n], x_labels[:n], rotation=90, fontsize=20)
        plt.ylabel("First token norm / sum of all norms", fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.title("Relative norm of the first token", fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_plots:
            plt.savefig(Path(plot_dir) / "first_norm_to_sum_ratio.png", dpi=150)
        plt.show()
    
    # Plot 8: ratio of maximum norm to the sum of the others
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos[:len(max_to_sum_values)], max_to_sum_values, marker='^', linestyle='-', color='red')
    plt.xticks(x_pos[:len(max_to_sum_values)], x_labels[:len(max_to_sum_values)], rotation=90)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("max(norm) / sum(other norms)")
    plt.title("Maximum norm dominance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "max_norm_to_sum_others.png", dpi=150)
    plt.show()

    #  Вывод сводки в консоль 
    print("\n" + "=" * 60)
    print("Сводка метрик по этапам:")
    for layer, point in all_points:
        m = metrics[(layer, point)]
        layer_label = f"{layer}" if layer < len(layers_container) else "Final"
        print(f"{layer_label:<4} {point:10s}: cosine={m['cosine_mean']:.4f}±{m['cosine_std']:.4f}, "
              f"anisotropy={m['anisotropy']:.4f}")


    generate_latex_table(metrics, len(layers_container), model, save_plots=save_plots, plot_dir=plot_dir if save_plots else None)
    # Возвращаем собранные метрики для возможного дальнейшего использования
    return {'metrics': metrics, 'norm_metrics': norm_metrics}

    #  Вывод сводки в консоль 
    print("\n" + "=" * 60)
    print("Сводка метрик по этапам:")
    for layer, point in all_points:
        m = metrics[(layer, point)]
        layer_label = f"{layer}" if layer < len(layers_container) else "Final"
        print(f"{layer_label:<4} {point:10s}: cosine={m['cosine_mean']:.4f}±{m['cosine_std']:.4f}, "
              f"anisotropy={m['anisotropy']:.4f}")

    # Возвращаем собранные метрики
    return {'metrics': metrics, 'norm_metrics': norm_metrics}

    # Извлечение метрик
    cosine_means = [metrics[k]['cosine_mean'] for k in all_points]
    anisotropies = [metrics[k]['anisotropy'] for k in all_points]
    part_ratios = [metrics[k]['participation_ratio'] for k in all_points]

    # График 1: среднее косинусное сходство
    plt.figure(figsize=(14, 6))
    plt.plot(x_pos, cosine_means, marker='o', linestyle='-', color='b')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.xlabel("Этап (слой + точка)")
    plt.ylabel("Среднее попарное косинусное сходство")
    plt.title(f"Косинусная близость скрытых состояний – {config.model_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "cosine_mean_per_stage.png", dpi=150)
        print("График косинусной близости сохранён.")
    plt.show()

    # График 2: анизотропия
    plt.figure(figsize=(14, 6))
    plt.plot(x_pos, anisotropies, marker='s', linestyle='-', color='r')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.xlabel("Этап (слой + точка)")
    plt.ylabel("Анизотропия (σ₁² / Σ σᵢ²)")
    plt.title(f"Анизотропия скрытых состояний – {config.model_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "anisotropy_per_stage.png", dpi=150)
        print("График анизотропии сохранён.")
    plt.show()

    # График 3: эффективная размерность
    plt.figure(figsize=(14, 6))
    plt.plot(x_pos, part_ratios, marker='^', linestyle='-', color='g')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.xlabel("Этап (слой + точка)")
    plt.ylabel("Эффективная размерность (participation ratio)")
    plt.title(f"Эффективная размерность скрытых состояний – {config.model_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "participation_ratio_per_stage.png", dpi=150)
        print("График эффективной размерности сохранён.")
    plt.show()

    #  Вывод сводки 
    print("\n" + "=" * 60)
    print("Сводка метрик по этапам:")
    for layer, point in all_points:
        m = metrics[(layer, point)]
        layer_label = str(layer) if layer < len(layers_container) else 'Final'
        print(f"{layer_label:<4} {point:10s}: cosine={m['cosine_mean']:.4f}±{m['cosine_std']:.4f}, "
              f"anisotropy={m['anisotropy']:.4f}, part. ratio={m['participation_ratio']:.2f}")

    generate_latex_table(metrics, len(layers_container), model, save_plots=save_plots, plot_dir=plot_dir if save_plots else None)
    return metrics





def analyze_intrinsic_dimension_distribution(
    model, batch, config,
    layers_to_plot=None,
    save_plots=False,
    k=10,
    num_samples=None
):
    """
    Расширенный анализ внутренней размерности (ID) для моделей Qwen2.5.
    Собирает скрытые состояния на 7 этапах внутри каждого decoder layer:
        - вход в слой (input)
        - после input_layernorm (ln1)
        - после self_attention (attn)
        - после первого residual (res1)
        - после post_attention_layernorm (ln2)
        - после MLP (mlp)
        - после второго residual / выход слоя (res2)
    Для каждого этапа вычисляет внутреннюю размерность методом Левиной–Бикеля.
    Строит график ID по всем этапам всех слоёв.
    Дополнительно добавляет два финальных этапа:
        - после final_layernorm (перед lm_head)
        - после lm_head (логиты)
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from intrinsic import intrinsic_dimension_levina_bickel_vectorized
    from save import setup_plot_saving, save_checkpoint_simple, manage_last_checkpoints

    # Определяем устройство и перемещаем batch
    model_device = next(model.parameters()).device
    print("model_device", model_device)
    if batch['input_ids'].device != model_device:
        print(f"Перемещаем данные с {batch['input_ids'].device} на {model_device}")
        batch = {k: v.to(model_device) for k, v in batch.items()}
        print(f"Модель на устройстве: {next(model.parameters()).device}")
        print(f"Batch на устройстве: {batch['input_ids'].device}")

    pad_token_id = 151643  # часто используемый для Qwen2.5

    # Определяем список анализируемых слоёв
    if layers_to_plot is None:
        layers_to_plot = list(range(10, model.config.num_hidden_layers + 1 - 9, 1))

    # Директория для сохранения графиков
    plot_dir = setup_plot_saving(config, suffix="intrinsic_dim_analysis")
    print(f"Графики будут сохранены в: {plot_dir}")

    # - Точки внутри слоя и их сокращения -
    POINTS = ['ln1', 'attn', 'res1', 'ln2', 'mlp', 'res2']
    POINT_ABBR = {
        'ln1': 'RMSNorm 1',
        'attn': 'attention',
        'res1': 'residual connection 1',
        'ln2': 'RMSNorm 2',
        'mlp': 'FNN',
        'res2': 'residual connection 2'
    }

    # Контейнер для результатов
    metrics = {}

    # Параметр подвыборки
    if num_samples is None:
        num_samples = config.num_samples

    # - Функция обработки одного тензора (вычисление ID) -
    def compute_id(tensor_hidden, input_ids_flat, pad_token_id, num_samples, k):
        """
        tensor_hidden: (batch*seq_len, hidden_dim) на GPU
        input_ids_flat: (batch*seq_len,) на GPU
        Возвращает словарь с внутренней размерностью или None, если данных недостаточно.
        """
        valid_mask = (input_ids_flat != pad_token_id)
        embeddings = tensor_hidden[valid_mask]         # [N_valid, hidden_dim]

        if embeddings.shape[0] == 0:
            return None

        # Подвыборка для ускорения
        if embeddings.shape[0] > num_samples:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:num_samples]
            embeddings = embeddings[idx]

        # Преобразуем в numpy для оценки ID
        embeddings_np = embeddings.cpu().float().numpy()

        # Проверка на минимальное количество точек
        if embeddings_np.shape[0] <= k + 1:
            return None

        # Вычисление ID
        id_value = intrinsic_dimension_levina_bickel_vectorized(embeddings_np, k=k)

        return {'intrinsic_dimension': id_value}

    # - Регистрация хуков для каждого слоя -
    handles = []

    # Определяем, где находятся decoder layers
    if hasattr(model, 'layers') and isinstance(model.layers, torch.nn.ModuleList):
        layers_container = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_container = model.model.layers
    else:
        raise AttributeError("Не удалось найти decoder layers модели. Ожидается model.layers или model.model.layers")

    print(f"Найдено слоёв: {len(layers_container)}")
    print("=" * 60)

    input_ids_flat = batch['input_ids'].view(-1)  # [batch*seq_len] на устройстве

    model.eval()
    with torch.no_grad():
        for layer_idx in layers_to_plot:
            if layer_idx >= len(layers_container):
                print(f"Пропускаем слой {layer_idx}, так как его нет в модели")
                continue

            layer = layers_container[layer_idx]

            residual = None
            res1_state = None

            # - Pre-hook на самом слое: сохраняем вход -
            def create_pre_hook(l_idx, point):
                def pre_hook(module, args):
                    nonlocal residual
                    residual = args[0]
                    hidden_flat = residual.view(-1, residual.shape[-1])
                    met = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                    return args
                return pre_hook

            pre_handle = layer.register_forward_pre_hook(create_pre_hook(layer_idx, 'input'))
            handles.append(pre_handle)

            # - Hook на input_layernorm -
            def create_ln1_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                return hook

            ln1_handle = layer.input_layernorm.register_forward_hook(create_ln1_hook(layer_idx, 'ln1'))
            handles.append(ln1_handle)

            # - Hook на self_attn и вычисление res1 -
            def create_attn_hook(l_idx, point_attn, point_res1):
                def hook(module, input, output):
                    nonlocal residual, res1_state
                    attn_out = output[0] if isinstance(output, tuple) else output
                    hidden_flat = attn_out.view(-1, attn_out.shape[-1])
                    met_attn = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met_attn is not None:
                        metrics[(l_idx, point_attn)] = met_attn

                    if residual is not None:
                        res1 = residual + attn_out
                        res1_flat = res1.view(-1, res1.shape[-1])
                        met_res1 = compute_id(res1_flat, input_ids_flat, pad_token_id, num_samples, k)
                        if met_res1 is not None:
                            metrics[(l_idx, point_res1)] = met_res1
                        res1_state = res1
                return hook

            attn_handle = layer.self_attn.register_forward_hook(
                create_attn_hook(layer_idx, 'attn', 'res1')
            )
            handles.append(attn_handle)

            # - Hook на post_attention_layernorm -
            def create_ln2_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                return hook

            ln2_handle = layer.post_attention_layernorm.register_forward_hook(
                create_ln2_hook(layer_idx, 'ln2')
            )
            handles.append(ln2_handle)

            # - Hook на mlp и вычисление res2 -
            def create_mlp_hook(l_idx, point_mlp, point_res2):
                def hook(module, input, output):
                    nonlocal res1_state
                    mlp_out = output
                    hidden_flat = mlp_out.view(-1, mlp_out.shape[-1])
                    met_mlp = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met_mlp is not None:
                        metrics[(l_idx, point_mlp)] = met_mlp

                    if res1_state is not None:
                        res2 = res1_state + mlp_out
                        res2_flat = res2.view(-1, res2.shape[-1])
                        met_res2 = compute_id(res2_flat, input_ids_flat, pad_token_id, num_samples, k)
                        if met_res2 is not None:
                            metrics[(l_idx, point_res2)] = met_res2
                return hook

            mlp_handle = layer.mlp.register_forward_hook(
                create_mlp_hook(layer_idx, 'mlp', 'res2')
            )
            handles.append(mlp_handle)

        # - Хуки на финальные слои -
        final_norm = None
        if hasattr(model, 'norm'):
            final_norm = model.norm
        elif hasattr(model, 'model') and hasattr(model.model, 'norm'):
            final_norm = model.model.norm

        if final_norm is not None:
            def create_final_norm_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    met = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                return hook
            final_layer_idx = len(layers_container)
            norm_handle = final_norm.register_forward_hook(create_final_norm_hook(final_layer_idx, 'final_norm'))
            handles.append(norm_handle)
            print("Хук на final_layernorm зарегистрирован.")
        else:
            print("Предупреждение: не найден final_layernorm")

        lm_head = None
        if hasattr(model, 'lm_head'):
            lm_head = model.lm_head
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            lm_head = model.model.lm_head

        if lm_head is not None:
            def create_lm_head_hook(l_idx, point):
                def hook(module, input, output):
                    logits = output[0] if isinstance(output, tuple) else output
                    hidden_flat = logits.view(-1, logits.shape[-1])
                    met = compute_id(hidden_flat, input_ids_flat, pad_token_id, num_samples, k)
                    if met is not None:
                        metrics[(l_idx, point)] = met
                return hook
            lm_head_idx = len(layers_container) + 1
            lm_head_handle = lm_head.register_forward_hook(create_lm_head_hook(lm_head_idx, 'lm_head'))
            handles.append(lm_head_handle)
            print("Хук на lm_head зарегистрирован.")
        else:
            print("Предупреждение: не найден lm_head")

        # - Запуск forward pass -
        print("Запуск forward pass для сбора промежуточных состояний...")
        _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=False
        )
        print("Forward pass завершён.")

    # - Удаление хуков -
    for h in handles:
        h.remove()
    print(f"Удалено {len(handles)} хуков.")

    # - Построение графика -
    if not metrics:
        print("Не удалось собрать метрики. Проверьте модель и входные данные.")
        return {}

    # Формирование упорядоченного списка точек для графика
    layer_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] < len(layers_container)))
    final_numbers = sorted(set(k[0] for k in metrics.keys() if isinstance(k[0], int) and k[0] >= len(layers_container)))

    all_points = []
    for layer in layer_numbers:
        for point in POINTS:
            key = (layer, point)
            if key in metrics:
                all_points.append(key)

    final_desired_order = ['final_norm', 'lm_head']
    for point_name in final_desired_order:
        for layer in final_numbers:
            key = (layer, point_name)
            if key in metrics:
                all_points.append(key)
                break

    # Подписи оси X
    x_labels = []
    for (layer, point) in all_points:
        if layer < len(layers_container):
            x_labels.append(f"{layer} {POINT_ABBR.get(point, point)}")
        else:
            if point == 'final_norm':
                x_labels.append("Final LayerNorm")
            elif point == 'lm_head':
                x_labels.append("LM Head")
            else:
                x_labels.append(f"Final {point}")
    x_pos = np.arange(len(all_points))

    # Извлекаем значения ID
    id_values = [metrics[k]['intrinsic_dimension'] for k in all_points]

    # График внутренней размерности
    plt.figure(figsize=(14, 6))
    plt.plot(x_pos, id_values, marker='o', linestyle='-', color='purple')
    plt.xticks(x_pos, x_labels, rotation=90)
    plt.xlabel("Этап (слой + точка)")
    plt.ylabel("Внутренняя размерность (ID)")
    plt.title(f"Внутренняя размерность скрытых состояний – {config.model_name} (k={k})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "intrinsic_dim_per_stage.png", dpi=150)
        print(f"График внутренней размерности сохранён.")
    plt.show()

    # - Вывод сводки в консоль -
    print("\n" + "=" * 60)
    print("Сводка внутренней размерности по этапам:")
    for (layer, point), val in zip(all_points, id_values):
        layer_str = f"{layer:<3}" if layer < len(layers_container) else "Final"
        print(f"{layer_str} {point:10s}: ID = {val:.2f}")

    # Дополнительная статистика
    if len(id_values) > 0:
        print("\nСтатистика:")
        print(f"  Среднее ID: {np.mean(id_values):.2f} ± {np.std(id_values):.2f}")
        print(f"  Минимум: {np.min(id_values):.2f}")
        print(f"  Максимум: {np.max(id_values):.2f}")

    return metrics






import torch
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib.pyplot as plt

def compute_correlations_attention_weights(model, batch, config, layers_to_plot=None):

    """
    Вычисляет корреляции Пирсона между свойствами весовых матриц внимания
    (W_V, W_Q, W_K, W_Q^T W_K) и средним косинусным сходством (mean cosine)
    скрытых состояний после слоя attention.

    Для каждого слоя:
        - получает метрики после attention (среднее косинусное сходство)
        - извлекает веса W_V, W_Q, W_K из объединённой матрицы query_key_value
        - вычисляет для каждой матрицы спектральную норму (наибольшее сингулярное число)
          и сингулярную анизотропию (σ₁² / Σσᵢ²)
        - аналогично для произведения W_Q^T W_K

    Затем по всем слоям считает корреляцию Пирсона между:
        1) spectral_norm(W_V)  и cosine_mean(attn)
        2) anisotropy(W_V)     и cosine_mean(attn)
        3) spectral_norm(W_Q)  и cosine_mean(attn)
        4) anisotropy(W_Q)     и cosine_mean(attn)
        5) spectral_norm(W_K)  и cosine_mean(attn)
        6) anisotropy(W_K)     и cosine_mean(attn)
        7) spectral_norm(W_Q^T W_K) и cosine_mean(attn)
        8) anisotropy(W_Q^T W_K)    и cosine_mean(attn)

    Параметры:
        model          – загруженная модель Pythia (GPT-NeoX)
        batch          – словарь с 'input_ids', 'attention_mask' (уже на нужном устройстве)
        config         – объект с атрибутами:
                         num_samples    – число случайных токенов для compute_metrics
                         model_name     – имя модели (для подписей графиков)
                         (также может содержать bos_token_id, eos_token_id)
        layers_to_plot – список индексов слоёв для анализа (по умолчанию все слои)

    Возвращает:
        dict {
            'layer_data': список словарей с данными по каждому слою,
            'correlations': словарь с коэффициентами корреляции и p-value
        }
    """
    #  Устройство и pad_token_id 
    device = next(model.parameters()).device
    if batch['input_ids'].device != device:
        batch = {k: v.to(device) for k, v in batch.items()}

    pad_token_id = getattr(config, 'eos_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0
    bos_token_id = getattr(config, 'bos_token_id', pad_token_id)  # не используется, но оставим

    #  Определяем слои 
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        layers = model.gpt_neox.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise AttributeError("Не найдены decoder layers (ожидается model.gpt_neox.layers или model.layers)")

    num_layers = len(layers)
    if layers_to_plot is None:
        layers_to_plot = list(range(num_layers))
    else:
        layers_to_plot = [l for l in layers_to_plot if l < num_layers]

    print(f"Анализ слоёв: {layers_to_plot}")

    #  Вспомогательная функция compute_metrics (как в исходном коде) 
    def compute_metrics(tensor_hidden, input_ids_flat, pad_token_id, num_samples):
        valid_mask = (input_ids_flat != pad_token_id)
        embeddings = tensor_hidden[valid_mask]
        if embeddings.shape[0] == 0:
            return None
        if embeddings.shape[0] > num_samples:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:num_samples]
            embeddings = embeddings[idx]

        # Косинусное сходство
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / torch.clamp(norms, min=1e-8)
        cos_matrix = normalized @ normalized.T
        n = cos_matrix.shape[0]
        triu = torch.triu_indices(n, n, offset=1, device=cos_matrix.device)
        cos_vals = cos_matrix[triu[0], triu[1]]
        cosine_mean = cos_vals.mean().item()
        cosine_std = cos_vals.std().item()

        # Анизотропия через SVD
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(embeddings.float())
        s_sq = s ** 2
        sum_sq = s_sq.sum()
        anisotropy = (s_sq[0] / sum_sq).item() if sum_sq > 0 else 0.0
        participation_ratio = (sum_sq ** 2 / (s_sq ** 2).sum()).item() if sum_sq > 0 else 0.0
        rank = (s > 1e-7).sum().item()
        max_sing = s[0].item() if s.numel() > 0 else 0.0

        return {
            'cosine_mean': cosine_mean,
            'cosine_std': cosine_std,
            'anisotropy': anisotropy,
            'participation_ratio': participation_ratio,
            'rank': rank,
            'max_sing': max_sing
        }

    #  Сбор метрик после attention через хуки 
    handles = []
    attn_metrics = {}          # (layer_idx, 'attn') -> метрики
    input_ids_flat = batch['input_ids'].view(-1)

    model.eval()
    with torch.no_grad():
        for layer_idx in layers_to_plot:
            layer = layers[layer_idx]

            def create_attn_hook(l_idx):
                def hook(module, inp, out):
                    # out может быть кортежем (hidden_states, ...)
                    attn_out = out[0] if isinstance(out, tuple) else out
                    hidden_flat = attn_out.view(-1, attn_out.shape[-1])
                    met = compute_metrics(hidden_flat, input_ids_flat, pad_token_id, config.num_samples)
                    if met is not None:
                        attn_metrics[(l_idx, 'attn')] = met
                return hook

            if hasattr(layer, 'attention'):
                handle = layer.attention.register_forward_hook(create_attn_hook(layer_idx))
                handles.append(handle)
            else:
                print(f"Слой {layer_idx} не имеет attention, пропускаем")

        # Запуск forward pass
        print("Запуск forward pass для сбора метрик после attention...")
        _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=False
        )
        print("Forward завершён.")

        # Удаление хуков
        for h in handles:
            h.remove()
        print(f"Удалено {len(handles)} хуков.")

    if not attn_metrics:
        raise RuntimeError("Не удалось собрать метрики после attention. Проверьте модель и хуки.")

    #  Извлечение весовых матриц и вычисление их свойств 
    layer_data = []  # список словарей для каждого слоя

    for layer_idx in layers_to_plot:
        layer = layers[layer_idx]
        if not hasattr(layer, 'attention'):
            continue

        # Получаем объединённую матрицу query_key_value
        qkv_weight = layer.attention.query_key_value.weight  # форма: (hidden_size, 3*hidden_size) или (3*hidden_size, hidden_size)
        # Определим по размерности:
        if qkv_weight.shape[0] == qkv_weight.shape[1] // 3:
            # случай (hidden_size, 3*hidden_size)
            hidden_size = qkv_weight.shape[0]
            W_Q = qkv_weight[:, :hidden_size]
            W_K = qkv_weight[:, hidden_size:2*hidden_size]
            W_V = qkv_weight[:, 2*hidden_size:]
        elif qkv_weight.shape[1] == qkv_weight.shape[0] // 3:
            # случай (3*hidden_size, hidden_size)
            hidden_size = qkv_weight.shape[1]
            W_Q = qkv_weight[:hidden_size, :]
            W_K = qkv_weight[hidden_size:2*hidden_size, :]
            W_V = qkv_weight[2*hidden_size:, :]
        else:
            raise ValueError(f"Неожиданная форма qkv_weight: {qkv_weight.shape}")

        # Переводим в float32 для SVD (если не double)
        W_Q = W_Q.float()
        W_K = W_K.float()
        W_V = W_V.float()

        # Функция для вычисления спектральной нормы и анизотропии матрицы
        def matrix_properties(mat):
            # mat: 2D тензор
            s = torch.linalg.svdvals(mat)  # сингулярные числа
            s_sq = s ** 2
            sum_sq = s_sq.sum().item()
            spectral_norm = s[0].item()
            anisotropy = (s_sq[0].item() / sum_sq) if sum_sq > 0 else 0.0
            return spectral_norm, anisotropy

        spec_V, aniso_V = matrix_properties(W_V)
        spec_Q, aniso_Q = matrix_properties(W_Q)
        spec_K, aniso_K = matrix_properties(W_K)

        # Произведение W_Q^T W_K (предполагаем, что матрицы квадратные)
        W_Q_T_W_K = W_Q.T @ W_K   # (hidden_size, hidden_size)
        spec_QK, aniso_QK = matrix_properties(W_Q_T_W_K)

        # Получаем метрики после attention для этого слоя
        key = (layer_idx, 'attn')
        if key not in attn_metrics:
            print(f"Предупреждение: нет метрик для слоя {layer_idx}, пропускаем")
            continue
        attn_met = attn_metrics[key]
        attn_cosine = attn_met['cosine_mean']

        layer_data.append({
            'layer': layer_idx,
            'attn_cosine': attn_cosine,
            'spec_V': spec_V,
            'aniso_V': aniso_V,
            'spec_Q': spec_Q,
            'aniso_Q': aniso_Q,
            'spec_K': spec_K,
            'aniso_K': aniso_K,
            'spec_QK': spec_QK,
            'aniso_QK': aniso_QK,
        })

    if len(layer_data) < 2:
        raise ValueError(f"Недостаточно слоёв с данными ({len(layer_data)}) для вычисления корреляций.")

    #  Вычисление корреляций Пирсона 
    def corr(x_list, y_list):
        # x_list, y_list – списки чисел
        r, p = pearsonr(x_list, y_list)
        return r, p

    # Собираем векторы
    attn_cosine_vals = [d['attn_cosine'] for d in layer_data]
    spec_V_vals = [d['spec_V'] for d in layer_data]
    aniso_V_vals = [d['aniso_V'] for d in layer_data]
    spec_Q_vals = [d['spec_Q'] for d in layer_data]
    aniso_Q_vals = [d['aniso_Q'] for d in layer_data]
    spec_K_vals = [d['spec_K'] for d in layer_data]
    aniso_K_vals = [d['aniso_K'] for d in layer_data]
    spec_QK_vals = [d['spec_QK'] for d in layer_data]
    aniso_QK_vals = [d['aniso_QK'] for d in layer_data]

    correlations = {
        'spec_V_vs_attn_cosine': corr(spec_V_vals, attn_cosine_vals),
        'aniso_V_vs_attn_cosine': corr(aniso_V_vals, attn_cosine_vals),
        'spec_Q_vs_attn_cosine': corr(spec_Q_vals, attn_cosine_vals),
        'aniso_Q_vs_attn_cosine': corr(aniso_Q_vals, attn_cosine_vals),
        'spec_K_vs_attn_cosine': corr(spec_K_vals, attn_cosine_vals),
        'aniso_K_vs_attn_cosine': corr(aniso_K_vals, attn_cosine_vals),
        'spec_QK_vs_attn_cosine': corr(spec_QK_vals, attn_cosine_vals),
        'aniso_QK_vs_attn_cosine': corr(aniso_QK_vals, attn_cosine_vals),
    }

    # Вывод результатов
    print("\n" + "=" * 60)
    print("Корреляции Пирсона (r, p-value):")
    for name, (r, p) in correlations.items():
        print(f"{name:30s}: r = {r:.4f}, p = {p:.4e}")

    # Опционально: построить графики рассеяния для наглядности
    try:
        plot_dir = setup_plot_saving(config, suffix="weight_correlations_cosine")
        n_pairs = len(correlations)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        for i, (name, (r, p)) in enumerate(correlations.items()):
            ax = axes[i]
            # Определяем переменную по оси X
            if 'spec_V' in name:
                x_vals = spec_V_vals
                x_label = 'spec_V'
            elif 'aniso_V' in name:
                x_vals = aniso_V_vals
                x_label = 'aniso_V'
            elif 'spec_Q' in name:
                x_vals = spec_Q_vals
                x_label = 'spec_Q'
            elif 'aniso_Q' in name:
                x_vals = aniso_Q_vals
                x_label = 'aniso_Q'
            elif 'spec_K' in name:
                x_vals = spec_K_vals
                x_label = 'spec_K'
            elif 'aniso_K' in name:
                x_vals = aniso_K_vals
                x_label = 'aniso_K'
            elif 'spec_QK' in name:
                x_vals = spec_QK_vals
                x_label = 'spec_QK'
            elif 'aniso_QK' in name:
                x_vals = aniso_QK_vals
                x_label = 'aniso_QK'
            else:
                continue
            ax.scatter(x_vals, attn_cosine_vals, alpha=0.7)
            ax.set_xlabel(x_label)
            ax.set_ylabel('attn cosine mean')
            ax.set_title(f'{name}\nr={r:.3f}, p={p:.3f}')
            ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if plot_dir:
            plt.savefig(Path(plot_dir) / "correlation_scatter_cosine.png", dpi=150)
        plt.show()
    except Exception as e:
        print(f"Не удалось построить графики рассеяния: {e}")

    return {
        'layer_data': layer_data,
        'correlations': correlations
    }

def setup_plot_saving(config, suffix=""):
    from pathlib import Path
    model_name = getattr(config, 'model_name', 'model')
    plot_dir = Path(f"plots/{model_name}_{suffix}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir



import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Предполагается, что функция intrinsic_dimension_levina_bickel_vectorized уже определена
# (см. код в вопросе). При необходимости её можно скопировать сюда.

def analyze_intrinsic_dimension_pythia(
    model,
    batch,
    config,
    k=10,
    max_points=5000,
    layers_to_plot=None,
    save_plots=False,
    ALL=False
):
    """
    Вычисление внутренней размерности (ID) методом Левины‑Бикеля для скрытых состояний
    на разных этапах внутри каждого decoder layer модели Pythia (GPT‑NeoX).
    
    Собираются состояния на этапах:
        - вход в слой (input) – только для нулевого слоя
        - после input_layernorm (ln1)
        - после attention (attn)
        - после первого residual (res1)
        - после post_attention_layernorm (ln2)
        - после MLP (mlp)
        - после второго residual (res2)
        - после final_layer_norm (final_norm)
        - после lm_head (lm_head)
    
    Для каждого этапа вычисляется intrinsic dimension.
    
    Параметры:
    -
    model : torch.nn.Module
        Модель Pythia (GPT‑NeoX), должна иметь структуру model.gpt_neox.layers.
    batch : dict
        Словарь с ключами 'input_ids', 'attention_mask' (и возможно 'labels').
    config : object
        Объект конфигурации модели, должен содержать:
            - num_hidden_layers (или n_layer)
            - eos_token_id / pad_token_id / bos_token_id
            - _name_or_path (для заголовков графиков)
    k : int, default=10
        Число ближайших соседей в MLE оценке Левины‑Бикеля.
    max_points : int, default=5000
        Максимальное количество токенов, используемых для оценки ID
        (при большем числе производится случайная подвыборка).
    layers_to_plot : list, optional
        Список индексов слоёв для анализа (по умолчанию все).
    save_plots : bool, default=False
        Сохранять ли графики в файлы.
    ALL : bool, default=False
        Не используется, оставлено для совместимости с сигнатурой первой функции.
    
    Возвращает:
    -
    id_metrics : dict
        Словарь с ключами (layer_idx, point_name) и значениями – оценка ID.
    """
    #  Устройство и данные 
    model_device = next(model.parameters()).device
    print("model_device", model_device)
    if batch['input_ids'].device != model_device:
        print(f"Перемещаем данные с {batch['input_ids'].device} на {model_device}")
        batch = {k: v.to(model_device) for k, v in batch.items()}

    #  Pad token id (обычно eos_token_id = 0) 
    pad_token_id = getattr(config, 'eos_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0
    print(f"Используется pad_token_id = {pad_token_id}")

    #  BOS token id (не используется в ID, но может понадобиться для масок) 
    bos_token_id = getattr(config, 'bos_token_id', getattr(config, 'eos_token_id', 0))
    if bos_token_id is None:
        bos_token_id = 0

    #  Определяем список слоёв для анализа 
    num_layers = model.config.num_hidden_layers
    if layers_to_plot is None:
        layers_to_plot = list(range(0, num_layers + 1))  # все слои + финальные

    # Директория для сохранения графиков
    if save_plots:
        from your_utils import setup_plot_saving  # замените на свою функцию или реализуйте
        plot_dir = setup_plot_saving(config, suffix="intrinsic_dimension_pythia")
        print(f"Графики будут сохранены в: {plot_dir}")

    #  Точки внутри слоя и их подписи 
    POINTS = ['input', 'ln1', 'attn', 'res1', 'ln2', 'mlp', 'res2']
    POINT_ABBR = {
        'input': 'input',
        'ln1': 'LayerNorm 1',
        'attn': 'attention',
        'res1': 'residual 1',
        'ln2': 'LayerNorm 2',
        'mlp': 'MLP',
        'res2': 'residual 2'
    }

    #  Контейнер для результатов 
    id_metrics = {}

    #  Вспомогательная функция для вычисления ID по тензору эмбеддингов 
    def compute_id(tensor_hidden, input_ids_flat, pad_token_id, max_pts, k_neighbors):
        """
        tensor_hidden : (batch*seq_len, hidden_dim) на GPU
        input_ids_flat : (batch*seq_len,) на GPU
        Возвращает float – intrinsic dimension (0, если данных недостаточно)
        """
        valid_mask = (input_ids_flat != pad_token_id)
        embeddings = tensor_hidden[valid_mask]

        if embeddings.shape[0] == 0:
            return 0.0

        # Подвыборка для ускорения
        if embeddings.shape[0] > max_pts:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:max_pts]
            embeddings = embeddings[idx]

        # При слишком малом числе точек вернуть 0
        if embeddings.shape[0] <= k_neighbors + 1:
            return 0.0

        # Вычисление ID (результат – число)
        embeddings = embeddings.float()
        dim = intrinsic_dimension_levina_bickel_vectorized(embeddings, k=k_neighbors)
        return dim if not np.isnan(dim) else 0.0

    #  Регистрация хуков 
    handles = []

    # Доступ к списку слоёв: для Pythia это model.gpt_neox.layers
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        layers_container = model.gpt_neox.layers
    elif hasattr(model, 'layers'):
        layers_container = model.layers
    else:
        raise AttributeError("Не удалось найти decoder layers. Ожидается model.gpt_neox.layers или model.layers")

    print(f"Найдено слоёв: {len(layers_container)}")
    print("=" * 60)

    # Плоский тензор input_ids для фильтрации паддингов в compute_id
    input_ids_flat = batch['input_ids'].view(-1)

    model.eval()
    with torch.no_grad():
        for layer_idx in layers_to_plot:
            if layer_idx >= len(layers_container):
                print(f"Пропускаем слой {layer_idx} (нет в модели)")
                continue

            layer = layers_container[layer_idx]

            # Переменные для сохранения residual между хуками (замыкание)
            residual = None
            res1_state = None

            # - Pre-hook на входе слоя: сохраняем вход (точка 'input') -
            def create_pre_hook(l_idx, point):
                def pre_hook(module, args):
                    nonlocal residual
                    residual = args[0]  # hidden_states
                    hidden_flat = residual.view(-1, residual.shape[-1])
                    id_val = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                    if id_val > 0:
                        id_metrics[(l_idx, point)] = id_val
                    else:
                        # можно сохранить 0, но он может быть артефактом
                        id_metrics[(l_idx, point)] = id_val
                    return args
                return pre_hook

            # Добавляем input только для нулевого слоя (как в первой функции)
            if layer_idx == 0:
                pre_handle = layer.register_forward_pre_hook(create_pre_hook(layer_idx, 'input'))
                handles.append(pre_handle)

            # - Hook на input_layernorm (ln1) -
            def create_ln1_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    id_val = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                    if id_val > 0:
                        id_metrics[(l_idx, point)] = id_val
                    else:
                        id_metrics[(l_idx, point)] = id_val
                return hook

            if hasattr(layer, 'input_layernorm'):
                ln1_handle = layer.input_layernorm.register_forward_hook(create_ln1_hook(layer_idx, 'ln1'))
                handles.append(ln1_handle)
            else:
                print(f"Слой {layer_idx} не имеет input_layernorm")

            # - Hook на attention (attn) и residual 1 (res1) -
            def create_attn_hook(l_idx, point_attn, point_res1):
                def hook(module, input, output):
                    nonlocal residual, res1_state
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # ID для attention
                    hidden_flat = attn_out.view(-1, attn_out.shape[-1])
                    id_attn = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                    id_metrics[(l_idx, point_attn)] = id_attn

                    # Вычисляем residual 1 и его ID
                    if residual is not None:
                        res1 = residual + attn_out
                        res1_flat = res1.view(-1, res1.shape[-1])
                        id_res1 = compute_id(res1_flat, input_ids_flat, pad_token_id, max_points, k)
                        id_metrics[(l_idx, point_res1)] = id_res1
                        res1_state = res1
                return hook

            if hasattr(layer, 'attention'):
                attn_handle = layer.attention.register_forward_hook(
                    create_attn_hook(layer_idx, 'attn', 'res1')
                )
                handles.append(attn_handle)
            else:
                print(f"Слой {layer_idx} не имеет attention")

            # - Hook на post_attention_layernorm (ln2) -
            def create_ln2_hook(l_idx, point):
                def hook(module, input, output):
                    hidden_flat = output.view(-1, output.shape[-1])
                    id_val = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                    id_metrics[(l_idx, point)] = id_val
                return hook

            if hasattr(layer, 'post_attention_layernorm'):
                ln2_handle = layer.post_attention_layernorm.register_forward_hook(create_ln2_hook(layer_idx, 'ln2'))
                handles.append(ln2_handle)
            else:
                print(f"Слой {layer_idx} не имеет post_attention_layernorm")

            # - Hook на MLP (mlp) и residual 2 (res2) -
            def create_mlp_hook(l_idx, point_mlp, point_res2):
                def hook(module, input, output):
                    nonlocal res1_state
                    mlp_out = output
                    # ID для MLP
                    hidden_flat = mlp_out.view(-1, mlp_out.shape[-1])
                    id_mlp = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                    id_metrics[(l_idx, point_mlp)] = id_mlp

                    # Вычисляем residual 2 и его ID
                    if res1_state is not None:
                        res2 = res1_state + mlp_out
                        res2_flat = res2.view(-1, res2.shape[-1])
                        id_res2 = compute_id(res2_flat, input_ids_flat, pad_token_id, max_points, k)
                        id_metrics[(l_idx, point_res2)] = id_res2
                return hook

            if hasattr(layer, 'mlp'):
                mlp_handle = layer.mlp.register_forward_hook(
                    create_mlp_hook(layer_idx, 'mlp', 'res2')
                )
                handles.append(mlp_handle)
            else:
                print(f"Слой {layer_idx} не имеет mlp")

        #  Финальные этапы: final_layer_norm и lm_head 
        # Добавляем их только если в layers_to_plot есть индексы за пределами слоёв
        if max(layers_to_plot) >= num_layers:
            # Final LayerNorm
            final_norm = None
            if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
                final_norm = model.gpt_neox.final_layer_norm
            elif hasattr(model, 'final_layer_norm'):
                final_norm = model.final_layer_norm

            if final_norm is not None:
                def create_final_norm_hook(l_idx, point):
                    def hook(module, input, output):
                        hidden_flat = output.view(-1, output.shape[-1])
                        id_val = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                        id_metrics[(l_idx, point)] = id_val
                    return hook
                final_layer_idx = num_layers  # индекс за последним слоем
                norm_handle = final_norm.register_forward_hook(create_final_norm_hook(final_layer_idx, 'final_norm'))
                handles.append(norm_handle)
                print("Хук на final_layer_norm зарегистрирован.")
            else:
                print("Предупреждение: не найден final_layer_norm")

            # LM Head
            lm_head = None
            if hasattr(model, 'lm_head'):
                lm_head = model.lm_head
            elif hasattr(model, 'embed_out'):
                lm_head = model.embed_out

            if lm_head is not None:
                def create_lm_head_hook(l_idx, point):
                    def hook(module, input, output):
                        logits = output[0] if isinstance(output, tuple) else output
                        hidden_flat = logits.view(-1, logits.shape[-1])
                        id_val = compute_id(hidden_flat, input_ids_flat, pad_token_id, max_points, k)
                        id_metrics[(l_idx, point)] = id_val
                    return hook
                lm_head_idx = num_layers + 1
                lm_head_handle = lm_head.register_forward_hook(create_lm_head_hook(lm_head_idx, 'lm_head'))
                handles.append(lm_head_handle)
                print("Хук на lm_head зарегистрирован.")
            else:
                print("Предупреждение: не найден lm_head")

        #  Один forward pass 
        print("Запуск forward pass для сбора промежуточных состояний...")
        _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=False
        )
        print("Forward pass завершён.")

    #  Удаление хуков 
    for h in handles:
        h.remove()
    print(f"Удалено {len(handles)} хуков.")

    if not id_metrics:
        print("Не удалось собрать метрики ID. Проверьте модель и входные данные.")
        return {}

    #  Подготовка данных для графиков 
    # Сортируем ключи: сначала обычные слои (0..num_layers-1), для каждого слоя точки в порядке POINTS,
    # затем финальные точки.
    layer_numbers = sorted(set(k[0] for k in id_metrics.keys() if isinstance(k[0], int) and k[0] < num_layers))
    final_numbers = sorted(set(k[0] for k in id_metrics.keys() if isinstance(k[0], int) and k[0] >= num_layers))

    all_points = []
    for layer in layer_numbers:
        for point in POINTS:
            # input добавляем только для нулевого слоя
            if point == 'input' and layer != 0:
                continue
            key = (layer, point)
            if key in id_metrics:
                all_points.append(key)

    final_desired_order = ['final_norm', 'lm_head']
    for point_name in final_desired_order:
        for layer in final_numbers:
            key = (layer, point_name)
            if key in id_metrics:
                all_points.append(key)
                break

    # Подписи для оси X (короткие и длинные)
    x_labels_all = []
    for (layer, point) in all_points:
        if layer < num_layers:
            if point == 'res2':
                x_labels_all.append(f"{layer}")
            else:
                x_labels_all.append("")
        else:
            if point == 'final_norm':
                x_labels_all.append("")
            elif point == 'lm_head':
                x_labels_all.append("")
            else:
                x_labels_all.append(f"Final {point}")

    x_labels = []
    for (layer, point) in all_points:
        if layer < num_layers:
            x_labels.append(f"{layer} {POINT_ABBR.get(point, point)}")
        else:
            if point == 'final_norm':
                x_labels.append("Final LayerNorm")
            elif point == 'lm_head':
                x_labels.append("LM Head")
            else:
                x_labels.append(f"Final {point}")

    x_pos = np.arange(len(all_points))

    dashed_indices = []
    for i, (layer, point) in enumerate(all_points):
        if (layer == 0 and point == 'input') or point == 'res2' or point == 'lm_head':
            dashed_indices.append(i)

    # Извлекаем значения ID
    id_values = [id_metrics[k] for k in all_points]

    #  График 1: полный (все точки) 
    plt.figure(figsize=(20, 8))
    plt.plot(x_pos, id_values, marker='o', linestyle='-', color='b')
    if dashed_indices:
        dashed_vals = [id_values[i] for i in dashed_indices if i < len(id_values)]
        plt.plot(dashed_indices[:len(dashed_vals)], dashed_vals, 'r--', marker='o', markersize=6, color="blue", linewidth=2, label='layer-wise ID')
    plt.xticks(x_pos, x_labels_all, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Intrinsic Dimension (Levina-Bickel)", fontsize=20)
    plt.xlabel("Layer / stage", fontsize=20)
    plt.title(f"Intrinsic dimension of hidden states – {model.config._name_or_path} (k={k})", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "intrinsic_dimension_per_stage.png", dpi=150)
    plt.show()

    #  График 2: обрезанный до первых 55 точек (как в первой функции) 
    plt.figure(figsize=(20, 8))
    limit1= min(55, len(id_values))
    limit2= min(55, len(id_values))+70
    # limit = min(55, len(id_values))
    plt.plot(x_pos[limit1:limit2], id_values[limit1:limit2], marker='o', linestyle='-', color='b')
    dashed_indices_limited = [i for i in dashed_indices if i < limit2 and i>limit1]
    dashed_vals_limited = [id_values[i] for i in dashed_indices_limited]
    plt.plot(dashed_indices_limited, dashed_vals_limited, 'b--', marker='s', markersize=6, linewidth=2, label='layer-wise ID')
    plt.xticks(x_pos[limit1:limit2], x_labels[limit1:limit2], rotation=90, fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Intrinsic Dimension", fontsize=20)
    plt.title(f"Intrinsic dimension of hidden states – {model.config._name_or_path} (k={k})", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(Path(plot_dir) / "intrinsic_dimension_per_stage_limited.png", dpi=150)
    plt.show()

    #  Вывод сводки в консоль 
    print("\n" + "=" * 60)
    print("Сводка intrinsic dimension по этапам:")
    for layer, point in all_points:
        layer_label = f"{layer}" if layer < num_layers else "Final"
        print(f"{layer_label:<4} {point:10s}: ID = {id_metrics[(layer, point)]:.2f}")

    return id_metrics