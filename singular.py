import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd as scipy_svd
import torch
import json
import matplotlib.pyplot as plt
from save import setup_plot_saving, save_checkpoint_simple, manage_last_checkpoints

def compute_svd_anisotropy(activations, k=None, method='full', variance_threshold=0.95):
    """
    Вычисляет анизотропию и эффективную размерность.
    
    Returns:
        anisotropy: σ₁² / Σ σ_i²
        singular_values: все сингулярные числа
        eff_dim_threshold: минимальное число компонент для накопленной variance_threshold дисперсии
        eff_dim_pr: участие (participation ratio) (Σλ_i)² / Σλ_i²
    """
    X = activations - activations.mean(dim=0, keepdim=True)
    
    # 2. SVDvals — только сингулярные числа, без векторов
    S = torch.linalg.svdvals(X.float())  # возвращает (min(n,m),)
    
    if k is not None:
        S = S[:min(k, len(S))]
    
    sigma_squared = S ** 2
    total_variance = sigma_squared.sum()
    
    if total_variance == 0:
        return 0.0, S, 0, 0.0
    
    anisotropy = sigma_squared[0] / total_variance
    
    # Эффективная размерность по порогу
    cumsum = sigma_squared.cumsum(dim=0)
    eff_dim_threshold = (cumsum >= variance_threshold * total_variance).nonzero(as_tuple=True)[0][0].item() + 1
    
    # Participation ratio
    sum_lambda = total_variance
    sum_lambda_sq = (sigma_squared ** 2).sum()
    eff_dim_pr = (sum_lambda ** 2) / sum_lambda_sq if sum_lambda_sq > 0 else 0.0
    
    # Энтропийная размерность
    p = sigma_squared / total_variance
    # Убираем нулевые p, чтобы log(0) не возникало
    p_nonzero = p[p > 0]
    entropy = -torch.sum(p_nonzero * torch.log(p_nonzero))
    eff_dim_entropy = torch.exp(entropy).item()

    X = activations  # или activations - mean, если хотите
    S = torch.linalg.svdvals(X.float())   # (min(N,d),)
    
    # Нормировка по сумме сингулярных чисел (не квадратов!)
    total_sum = S.sum()
    if total_sum == 0:
        return 0.0
    p = S / total_sum     
    
    p_nonzero = p[p > 0]
    entropy = -torch.sum(p_nonzero * torch.log(p_nonzero))
    rankme = torch.exp(entropy).item()
    
    return anisotropy.item(), S.cpu(), eff_dim_threshold, eff_dim_pr, eff_dim_entropy, rankme
    

def analyze_svd_anisotropy_by_layer(model, batch, config, layers_to_plot=None):
    """
    Анализирует анизотропию через SVD по слоям модели.
    
    Args:
        model: Модель трансформера
        batch: Батч данных
        config: Конфиг обучения
        layers_to_plot: Список слоев для анализа
    
    Returns:
        layer_anisotropy: Словарь с анизотропией по слоям
        all_singular_values: Сингулярные значения по слоям
    """
    device = next(model.parameters()).device
    
    if batch['input_ids'].device != device:
        batch = {k: v.to(device) for k, v in batch.items()}
    
    model.eval()
    
    if layers_to_plot is None:
        if config.model_name == "GPT2":
            n_layers = model.config.n_layer
        else:
            n_layers = model.config.num_hidden_layers
        layers_to_plot = list(range(n_layers + 1))
    
    if config.model_name == "GPT2":
        pad_token_id = 50256
    elif config.model_name=="Qwen":
        pad_token_id = 151643
    else:
        pad_token_id = 0
    
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ SVD АНИЗОТРОПИИ ДЛЯ МОДЕЛИ {config.model_name}")
    print(f"{'='*60}")
    
    layer_anisotropy = {}
    all_singular_values = {}
    accumulated_anisotropy = []
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        
        for layer_idx in layers_to_plot:
            if layer_idx < len(hidden_states):
                activation = hidden_states[layer_idx]          # уже лежит на GPU
                batch_size, seq_len, hidden_dim = activation.shape
                activation_reshaped = activation.view(-1, hidden_dim)
                
                # Маска для исключения pad токенов (input_ids уже на GPU)
                input_ids_flat = batch["input_ids"].view(-1)
                valid_mask = (input_ids_flat != pad_token_id)   # pad_token_id укажите корректно
                
                embeddings = activation_reshaped[valid_mask]
                
                # Подвыборка (если нужно) – прямо на GPU
                if embeddings.shape[0] > config.num_samples:
                    idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:config.num_samples]
                    embeddings = embeddings[idx]
                
                # Вычисляем анизотропию БЕЗ .cpu() до вызова
                anisotropy, singular_values, eff_thr, eff_pr, eff_ent, rankme = compute_svd_anisotropy(
                    embeddings, k=None, variance_threshold=0.95
                )
                
                layer_anisotropy[layer_idx] = {
                    'anisotropy': float(anisotropy),
                    'top_singular_values': singular_values[:5].tolist(),
                    'energy_ratio': float(anisotropy),
                    'eff_dim_threshold': int(eff_thr),
                    'eff_dim_pr': float(eff_pr),
                    'eff_dim_entropy': float(eff_ent),
                    'rankme': float(rankme)
                }
                all_singular_values[layer_idx] = singular_values
                accumulated_anisotropy.append(anisotropy)

                non_zero_count = torch.sum(singular_values > 0.0001).item()
                
                print(f"Слой {layer_idx:3d}: Анизотропия = {anisotropy:.6f}, "
                    f"Ранг = {non_zero_count:d}, "
                    f"EffDim95 = {eff_thr:3d}, EffDimPR = {eff_pr:6.2f}, EffDimEnt = {eff_ent:6.2f}," f"RankMe = {rankme}")
    
    # Выводим статистику
    avg_anisotropy = np.mean(accumulated_anisotropy)
    std_anisotropy = np.std(accumulated_anisotropy)
    print(f"\nСредняя анизотропия по слоям: {avg_anisotropy:.6f} ± {std_anisotropy:.6f}")
    print(f"{'='*60}")
    
    # Визуализация результатов
    plot_svd_anisotropy_results(layer_anisotropy, all_singular_values, config)
    
    return layer_anisotropy, all_singular_values

def plot_svd_anisotropy_results(layer_anisotropy, all_singular_values, config, 
                                top_n=20, plot_per_layer=False, save_to_file=False):
    """
    Визуализирует результаты анализа SVD анизотропии.
    
    Args:
        layer_anisotropy: Словарь с анизотропией по слоям
        all_singular_values: Сингулярные значения по слоям
        config: Конфиг обучения
        top_n: Количество топовых сингулярных чисел для отображения
        plot_per_layer: Флаг для построения графиков для каждого слоя
        save_to_file: Если True - сохраняет в файл, если False - выводит на экран
    """
    from datetime import datetime
    import os
    
    # Определяем директорию для сохранения (только если save_to_file=True)
    if save_to_file:
        plot_dir = setup_plot_saving(config, suffix="svd_results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    layers = sorted(layer_anisotropy.keys())
    anisotropies = [layer_anisotropy[l]['anisotropy'] for l in layers]
    energy_ratios = [layer_anisotropy[l]['energy_ratio'] for l in layers]
    eff_thr = [layer_anisotropy[l]['eff_dim_threshold'] for l in layers]
    eff_pr = [layer_anisotropy[l]['eff_dim_pr'] for l in layers]
    eff_ent = [layer_anisotropy[l]['eff_dim_entropy'] for l in layers]
    rankme = [layer_anisotropy[l]['rankme'] for l in layers]
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))   # 3 строки, 2 колонки
    
    # 1. Анизотропия (ряд 0, колонка 0)
    ax1 = axes[0, 0]
    ax1.set_ylim(0, 1)
    ax1.plot(layers, anisotropies, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Singular anisotropy', fontsize=12)
    ax1.set_title('Singular anisotropy (σ₁²/Σσᵢ²)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Энергия первого компонента (ряд 0, колонка 1)
    ax2 = axes[0, 1]
    ax2.plot(layers, rankme, 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('RankMe', fontsize=12)
    ax2.set_title('Effective dimensions (RankMe)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Пороговая эффективная размерность (95% дисперсии) – ряд 1, колонка 0
    ax3 = axes[1, 0]
    ax3.plot(layers, eff_thr, 'm-o', linewidth=2, markersize=6, color='purple')
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Effective dim (95% var)', fontsize=12)
    ax3.set_title('Threshold-based (95% variance)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Participation ratio – ряд 1, колонка 1
    ax4 = axes[1, 1]
    ax4.plot(layers, eff_pr, 'c-s', linewidth=2, markersize=6)
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Participation ratio', fontsize=12)
    ax4.set_title('Participation ratio (PR)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Энтропийная эффективная размерность – ряд 2, колонка 0
    ax5 = axes[2, 0]
    ax5.plot(layers, eff_ent, 'orange', marker='^', linewidth=2, markersize=6)
    ax5.set_xlabel('Layer', fontsize=12)
    ax5.set_ylabel('Entropy-based effective dim', fontsize=12)
    ax5.set_title('Shannon entropy dimension (exp(H))', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. Спектр сингулярных чисел (нормализованный) для выбранных слоёв – ряд 2, колонка 1
    ax6 = axes[2, 1]
    selected_layers = [layers[0], layers[len(layers)//2], layers[-1]]
    colors = ['red', 'green', 'blue']
    for i, layer_idx in enumerate(selected_layers):
        if layer_idx in all_singular_values:
            sv = all_singular_values[layer_idx]
            normalized = sv / sv[0] if sv[0] > 0 else sv
            ax6.plot(normalized[:50], color=colors[i], label=f'Layer {layer_idx}')
    ax6.set_xlabel('Singular value index', fontsize=12)
    ax6.set_ylabel('Normalized value', fontsize=12)
    ax6.set_title('Singular value spectrum (first 50)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.suptitle(f'SVD anisotropy analysis – {config.model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_to_file:
        # Сохраняем общий график
        general_filename = f"svd_anisotropy_general_{config.model_name}_{timestamp}.png"
        general_path = os.path.join(plot_dir, general_filename)
        plt.savefig(general_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nОбщие графики SVD анализа сохранены в: {general_path}")
    else:
        plt.show()
        plt.close()
    
    # 2. Графики топ-N сингулярных чисел для каждого слоя (если нужно)
    if plot_per_layer:
        # Создаем поддиректорию для графиков по слоям (только если save_to_file=True)
        if save_to_file:
            layer_plots_dir = os.path.join(plot_dir, f"layerwise_svd_{timestamp}")
            os.makedirs(layer_plots_dir, exist_ok=True)
        
        print(f"\nСоздание графиков топ-{top_n} сингулярных чисел для каждого слоя...")
        
        # Определяем, для скольких слоев строить графики
        if len(layers) > 30:  # Если слоев много, строим только для выборки
            # Берем каждый 3-й слой + первый и последний
            step = max(1, len(layers) // 10)
            layers_to_plot = list(range(0, len(layers), step))[:10]
            layers_to_plot = sorted(set(layers_to_plot + [0, layers[-1]]))
            layers_to_plot = [layers[i] for i in layers_to_plot]
        else:
            layers_to_plot = layers
        
        for layer_idx in layers_to_plot:
            if layer_idx in all_singular_values:
                singular_values = all_singular_values[layer_idx]
                
                # Берем топ-N сингулярных чисел
                n_values = min(top_n, len(singular_values))
                top_singular_values = singular_values[:n_values]
                indices = np.arange(1, n_values + 1)  # Нумерация с 1
                
                # Создаем график
                plt.figure(figsize=(10, 6))
                
                # График: Столбчатая диаграмма топ-N сингулярных чисел
                bars = plt.bar(indices, top_singular_values, color='skyblue', edgecolor='black')
                plt.xlabel('Номер сингулярного числа (по убыванию)', fontsize=11)
                plt.ylabel('Значение сингулярного числа', fontsize=11)
                plt.title(f'Топ-{n_values} сингулярных чисел (слой {layer_idx})', fontsize=13, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Простая разметка оси X
                if n_values <= 20:
                    plt.xticks(indices)
                else:
                    plt.xticks(indices[::max(1, n_values//10)])
                
                plt.tight_layout()
                
                if save_to_file:
                    # Сохраняем график для этого слоя
                    layer_filename = f"layer_{layer_idx:03d}_top{top_n}_svd.png"
                    layer_path = os.path.join(layer_plots_dir, layer_filename)
                    plt.savefig(layer_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  График для слоя {layer_idx} сохранен в {layer_filename}")
                else:
                    plt.show()
                    plt.close()
        
        if save_to_file:
            print(f"Графики по слоям сохранены в директории: {layer_plots_dir}")
            
            # 3. Создаем сводный график для нескольких выбранных слоев
            print("\nСоздание сводного графика для выбранных слоев...")
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            # Выбираем 6 слоев для сводного графика
            if len(layers) >= 6:
                selected_for_summary = [
                    0, 
                    len(layers)//5, 
                    len(layers)//2, 
                    3*len(layers)//4,
                    len(layers)-1
                ]
                # Добавляем еще один, если есть место
                if len(layers) > 5:
                    selected_for_summary.append(len(layers)//3)
            else:
                selected_for_summary = layers[:min(6, len(layers))]
            
            for i, layer_idx in enumerate(selected_for_summary[:6]):
                if i < len(axes) and layer_idx in all_singular_values:
                    singular_values = all_singular_values[layer_idx]
                    n_values = min(top_n, len(singular_values))
                    top_sv = singular_values[:n_values]
                    indices = np.arange(1, n_values + 1)
                    
                    ax = axes[i]
                    bars = ax.bar(indices, top_sv, color=plt.cm.viridis(i/6), alpha=0.7)
                    ax.set_xlabel('Номер σ', fontsize=10)
                    ax.set_ylabel('Значение', fontsize=10)
                    ax.set_title(f'Слой {layer_idx}\nАнизотропия: {layer_anisotropy[layer_idx]["anisotropy"]:.4f}', 
                               fontsize=11)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_xticks(indices[::max(1, n_values//5)])
            
            plt.suptitle(f'Топ-{top_n} сингулярных чисел для выбранных слоев ({config.model_name})', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            summary_filename = f"svd_top{top_n}_summary_{config.model_name}_{timestamp}.png"
            summary_path = os.path.join(plot_dir, summary_filename)
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Сводный график сохранен в: {summary_path}")
    
    # Сохраняем данные в JSON (только если save_to_file=True)
    if save_to_file:
        data_filename = f"svd_anisotropy_data_{config.model_name}_{timestamp}.json"
        data_filepath = os.path.join(plot_dir, data_filename)
        
        # Подготовка данных для сохранения
        save_data = {
            'model_name': config.model_name,
            'timestamp': timestamp,
            'layer_anisotropy': layer_anisotropy,
            'summary': {
                'mean_anisotropy': float(np.mean(anisotropies)),
                'std_anisotropy': float(np.std(anisotropies)),
                'max_anisotropy': float(np.max(anisotropies)),
                'min_anisotropy': float(np.min(anisotropies))
            }
        }
        
        # Сохраняем топ-20 сингулярных значений для каждого слоя
        svd_data = {}
        for layer_idx, sv in all_singular_values.items():
            svd_data[str(layer_idx)] = sv[:20].tolist()  # Только первые 20 значений
        
        save_data['singular_values_top20'] = svd_data
        
        with open(data_filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Данные SVD анализа сохранены в: {data_filepath}")
    else:
        # Выводим краткую сводку на экран
        print(f"\n{'='*60}")
        print(f"СВОДКА SVD АНАЛИЗА АНИЗОТРОПИИ")
        print(f"{'='*60}")
        print(f"Модель: {config.model_name}")
        print(f"Средняя анизотропия по слоям: {np.mean(anisotropies):.6f}")
        print(f"Стандартное отклонение: {np.std(anisotropies):.6f}")
        print(f"Максимальная анизотропия (слой {np.argmax(anisotropies)}): {np.max(anisotropies):.6f}")
        print(f"Минимальная анизотропия (слой {np.argmin(anisotropies)}): {np.min(anisotropies):.6f}")
        print(f"{'='*60}")
        
        # Выводим топ-5 слоев по анизотропии
        print("\nТоп-5 слоев по анизотропии:")
        sorted_layers = sorted(layer_anisotropy.items(), key=lambda x: x[1]['anisotropy'], reverse=True)
        for layer_idx, data in sorted_layers[:5]:
            print(f"  Слой {layer_idx:3d}: анизотропия = {data['anisotropy']:.6f}, "
                  f"энергия σ₁ = {data['energy_ratio']*100:.2f}%")