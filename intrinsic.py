
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


def intrinsic_dimension_levina_bickel_vectorized(data, k=10):
    n = len(data)
    if n <= k + 1:
        k = min(k, n - 2)
        if k < 2:
            return 0
    
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Находим k+1 соседей (включая саму точку)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Расстояния до соседей (исключая саму точку, которая на позиции 0)
    neighbor_distances = distances[:, 1:]  # shape (n_samples, k)
    
    # Расстояние до k-го соседа (самого дальнего из k соседей)
    d_k = neighbor_distances[:, -1:]  # последний столбец, shape (n_samples, 1)
    
    # Расстояния до соседей 1..(k-1)
    d_i = neighbor_distances[:, :-1]  # все кроме последнего, shape (n_samples, k-1)
    
    eps = 1e-10
    ratios = d_k / (d_i + eps)
    
    # Логарифм отношений
    log_ratios = np.log(ratios + eps)
    
    # Сумма логарифмов по i=1..(k-1)
    sum_log = np.sum(log_ratios, axis=1)
    
    # Локальная оценка ID
    local_ids = (k - 1) / (sum_log + eps)
    
    # Фильтрация
    valid_local_ids = local_ids[(local_ids > 0) & (local_ids < 1e6) & (~np.isnan(local_ids))]
    
    if len(valid_local_ids) == 0:
        return 0
    
    # Гармоническое среднее
    global_id = len(valid_local_ids) / np.sum(1.0 / (valid_local_ids + eps))
    
    return global_id



def measure_intrinsic_dimension_per_layer(model, batch, k=10, plot=True, 
                                          include_embedding=True, max_points=10000):
    """
    Measure intrinsic dimension for activations in each layer of Qwen2.5
    
    Args:
        model: Qwen2.5 model
        batch: Input batch with input_ids and attention_mask
        k: Number of neighbors for Levina-Bickel estimator
        plot: Whether to plot results
        include_embedding: Whether to include embedding layer (layer 0)
        max_points: Maximum number of points to sample for ID estimation
    """
    model.eval()
    with torch.no_grad():
        # Forward pass with hidden states
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        
        # Get all hidden states
        # hidden_states is a tuple with (embedding_output, layer1_output, ..., layerN_output)
        hidden_states = outputs.hidden_states
        
        layer_dimensions = {}
        layer_indices = []
        dim_values = []
        
        # Determine which layers to process
        start_idx = 0 if include_embedding else 1
        num_layers = len(hidden_states)
        
        for layer_idx in range(start_idx, num_layers):
            activation = hidden_states[layer_idx]
            
            # Reshape activations: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
            # Remove padding tokens if needed
            if "attention_mask" in batch:
                # Get mask and flatten
                mask = batch["attention_mask"].bool()
                # Apply mask to select only non-padding tokens
                batch_size, seq_len = activation.shape[:2]
                
                # Reshape mask and activation
                mask_flat = mask.reshape(-1)
                activation_flat = activation.reshape(-1, activation.size(-1))
                
                # Select only non-padding tokens
                activation_filtered = activation_flat[mask_flat]
            else:
                # No mask available, use all tokens
                activation_filtered = activation.reshape(-1, activation.size(-1))
            
            # Subsample if too many points
            if activation_filtered.size(0) > max_points:
                indices = torch.randperm(activation_filtered.size(0))
                print("Количество анализируемых токенов до обрезки", len(indices))
                indices = indices[:max_points]
                activation_filtered = activation_filtered[indices]
            
            # Skip if too few points
            if activation_filtered.size(0) <= k + 1:
                print(f"Warning: Layer {layer_idx} has too few points ({activation_filtered.size(0)})")
                dim = 0
            else:
                # Calculate intrinsic dimension
                dim = intrinsic_dimension_levina_bickel_vectorized(activation_filtered, k=k)
            
            layer_dimensions[layer_idx] = dim
            
            # Store for plotting
            layer_indices.append(layer_idx)
            dim_values.append(dim)
        
        # Plot layer-wise dimension dynamics
        if plot and len(layer_indices) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot line with markers
            ax.plot(layer_indices, dim_values, marker='o', markersize=8, 
                   linewidth=2, color='steelblue', label='ID по слоям')
            
            # Add horizontal grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Customize axis labels
            ax.set_xlabel('Номер слоя', fontsize=12)
            ax.set_ylabel('Внутренняя размерность (ID)', fontsize=12)
            
            # Title with hyperparameter info
            title = f'Динамика внутренней размерности (k={k})'
            if batch is not None and "input_ids" in batch:
                batch_size = batch["input_ids"].shape[0]
                seq_len = batch["input_ids"].shape[1]
                title += f', batch={batch_size}, seq_len={seq_len}'
            ax.set_title(title, fontsize=14, pad=15)
            
            # Set x-ticks as integers
            ax.set_xticks(layer_indices)
            
            # Add value annotations on points
            for i, (layer, dim) in enumerate(zip(layer_indices, dim_values)):
                ax.annotate(f'{dim:.1f}', 
                           xy=(layer, dim), 
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9,
                           color='darkred')
            
            # Calculate and display statistics
            if len(dim_values) > 0:
                min_dim = min(dim_values)
                max_dim = max(dim_values)
                mean_dim = np.mean(dim_values)
                std_dim = np.std(dim_values)
                
                # Add text box with statistics
                stats_text = f'Min: {min_dim:.1f}\nMax: {max_dim:.1f}\nMean: {mean_dim:.1f}\nStd: {std_dim:.1f}'
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=10)
                
                # Highlight minimum dimension
                min_idx = dim_values.index(min_dim)
                ax.scatter(layer_indices[min_idx], min_dim, 
                          s=150, color='red', marker='*', 
                          label=f'Минимум (слой {layer_indices[min_idx]})', 
                          zorder=5)
            else:
                print("Warning: No valid dimension values to plot")
            
            ax.legend(loc='best')
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            if len(dim_values) > 0:
                print(f"\nСтатистика внутренней размерности (k={k}):")
                print(f"   Среднее по слоям: {mean_dim:.2f} ± {std_dim:.2f}")
                print(f"   Диапазон: {min_dim:.1f} - {max_dim:.1f}")
                print(f"   Минимум на слое {layer_indices[min_idx]}: {min_dim:.1f}")
                
                # Optional: Print all values
                print("\nЗначения по слоям:")
                print("dim_values", dim_values)
                for layer, dim in zip(layer_indices, dim_values):
                    print(f"   Слой {layer:2d}: {dim:6.1f}")
    
    return layer_dimensions


# Альтернативная версия для отладки - с сохранением активаций
def measure_intrinsic_dimension_per_layer_debug(model, batch, k=10, max_points=10000):
    """
    Debug version that returns more information
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        
        results = {
            'layer_dimensions': {},
            'activations_info': {},
            'sample_sizes': {}
        }
        
        for layer_idx in range(len(hidden_states)):
            activation = hidden_states[layer_idx]
            
            # Get activation statistics
            act_mean = activation.mean().item()
            act_std = activation.std().item()
            act_min = activation.min().item()
            act_max = activation.max().item()
            
            results['activations_info'][layer_idx] = {
                'mean': act_mean,
                'std': act_std,
                'min': act_min,
                'max': act_max,
                'shape': activation.shape
            }
            
            # Process activation for ID calculation
            if "attention_mask" in batch:
                mask = batch["attention_mask"].bool()
                mask_flat = mask.reshape(-1)
                activation_flat = activation.reshape(-1, activation.size(-1))
                activation_filtered = activation_flat[mask_flat]
            else:
                activation_filtered = activation.reshape(-1, activation.size(-1))
            
            results['sample_sizes'][layer_idx] = activation_filtered.size(0)
            
            # Subsample
            if activation_filtered.size(0) > max_points:
                indices = torch.randperm(activation_filtered.size(0))[:max_points]
                activation_filtered = activation_filtered[indices]
            
            # Calculate ID
            if activation_filtered.size(0) > k + 1:
                dim = intrinsic_dimension_levina_bickel_vectorized(activation_filtered, k=k)
                results['layer_dimensions'][layer_idx] = dim
            else:
                results['layer_dimensions'][layer_idx] = 0
        
        return results



def compute_intrinsic_dim_per_sample(model, config, tokenized_texts, device, 
                                    sample_size=100, k=10):
    """
    Вычисляет внутреннюю размерность для каждого отдельного текста в каждом слое
    
    Parameters:
    -----------
    model : GPT2WithHooks
        Модель с хуками
    tokenized_texts : list of dict
        Список уже токенизированных текстов в виде словарей с ключами:
        - 'input_ids': torch.Tensor
        - 'attention_mask': torch.Tensor
    device : torch.device
        Устройство для вычислений
    sample_size : int
        Количество текстов для анализа (None для всех)
    k : int
        Количество ближайших соседей для MLE оценки
        
    Returns:
    --------
    results : dict
        Словарь с результатами
    """
    if sample_size is None or sample_size > len(tokenized_texts):
        sample_size = len(tokenized_texts)
    
    # Выбираем случайные тексты для анализа
    indices = torch.randperm(len(tokenized_texts))[:sample_size].tolist()
    selected_items = [tokenized_texts[i] for i in indices]
    
    print(f"Вычисление внутренней размерности для {len(selected_items)} токенизированных текстов...")
    
    # Инициализируем структуры для хранения результатов
    n_layers = model.config.n_layer if config.model_name=="GPT2" else model.config.num_hidden_layers
    layer_dimensions = {layer_idx: [] for layer_idx in range(n_layers)}
    
    model.eval()
    
    for item_idx, item in enumerate(tqdm(selected_items, desc="Обработка текстов")):
        try:
            # Проверяем, что элемент имеет правильную структуру
            if not isinstance(item, dict) or 'input_ids' not in item or 'attention_mask' not in item:
                print(f"Элемент {item_idx} не является словарем с input_ids и attention_mask")
                continue
            
            # Подготавливаем входные данные
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            
            # Добавляем размерность батча, если нужно
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            # Прямой проход для получения активаций
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Получаем скрытые состояния для всех слоев
                hidden_states = outputs.hidden_states
            
            # Для каждого слоя вычисляем внутреннюю размерность
            for layer_idx in range(n_layers):
                # Берем активации слоя
                activation = hidden_states[layer_idx + 1]  # +1 чтобы пропустить эмбеддинги
                
                # Преобразуем в 2D: (batch_size * seq_len, hidden_dim)
                # Используем только non-padding токены
                batch_size, seq_len, hidden_dim = activation.shape
                
                # Используем attention_mask для фильтрации padding токенов
                mask = attention_mask.unsqueeze(-1).expand(-1, -1, hidden_dim)
                activation_masked = activation * mask
                
                # Преобразуем в 2D и удаляем нулевые векторы (padding)
                act_reshaped = activation_masked.reshape(-1, hidden_dim)
                act_reshaped = act_reshaped[torch.norm(act_reshaped, dim=1) > 0]
                
                # Ограничиваем количество точек для вычисления
                max_points = 1000
                if act_reshaped.size(0) > max_points:
                    indices = torch.randperm(act_reshaped.size(0))[:max_points]
                    act_reshaped = act_reshaped[indices]
                elif act_reshaped.size(0) < k + 2:
                    continue
                
                # Вычисляем внутреннюю размерность
                dim = intrinsic_dimension_levina_bickel_vectorized(act_reshaped, k=k)
                
                if dim > 0 and not np.isnan(dim):
                    layer_dimensions[layer_idx].append(dim)
        
        except Exception as e:
            print(f"Ошибка при обработке элемента {item_idx}: {e}")
            continue
    
    # Вычисляем статистику по слоям
    layer_stats = {}
    for layer_idx, dims in layer_dimensions.items():
        if dims:  # Если есть измерения
            dims_array = np.array(dims)
            layer_stats[layer_idx] = {
                'mean': float(np.mean(dims_array)),
                'median': float(np.median(dims_array)),
                'std': float(np.std(dims_array)),
                'min': float(np.min(dims_array)),
                'max': float(np.max(dims_array)),
                'count': len(dims_array)
            }
    
    results = {
        'layer_dimensions': layer_dimensions,
        'text_indices': indices,
        'layer_stats': layer_stats,
        'k': k,
        'sample_size': sample_size
    }
    
    return results

def plot_layerwise_intrinsic_dim_distribution(results, model, batch=None, 
                                             layers_to_plot=None, bins=30,
                                             include_trend_plot=True):
    """
    Визуализация распределения внутренней размерности по слоям
    и графика динамики среднего значения
    
    Parameters:
    -----------
    results : dict
        Результаты из compute_intrinsic_dim_per_sample
    model : object
        Модель трансформера
    batch : dict, optional
        Батч данных (для информации о размере)
    layers_to_plot : list, optional
        Список слоев для построения распределений
    bins : int, optional
        Количество бинов для гистограмм
    include_trend_plot : bool, optional
        Включать ли график динамики среднего значения
    """
    # Проверяем наличие данных
    if 'layer_stats' not in results or not results['layer_stats']:
        print("Нет данных для построения графика")
        return
    
    # Определяем слои для построения
    if layers_to_plot is None:
        layers_to_plot = sorted(results['layer_stats'].keys())
    
    # Если нужно построить график динамики среднего
    if include_trend_plot:
        # Создаем фигуру с двумя subplots
        fig = plt.figure(figsize=(18, 10))
        
        # График 1: Динамика среднего значения
        ax1 = plt.subplot(2, 1, 1)
        
        # Собираем данные для графика динамики
        sorted_layers = sorted(results['layer_stats'].keys())
        layer_indices = []
        mean_values = []
        std_values = []
        
        for layer_idx in sorted_layers:
            stats = results['layer_stats'][layer_idx]
            layer_indices.append(layer_idx)
            mean_values.append(stats['mean'])
            std_values.append(stats['std'])
        
        # Строим график динамики
        ax1.plot(layer_indices, mean_values, marker='o', markersize=8,
                linewidth=2, color='steelblue', label='Средняя ID')
        
        # Добавляем область стандартного отклонения
        ax1.fill_between(layer_indices, 
                        np.array(mean_values) - np.array(std_values),
                        np.array(mean_values) + np.array(std_values),
                        alpha=0.2, color='steelblue', label='±1 std')
        
        # Настройки графика динамики
        ax1.set_xlabel('Номер слоя трансформера', fontsize=12)
        ax1.set_ylabel('Внутренняя размерность', fontsize=12)
        
        title = f'Динамика внутренней размерности по слоям (k={results["k"]})'
        if 'sample_size' in results:
            title += f', n={results["sample_size"]} текстов'
        ax1.set_title(title, fontsize=14, pad=15)
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(layer_indices)
        ax1.legend(loc='best')
        
        # Аннотации значений
        for layer, mean_val in zip(layer_indices, mean_values):
            ax1.annotate(f'{mean_val:.1f}', 
                        xy=(layer, mean_val), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=9,
                        color='darkred')
        
        # Выделяем минимум и максимум
        min_idx = np.argmin(mean_values)
        max_idx = np.argmax(mean_values)
        
        ax1.scatter(layer_indices[min_idx], mean_values[min_idx],
                   s=150, color='red', marker='*', 
                   label=f'Min (слой {layer_indices[min_idx]})', zorder=5)
        
        ax1.scatter(layer_indices[max_idx], mean_values[max_idx],
                   s=150, color='green', marker='^', 
                   label=f'Max (слой {layer_indices[max_idx]})', zorder=5)
        
        # Добавляем статистику
        stats_text = (f'Среднее: {np.mean(mean_values):.1f}\n'
                     f'Std: {np.std(mean_values):.1f}\n'
                     f'Min: {mean_values[min_idx]:.1f} (слой {layer_indices[min_idx]})\n'
                     f'Max: {mean_values[max_idx]:.1f} (слой {layer_indices[max_idx]})')
        
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        # График 2: Распределения по слоям
        ax2 = plt.subplot(2, 1, 2)
        
        # Ограничиваем количество слоев для гистограмм, если их слишком много
        max_layers_for_hist = 8
        if len(layers_to_plot) > max_layers_for_hist:
            # Выбираем слои равномерно
            step = len(layers_to_plot) // max_layers_for_hist
            layers_to_plot = layers_to_plot[::step][:max_layers_for_hist]
            print(f"Показываем распределения для {len(layers_to_plot)} слоев (из {len(results['layer_stats'])})")
        
        # Собираем данные для гистограмм
        all_dims = []
        layer_labels = []
        
        for layer_idx in layers_to_plot:
            if layer_idx in results['layer_dimensions']:
                dims = results['layer_dimensions'][layer_idx]
                if dims:  # Если есть данные
                    all_dims.append(dims)
                    layer_labels.append(f'Слой {layer_idx}')
        
        # Строим гистограммы
        if all_dims:
            # Используем boxplot или violin plot для наглядности
            ax2.boxplot(all_dims, labels=layer_labels)
            ax2.set_xlabel('Слой трансформера', fontsize=12)
            ax2.set_ylabel('Внутренняя размерность', fontsize=12)
            ax2.set_title('Распределение ID по слоям', fontsize=14)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Добавляем точки поверх boxplot для наглядности
            for i, dims in enumerate(all_dims):
                x = np.random.normal(i + 1, 0.04, size=len(dims))
                ax2.scatter(x, dims, alpha=0.6, s=30, color='blue')
        else:
            ax2.text(0.5, 0.5, 'Нет данных для построения распределений',
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Только распределения по слоям (старый вариант)
        fig, axes = plt.subplots(len(layers_to_plot), 1, 
                                figsize=(15, 4 * len(layers_to_plot)))
        if len(layers_to_plot) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers_to_plot):
            ax = axes[i]
            
            if layer_idx in results['layer_dimensions'] and results['layer_dimensions'][layer_idx]:
                dims = results['layer_dimensions'][layer_idx]
                stats = results['layer_stats'][layer_idx]
                
                # Строим гистограмму
                ax.hist(dims, bins=bins, alpha=0.7, label=f'Слой {layer_idx}', color='purple')
                ax.set_xlabel('Внутренняя размерность')
                ax.set_ylabel('Количество текстов')
                ax.set_title(f'Слой {layer_idx}: среднее = {stats["mean"]:.4f}, std = {stats["std"]:.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Слой {layer_idx}\nНет данных', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Слой {layer_idx}')
        
        plt.suptitle(f'Распределение внутренней размерности по слоям (k={results["k"]})', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    
    # Выводим статистику в консоль
    print(f"\n Сводная статистика по всем слоям (k={results['k']}):")
    print(f"{'Слой':<6} {'Среднее':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'N':<6}")
    print("-" * 60)
    
    for layer_idx in sorted(results['layer_stats'].keys()):
        stats = results['layer_stats'][layer_idx]
        print(f"{layer_idx:<6} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
              f"{stats['min']:<10.2f} {stats['max']:<10.2f} {stats['count']:<6}")

# Упрощенная версия только для графика динамики
def plot_intrinsic_dimension_trend(results, figsize=(14, 6)):
    """
    Только график динамики среднего значения по слоям
    """
    if 'layer_stats' not in results:
        print("Нет данных для построения графика")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Собираем данные
    sorted_layers = sorted(results['layer_stats'].keys())
    layer_indices = []
    mean_values = []
    std_values = []
    
    for layer_idx in sorted_layers:
        stats = results['layer_stats'][layer_idx]
        layer_indices.append(layer_idx)
        mean_values.append(stats['mean'])
        std_values.append(stats['std'])
    
    # График
    ax.plot(layer_indices, mean_values, marker='o', markersize=8,
            linewidth=2, color='steelblue', label='Средняя ID')
    
    # Область std
    ax.fill_between(layer_indices,
                   np.array(mean_values) - np.array(std_values),
                   np.array(mean_values) + np.array(std_values),
                   alpha=0.2, color='steelblue', label='±1 std')
    
    # Настройки
    ax.set_xlabel('Номер слоя трансформера', fontsize=12)
    ax.set_ylabel('Внутренняя размерность', fontsize=12)
    
    title = f'Динамика внутренней размерности по слоям (k={results["k"]})'
    if 'sample_size' in results:
        title += f', n={results["sample_size"]} текстов'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(layer_indices)
    
    # Аннотации
    for layer, mean_val in zip(layer_indices, mean_values):
        ax.annotate(f'{mean_val:.1f}',
                   xy=(layer, mean_val),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)
    
    # Статистика
    mean_all = np.mean(mean_values)
    std_all = np.std(mean_values)
    min_val = np.min(mean_values)
    max_val = np.max(mean_values)
    
    min_layer = layer_indices[np.argmin(mean_values)]
    max_layer = layer_indices[np.argmax(mean_values)]
    
    stats_text = (f'Среднее: {mean_all:.1f} ± {std_all:.1f}\n'
                  f'Диапазон: {min_val:.1f} - {max_val:.1f}\n'
                  f'Слой min: {min_layer} ({min_val:.1f})\n'
                  f'Слой max: {max_layer} ({max_val:.1f})')
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return fig, ax