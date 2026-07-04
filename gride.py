import numpy as np
from dadapy import Data
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


def intrinsic_dimension_gride(data_points, maxk=100, scale_method='last'):
    """
    Оценка внутренней размерности методом GRIDE.
    
    Параметры:
    ----------
    data_points : np.ndarray, shape (n_samples, n_features)
        Векторы представлений токенов (уже без padding, возможно подвыборка).
    maxk : int
        Максимальное число соседей для анализа масштабов.
    scale_method : str
        - 'last' : взять оценку ID при самом крупном масштабе (наиболее стабильную).
        - 'mean' : усреднить оценки по всем масштабам.
        - 'median' : медиана по масштабам.
        - 'auto' : автоматически выбрать масштаб (см. документацию DADApy).
    
    Возвращает:
    ----------
    id_estimate : float
        Оценка внутренней размерности.
    full_results : dict
        Детали: оценки ID по масштабам и соответствующие расстояния.
    """
    if data_points.shape[0] < maxk + 2:
        return np.nan, None
    
    # Инициализация объекта Data
    data = Data(data_points)
    
    # Вычисляем матрицу расстояний до maxk соседей
    # (опционально: можно сразу передать distances, если они уже есть)
    data.compute_distances(maxk=min(maxk, data_points.shape[0]-1))
    
    # Получаем оценки ID на разных масштабах (для соседей от 2 до maxk)
    ids, id_errors, scales = data.return_id_scaling_gride(range_max=maxk)
    
    if len(ids) == 0:
        return np.nan, None
    
    # Выбираем финальную оценку согласно scale_method
    if scale_method == 'last':
        id_val = ids[-1]
    elif scale_method == 'mean':
        id_val = np.mean(ids)
    elif scale_method == 'median':
        id_val = np.median(ids)
    else:
        id_val = ids[-1]  # по умолчанию
    
    return id_val, {'ids': ids, 'errors': id_errors, 'scales': scales}


def measure_intrinsic_dimension_per_layer_gride(model, batch, maxk=100, scale_method='last', plot=True, include_embedding=True, max_points=10000):
    """
    Оценивает внутреннюю размерность активаций каждого слоя с помощью GRIDE.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        
        layer_dimensions = {}
        layer_details = {}   # для сохранения полных кривых ID(scales)
        layer_indices = []
        dim_values = []
        
        start_idx = 0 if include_embedding else 1
        num_layers = len(hidden_states)
        
        for layer_idx in range(start_idx, num_layers):
            activation = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            
            # Убираем padding токены
            if "attention_mask" in batch:
                mask = batch["attention_mask"].bool()
                mask_flat = mask.reshape(-1)
                act_flat = activation.reshape(-1, activation.size(-1))
                act_filtered = act_flat[mask_flat].cpu().numpy()
            else:
                act_filtered = activation.reshape(-1, activation.size(-1)).cpu().numpy()
            
            # Подвыборка при необходимости
            print("количество анализируемых токенов", act_filtered.shape[0])
            if act_filtered.shape[0] > max_points:
                indices = np.random.choice(act_filtered.shape[0], max_points, replace=False)
                act_filtered = act_filtered[indices]
            
            if act_filtered.shape[0] <= maxk + 2:
                print(f"Слой {layer_idx}: слишком мало точек ({act_filtered.shape[0]}), пропускаем")
                dim_val = np.nan
                details = None
            else:
                dim_val, details = intrinsic_dimension_gride(act_filtered, maxk=maxk, scale_method=scale_method)
            
            layer_dimensions[layer_idx] = dim_val
            layer_details[layer_idx] = details
            layer_indices.append(layer_idx)
            dim_values.append(dim_val)
        
        # Построение графика 
        if plot and len(layer_indices) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            valid_mask = ~np.isnan(dim_values)
            valid_layers = np.array(layer_indices)[valid_mask]
            valid_dims = np.array(dim_values)[valid_mask]
            
            ax.plot(valid_layers, valid_dims, marker='o', markersize=8,
                    linewidth=2, color='steelblue', label='GRIDE ID')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(valid_layers)
            ax.set_xlabel('Номер слоя', fontsize=12)
            ax.set_ylabel('Внутренняя размерность (GRIDE)', fontsize=12)
            title = f'Динамика ID по слоям (GRIDE, maxk={maxk}, scale_method={scale_method})'
            if batch is not None:
                bs, seq = batch["input_ids"].shape[:2]
                title += f', batch={bs}, seq_len={seq}'
            ax.set_title(title, fontsize=14)
            
            # Аннотации точек
            for l, d in zip(valid_layers, valid_dims):
                ax.annotate(f'{d:.1f}', xy=(l, d), xytext=(0, 10),
                            textcoords='offset points', ha='center', fontsize=9)
            
            # Статистика
            if len(valid_dims) > 0:
                mean_d = np.mean(valid_dims)
                std_d = np.std(valid_dims)
                min_d = np.min(valid_dims); min_idx = valid_layers[np.argmin(valid_dims)]
                max_d = np.max(valid_dims); max_idx = valid_layers[np.argmax(valid_dims)]
                stats_text = f'Mean: {mean_d:.1f}±{std_d:.1f}\nMin: {min_d:.1f} (layer {min_idx})\nMax: {max_d:.1f} (layer {max_idx})'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                ax.scatter(min_idx, min_d, s=150, color='red', marker='*', label='Min', zorder=5)
                ax.scatter(max_idx, max_d, s=150, color='green', marker='^', label='Max', zorder=5)
            
            ax.legend()
            plt.tight_layout()
            plt.show()
            
            # Вывод в консоль
            print(f"\n GRIDE ID по слоям (maxk={maxk}, scale_method={scale_method}):")
            print("dim_values", dim_values)
            for l, d in zip(layer_indices, dim_values):
                print(f"   Layer {l:2d}: {d if not np.isnan(d) else 'N/A':>6.1f}")
        
        return layer_dimensions, layer_details