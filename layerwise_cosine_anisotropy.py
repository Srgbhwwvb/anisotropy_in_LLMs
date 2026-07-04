from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional, Any, Union



def compute_isoscore(embeddings: np.ndarray) -> float:
    """
    Вычисляет IsoScore для центрированного облака точек.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, n)
        Центрированные данные (среднее вычтено).

    Returns
    -------
    float
        Значение IsoScore в диапазоне [0, 1], где 1 – идеальная изотропия.
    """
    N, n = embeddings.shape
    if N < 2:
        return 0.0
    U, s, Vt = np.linalg.svd(embeddings, full_matrices=False)
    variances = (s ** 2) / (N - 1)
    sigma_D = np.zeros(n)
    sigma_D[:len(s)] = variances

    norm_sd = np.linalg.norm(sigma_D)
    if norm_sd == 0:
        return 0.0
    sigma_D_hat = np.sqrt(n) * sigma_D / norm_sd

    one_vector = np.ones(n)
    numerator = np.linalg.norm(sigma_D_hat - one_vector)
    denominator = np.sqrt(2 * (n - np.sqrt(n)))
    delta = numerator / denominator if denominator != 0 else 0.0

    phi = ((n - delta**2 * (n - np.sqrt(n))) ** 2) / (n ** 2)
    isoscore = (n * phi - 1) / (n - 1) if n > 1 else 0.0
    return float(np.clip(isoscore, 0.0, 1.0))
    

def _get_pad_token_id(model_name: str) -> int:
    """Возвращает ID паддинга для известных моделей."""
    if model_name == "GPT2":
        return 50256
    elif model_name == "Qwen":
        return 151643
    else:
        return 0

def _get_default_layers(model, config) -> List[int]:
    """Возвращает список слоёв по умолчанию."""
    if config.model_name == "GPT2":
        return list(range(model.config.n_layer + 1))
    else:
        return list(range(model.config.num_hidden_layers + 1))

def _extract_valid_embeddings(activation: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Извлекает активации только для не-падинговых токенов."""
    input_ids = input_ids.cpu()  
    B, L, D = activation.shape
    flat_ids = input_ids.view(-1)
    valid_mask = (flat_ids != pad_token_id)
    flat_act = activation.view(-1, D)
    return flat_act[valid_mask]

def _compute_cosine_statistics(X: np.ndarray, X_centered: np.ndarray):
    """Вычисляет верхние треугольники матриц косинусных сходств."""
    def _upper_triangle_from_matrix(mat: np.ndarray):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = mat / norms
        cos_mat = normed @ normed.T
        n = cos_mat.shape[0]
        return cos_mat[np.triu_indices(n, k=1)]

    return _upper_triangle_from_matrix(X), _upper_triangle_from_matrix(X_centered)

def _plot_cosine_histograms(cosine_values: Dict[int, np.ndarray], layers: List[int], title: str = ""):
    """Рисует гистограммы косинусных расстояний по слоям."""
    n_layers = len([l for l in layers if l in cosine_values])
    if n_layers == 0:
        return
    fig, axes = plt.subplots(n_layers, 1, figsize=(15, 4 * n_layers))
    if n_layers == 1:
        axes = [axes]
    for i, layer_idx in enumerate(layers):
        if layer_idx not in cosine_values:
            continue
        values = cosine_values[layer_idx]
        axes[i].hist(values, bins=250, alpha=0.7, label=f'Layer {layer_idx}')
        axes[i].set_xlabel('Cosine similarity', fontsize=16)
        axes[i].set_ylabel('Number of pairs', fontsize=16)
        axes[i].set_title(f'Layer {layer_idx}: mean={values.mean():.4f}, std={values.std():.4f}', fontsize=16)
        axes[i].legend(fontsize=12)
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def _plot_summary_metrics(metrics: Dict[int, Dict], model_name: str):
    """Строит графики среднего косинуса и IsoScore по слоям."""
    layers = sorted(metrics.keys())
    means = [metrics[l]['avg_cosine'] for l in layers]
    stds = [metrics[l]['cosine_std'] for l in layers]
    means_c = [metrics[l]['avg_cosine_centered'] for l in layers]
    stds_c = [metrics[l]['cosine_std_centered'] for l in layers]
    isoscores = [metrics[l]['isoscore'] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Первый график – нецентрированный
    axes[0].plot(layers, means, '-o', color='blue', label='Mean')
    axes[0].fill_between(layers, [m-s for m,s in zip(means,stds)],
                         [m+s for m,s in zip(means,stds)], alpha=0.15, color='blue')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean cosine similarity')
    axes[0].set_title(f'Mean cosine – {model_name}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Второй – центрированный
    axes[1].plot(layers, means_c, '-o', color='red', label='Mean centered')
    axes[1].fill_between(layers, [m-s for m,s in zip(means_c,stds_c)],
                         [m+s for m,s in zip(means_c,stds_c)], alpha=0.15, color='red')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean cosine similarity (centered)')
    axes[1].set_title(f'Mean cosine centered – {model_name}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.show()
    plt.close(fig)

    # IsoScore
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(layers, isoscores, '-o', color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('IsoScore')
    ax.set_title(f'IsoScore – {model_name}')
    ax.grid(True, alpha=0.3)
    plt.show()
    plt.close(fig)

def analyze_cosine_distribution(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    config: TrainingConfig,
    layers_to_plot: Optional[List[int]] = None,
    save_plots: bool = False
) -> Dict[int, Dict[str, float]]:
    """
    Анализирует косинусную анизотропию активаций всех слоёв модели.
    Возвращает словарь метрик по слоям.
    """
    model_device = next(model.parameters()).device
    batch = {k: v.to(model_device) for k, v in batch.items()}
    pad_token_id = _get_pad_token_id(config.model_name)

    if layers_to_plot is None:
        layers_to_plot = _get_default_layers(model, config)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states

    metrics = {}
    cosine_values = {}
    cosine_values_centered = {}

    for layer_idx in layers_to_plot:
        if layer_idx >= len(hidden_states):
            continue
        activation = hidden_states[layer_idx].cpu()
        valid_embeddings = _extract_valid_embeddings(activation, batch["input_ids"], pad_token_id)
        if valid_embeddings.shape[0] == 0:
            continue
        # подвыборка
        if valid_embeddings.shape[0] > config.num_samples:
            idx = torch.randperm(valid_embeddings.shape[0])[:config.num_samples]
            valid_embeddings = valid_embeddings[idx]

        X = valid_embeddings.numpy()
        mean_emb = X.mean(axis=0, keepdims=True)
        X_centered = X - mean_emb

        # метрики
        isoscore = compute_isoscore(X_centered)
        cos_vals, cos_vals_centered = _compute_cosine_statistics(X, X_centered)

        metrics[layer_idx] = {
            'isoscore': isoscore,
            'avg_cosine': float(cos_vals.mean()),
            'cosine_std': float(cos_vals.std()),
            'avg_cosine_centered': float(cos_vals_centered.mean()),
            'cosine_std_centered': float(cos_vals_centered.std()),
        }
        cosine_values[layer_idx] = cos_vals
        cosine_values_centered[layer_idx] = cos_vals_centered

    # Визуализация
    _plot_cosine_histograms(cosine_values, layers_to_plot, title="Non-centered")
    _plot_cosine_histograms(cosine_values_centered, layers_to_plot, title="Centered")
    _plot_summary_metrics(metrics, config.model_name)

    if save_plots:
        plot_dir = setup_plot_saving(config, suffix="results")
        plt.savefig(os.path.join(plot_dir, f"cosine_summary_{config.model_name}.png"), dpi=150, bbox_inches='tight')

    return metrics