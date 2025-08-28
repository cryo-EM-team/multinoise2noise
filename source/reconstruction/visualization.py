import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


colors = ["blue", "green", "orange", "purple", "brown", "pink", "yellow"]

def visualize_slices(tensor: torch.Tensor) -> Figure:
    """
    Visualize slices of a 3D PyTorch tensor along each axis.

    Parameters:
        tensor (torch.Tensor): A 3D PyTorch tensor representing the density map.
    """
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be 3-dimensional.")

    # Plot slices along each axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Slice at the middle of each dimension
    axes[0].imshow(tensor[tensor.shape[0] // 2, :, :].numpy(), cmap='gray')
    axes[0].set_title("Slice along X-axis")
    axes[0].set_xlabel("Y-axis")
    axes[0].set_ylabel("Z-axis")

    axes[1].imshow(tensor[:, tensor.shape[1] // 2, :].numpy(), cmap='gray')
    axes[1].set_title("Slice along Y-axis")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Z-axis")

    axes[2].imshow(tensor[:, :, tensor.shape[2] // 2].numpy(), cmap='gray')
    axes[2].set_title("Slice along Z-axis")
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")

    return fig

def visualize_projections(tensor: torch.Tensor) -> Figure:
    """
    Visualize slices of a 3D PyTorch tensor along each axis.

    Parameters:
        tensor (torch.Tensor): A 3D PyTorch tensor representing the density map.
    """
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be 3-dimensional.")
    
    # Plot slices along each axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Slice at the middle of each dimension
    axes[0].imshow(tensor.sum(0).numpy(), cmap='gray')
    axes[0].set_title("View along X-axis")
    axes[0].set_xlabel("Y-axis")
    axes[0].set_ylabel("Z-axis")

    axes[1].imshow(tensor.sum(1).numpy(), cmap='gray')
    axes[1].set_title("View along Y-axis")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Z-axis")

    axes[2].imshow(tensor.sum(2).numpy(), cmap='gray')
    axes[2].set_title("View along Z-axis")
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")

    return fig

def visualize_fsc(fsc: dict[str, torch.Tensor], resolution: str, angpix: float, show_line: bool = True) -> Figure:
    """Logs denoised samples as matplotlib figure. 

    Args:
        outputs (Dict[str, torch.Tensor]): Outputs from an epoch (training, validation, test).
        phase: (str): Phase (training, validation, test).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    data_range = None
    resolutions = None
    for idx, (label, data) in enumerate(fsc.items()):
        if resolutions is None:
            data_range = np.arange(1, data.shape[0])
            resolutions = ((data.shape[0] - 1) * 2 * angpix / data_range).round(decimals=3)
        ax.plot(data_range, data[1:].cpu().numpy(), label=label, color=colors[idx % len(colors)])
    if show_line:
        ax.axhline(y=0.143, color='red', alpha=0.5, linestyle='--')
    ax.axhline(y=0., color='black', alpha=0.5)
    ax.set_title(f"Resolution = {resolution} Å")
    ax.set_xlabel('Resolution (Å)')
    ax.set_ylabel('Correlation')
    ax.tick_params(axis='x', rotation=60)
    ax.set_xticks(data_range[0::15])
    ax.set_xticklabels(resolutions[0::15])
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, linestyle='--')
    if len(fsc) > 1:
        ax.legend()

    ticks = ax.get_yticks()
    # Check if any tick is sufficiently close to 0.143
    if not np.any(np.isclose(ticks, 0.143, atol=1e-3)):
        ticks = np.append(ticks, 0.143)
        ticks = np.sort(ticks)
        ax.set_yticks(ticks)

    return fig

def plot_with_weighted_dashes(x: torch.Tensor, 
                              y: torch.Tensor, 
                              w: torch.Tensor, 
                              ax: Axes, 
                              color: str, 
                              label: str, 
                              true_style: str = '-', 
                              false_style: str = '--') -> Figure:
    """
    Plot a line with parts styled as solid or dashed based on binary weights.

    Args:
        x (list or np.ndarray): X-coordinates of the line.
        y (list or np.ndarray): Y-coordinates of the line.
        weights (list or np.ndarray): Binary weights (0 or 1) for each segment of the line.
        ax (Axes): axis on which to plot data.
        label (str): Label of the line.
        true_style (str): Line style for weighted (1) segments (default is solid '-').
        false_style (str): Line style for unweighted (0) segments (default is dashed '--').
    """
    # Ensure inputs are numpy arrays
    x = x.numpy()
    y = y.numpy()
    weights = w.numpy()
    for i in range(len(x) - 1):
        label_ = label if i == len(x)//2 - 1 else None
        ax.plot(x[i:i+2], 
                y[i:i+2], 
                color=color, 
                linestyle=true_style if weights[i] == 1 else false_style, 
                label=label_)
        
def visualize_guiner(data: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], b_factor: str) -> Figure:
    """Logs denoised samples as matplotlib figure. 

    Args:
        outputs (List[Dict[str, Union[int, torch.Tensor]]]): Outputs from an epoch (training, validation, test).
        phase: (str): Phase (training, validation, test).
    """
    fig, ax = plt.subplots(figsize=(7,7))
    for idx, (label, (x, y, w)) in enumerate(data.items()):
        plot_with_weighted_dashes(x, y, w, ax, label=label, color=colors[idx % len(colors)])

    ax.legend()
    ax.set_xlabel("Resolution^2 (1/Å^2)")
    ax.set_ylabel("ln(amplitudes)")
    ax.set_title(f"Guiner plots - Bfactor = {b_factor}")
    ax.grid(True, linestyle='--')
    return fig

def visualize_histogram(hist: dict[str, torch.Tensor], angpix: float, length: int) -> Figure:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.bar(hist.bin_edges[1:], hist.hist, color='red', edgecolor='black', alpha=0.7)
    x_ticks = [f"{tick:.3f}" for tick in length * angpix / hist.bin_edges[1:]]
    ax.set_xticks(hist.bin_edges[1:][::10])
    ax.set_xticklabels(x_ticks[::10])
    ax.tick_params("x", rotation=45)
    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel("Voxel count")
    ax.set_title("Resolution Histogram")
    ax.grid(True, linestyle='--')
    return fig

def visualize_samples(samples: list[dict[str, torch.Tensor]], num_samples: int) -> Figure:
    """Logs denoised samples as matplotlib figure. 

    Args:
        samples (List[Dict[str, torch.Tensor]]): Outputs from an epoch (training, validation, test).
        num_samples (int): Number of samples to visualize.
    """
    num_samples = min(num_samples, len(samples['input']))
    fig, ax = plt.subplots(num_samples, len(samples.keys()), figsize=(4 * num_samples, 3 * len(samples.keys())))

    for i in range(num_samples):
        for key_idx, key in enumerate(samples.keys()):
            ax[i, key_idx].imshow(samples[key][i], cmap="gray")
            ax[i, key_idx].set_title(key)
            ax[i, key_idx].axis('off')

    return fig
