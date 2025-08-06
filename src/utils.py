import torch
import numpy as np
import random
import os
import json
from typing import Dict, Any

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config(config_dict: Dict[str, Any], filepath: str):
    """Save configuration to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {filepath}")
    return config

def format_time(seconds):
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_device_info():
    """Get information about available devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name()
        info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated()
        info['cuda_memory_cached'] = torch.cuda.memory_reserved()
    
    return info

def print_device_info():
    """Print device information."""
    info = get_device_info()
    print("Device Information:")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"Current Device: {info['cuda_current_device']}")
        print(f"Device Name: {info['cuda_device_name']}")
        print(f"Total Memory: {info['cuda_memory_total'] / 1e9:.1f} GB")
        print(f"Allocated Memory: {info['cuda_memory_allocated'] / 1e9:.3f} GB")
        print(f"Cached Memory: {info['cuda_memory_cached'] / 1e9:.3f} GB")

def create_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """Create a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch'], checkpoint['loss']

def estimate_memory_usage(model, batch_size, sequence_length):
    """Estimate memory usage for training."""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Rough estimate for activations (this is approximate)
    activation_memory = batch_size * sequence_length * model.config.n_embd * 4  # 4 bytes per float32
    
    # Gradients have same size as parameters
    gradient_memory = param_memory
    
    # Optimizer states (Adam has 2 states per parameter)
    optimizer_memory = param_memory * 2
    
    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'parameters': param_memory / 1e9,
        'activations': activation_memory / 1e9,
        'gradients': gradient_memory / 1e9,
        'optimizer': optimizer_memory / 1e9,
        'total': total_memory / 1e9
    }

def print_memory_estimate(model, batch_size, sequence_length):
    """Print memory usage estimate."""
    memory = estimate_memory_usage(model, batch_size, sequence_length)
    print("Estimated Memory Usage (GB):")
    print(f"Parameters: {memory['parameters']:.2f}")
    print(f"Activations: {memory['activations']:.2f}")
    print(f"Gradients: {memory['gradients']:.2f}")
    print(f"Optimizer: {memory['optimizer']:.2f}")
    print(f"Total: {memory['total']:.2f}")

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()

