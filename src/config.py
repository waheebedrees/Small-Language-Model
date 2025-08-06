import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    max_iters: int = 20000
    warmup_steps: int = 1000
    min_lr: float = 5e-4
    eval_iters: int = 500
    batch_size: int = 32
    block_size: int = 128
    gradient_accumulation_steps: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    seed: int = 42

def get_default_configs():
    """Get default model and training configurations."""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Ensure consistency between configs
    model_config.block_size = training_config.block_size
    
    return model_config, training_config

def calculate_model_size(config: ModelConfig) -> int:
    """Calculate the approximate number of parameters in the model."""
    # Token embeddings
    token_emb = config.vocab_size * config.n_embd
    
    # Position embeddings
    pos_emb = config.block_size * config.n_embd
    
    # Transformer blocks
    # Each block has: 2 layer norms + attention + MLP
    layer_norm_params = 2 * config.n_embd * 2  # 2 layer norms, each with weight and bias
    
    # Attention: q, k, v projections + output projection
    attn_params = (3 * config.n_embd * config.n_embd) + (config.n_embd * config.n_embd)
    
    # MLP: fc + proj
    mlp_params = (config.n_embd * 4 * config.n_embd) + (4 * config.n_embd * config.n_embd)
    
    block_params = layer_norm_params + attn_params + mlp_params
    total_block_params = block_params * config.n_layer
    
    # Final layer norm
    final_ln = config.n_embd * 2
    
    # Language model head (shares weights with token embeddings, so don't double count)
    
    total_params = token_emb + pos_emb + total_block_params + final_ln
    
    return total_params

def print_config_summary(model_config: ModelConfig, training_config: TrainingConfig):
    """Print a summary of the configuration."""
    print("=" * 50)
    print("MODEL CONFIGURATION")
    print("=" * 50)
    print(f"Vocabulary size: {model_config.vocab_size:,}")
    print(f"Block size: {model_config.block_size}")
    print(f"Number of layers: {model_config.n_layer}")
    print(f"Number of heads: {model_config.n_head}")
    print(f"Embedding dimension: {model_config.n_embd}")
    print(f"Dropout: {model_config.dropout}")
    print(f"Bias: {model_config.bias}")
    
    model_size = calculate_model_size(model_config)
    print(f"Estimated parameters: {model_size:,} ({model_size/1e6:.1f}M)")
    
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Max iterations: {training_config.max_iters:,}")
    print(f"Warmup steps: {training_config.warmup_steps:,}")
    print(f"Min learning rate: {training_config.min_lr}")
    print(f"Eval iterations: {training_config.eval_iters}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps}")
    print(f"Device: {training_config.device}")
    print(f"Data type: {training_config.dtype}")
    print(f"Random seed: {training_config.seed}")
    print("=" * 50)

