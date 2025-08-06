#!/usr/bin/env python3
"""
Small Language Model Training Script

This script trains a small GPT-style language model on the TinyStories dataset.
The model is designed to have 10-15 million parameters and generate coherent text.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from model import GPT, GPTConfig
from data_loader import DataLoader
from trainer import Trainer
from config import ModelConfig, TrainingConfig, get_default_configs, print_config_summary
from utils import set_seed, print_device_info, count_parameters

def main():
    parser = argparse.ArgumentParser(description="Train a Small Language Model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--force-reload", action="store_true", help="Force reload dataset")
    parser.add_argument("--no-train", action="store_true", help="Skip training (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get configurations
    model_config, training_config = get_default_configs()
    
    # Set seed for reproducibility
    set_seed(training_config.seed)
    
    # Print configuration summary
    print_config_summary(model_config, training_config)
    print_device_info()
    
    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = DataLoader(
        batch_size=training_config.batch_size,
        block_size=training_config.block_size,
        device=training_config.device
    )
    
    # Prepare data
    print("\nPreparing dataset...")
    data_loader.prepare_data(force_reload=args.force_reload)
    
    # Create model
    print("\nCreating model...")
    gpt_config = GPTConfig(
        vocab_size=model_config.vocab_size,
        block_size=model_config.block_size,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_embd=model_config.n_embd,
        dropout=model_config.dropout,
        bias=model_config.bias
    )
    
    model = GPT(gpt_config)
    param_count = count_parameters(model)
    print(f"Model created with {param_count:,} parameters ({param_count/1e6:.1f}M)")
    
    if args.no_train:
        print("Skipping training as requested.")
        return
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer_config = {
        'learning_rate': training_config.learning_rate,
        'max_iters': training_config.max_iters,
        'warmup_steps': training_config.warmup_steps,
        'min_lr': training_config.min_lr,
        'eval_iters': training_config.eval_iters,
        'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
        'device': training_config.device,
        'dtype': training_config.dtype,
    }
    
    trainer = Trainer(model, data_loader, trainer_config)
    
    # Train model
    print("\nStarting training...")
    model_save_path = os.path.join(args.output_dir, "best_model.pt")
    train_losses, val_losses = trainer.train(save_path=model_save_path)
    
    # Plot losses
    loss_plot_path = os.path.join(args.output_dir, "training_losses.png")
    trainer.plot_losses(save_path=loss_plot_path)
    
    # Test generation
    print("\nTesting text generation...")
    trainer.load_best_model(model_save_path)
    
    test_prompts = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "The cat sat on the mat and"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = trainer.generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
        print(f"Generated: {generated}")
        print("-" * 80)
    
    print(f"\nTraining completed! Model saved to {model_save_path}")

if __name__ == "__main__":
    main()

