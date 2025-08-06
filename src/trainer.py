import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from contextlib import nullcontext
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.max_iters = config.get('max_iters', 20000)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.min_lr = config.get('min_lr', 5e-4)
        self.eval_iters = config.get('eval_iters', 500)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 32)
        self.device = config.get('device', 'cpu')
        self.dtype = config.get('dtype', 'float32')
        
        # Setup device and context
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            betas=(0.9, 0.95), 
            weight_decay=0.1, 
            eps=1e-9
        )
        
        scheduler_warmup = LinearLR(self.optimizer, total_iters=self.warmup_steps)
        scheduler_decay = CosineAnnealingLR(self.optimizer, T_max=self.max_iters - self.warmup_steps, eta_min=self.min_lr)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[self.warmup_steps])
        
        # Setup gradient scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def estimate_loss(self):
        """Estimate loss on train and validation sets."""
        out = {}
        self.model.eval()
        with torch.inference_mode():
            for split in ['train', 'val']:
                losses = torch.zeros(self.eval_iters)
                for k in range(self.eval_iters):
                    X, Y = self.data_loader.get_batch(split)
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        self.model.train()
        return out
    
    def train(self, save_path="best_model.pt"):
        """Train the model."""
        print(f"Starting training for {self.max_iters} iterations...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.model = self.model.to(self.device)
        self.model.train()
        
        for epoch in tqdm(range(self.max_iters), desc="Training"):
            # Evaluation
            if epoch % self.eval_iters == 0 and epoch != 0:
                losses = self.estimate_loss()
                print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.5f}")
                
                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
                
                # Save best model
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"New best model saved with val loss: {self.best_val_loss:.4f}")
            
            # Training step
            X, y = self.data_loader.get_batch("train")
            
            with self.ctx:
                logits, loss = self.model(X, y)
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            
            if ((epoch + 1) % self.gradient_accumulation_steps == 0) or (epoch + 1 == self.max_iters):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            self.scheduler.step()
        
        print("Training completed!")
        return self.train_losses, self.val_losses
    
    def plot_losses(self, save_path="training_losses.png"):
        """Plot training and validation losses."""
        if not self.train_losses or not self.val_losses:
            print("No loss data to plot. Run training first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, 'g', label='Training Loss')
        plt.plot(self.val_losses, 'r', label='Validation Loss')
        plt.xlabel(f"Steps (Every {self.eval_iters} epochs)")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss plot saved to {save_path}")
    
    def load_best_model(self, model_path="best_model.pt"):
        """Load the best saved model."""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found!")
    
    def generate_text(self, prompt, max_new_tokens=200, temperature=1.0, top_k=None):
        """Generate text from a prompt."""
        self.model.eval()
        context = torch.tensor(self.data_loader.encode(prompt)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens, temperature, top_k)
        
        return self.data_loader.decode(generated.squeeze().tolist())

