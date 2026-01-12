import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.loss import GaussianCrossEntropyLoss

class CorrespondenceTrainer:
    def __init__(self, model, device, cfg, evaluator=None):
        self.device = device
        self.cfg = cfg
        self.evaluator = evaluator
        self.best_pck = 0.0
        self.loss_fn = GaussianCrossEntropyLoss()
        self.model = model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            params, 
            lr=self.cfg.TRAIN.LR, 
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.cfg.TRAIN.LR_MILESTONES, 
            gamma=self.cfg.TRAIN.LR_GAMMA
        )

    def save_checkpoint(self, path, epoch, weights_only=False):
        """
        Save model checkpoint.
        """
        if weights_only:
            torch.save(self.model.backbone.state_dict(), path)
        else:
            checkpoint = {
                'model_state_dict': self.model.backbone.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'best_pck': self.best_pck
            }
            torch.save(checkpoint, path)
            
        print(f"Checkpoint saved ({'weights only' if weights_only else 'full'}): {path}")
    
    def load_checkpoint(self, path):
        """
        Load checkpoint and restore training state. Returns next epoch to start from.
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}, starting from scratch.")
            return 0

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.backbone.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.best_pck = checkpoint.get('best_pck', 0.0)
        saved_epoch = checkpoint.get('epoch', -1)
        
        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from Epoch: {saved_epoch + 1}, Best PCK so far: {self.best_pck:.4f}")
        
        return saved_epoch + 1

    @torch.no_grad()
    def validate(self, val_loader, match_method='argmax', max_batch=None):
        """
        Run validation and return PCK@0.1 score.
        """
        if self.evaluator is None: 
            return 0.0

        eval_img, _ = self.evaluator.evaluate(
            model=self.model,
            loader=val_loader,
            match_method=match_method,
            max_batch=max_batch
        )

        results = eval_img.summerize_result()
        pck = results[f'{match_method}_pck0.1']['all']
        
        print("Validation Result")
        eval_img.print_summarize_result()
        
        return pck

    def train(self, train_loader, val_loader=None, max_steps=None, 
              eval_interval=None, checkpoint_dir=None,
              match_method='argmax', max_val_batch=None):
        """
        Main training loop with automatic checkpointing and validation.
        """
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be specified")
        
        epochs = self.cfg.TRAIN.EPOCHS
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        start_epoch = 0
        
        filename_base = f"{self.model.model_name}_{self.model.model_arch}_ft{self.model.fine_tune_layers}"
        
        resume_name = f"{filename_base}_resume.pth"
        resume_path = os.path.join(checkpoint_dir, resume_name)
        
        if os.path.exists(resume_path):
            start_epoch = self.load_checkpoint(resume_path)
        
        self.model.train()
        global_step = 0
        
        print(f"Training: {epochs} epochs | LR: {self.scheduler.get_last_lr()[0]:.1e} | Best PCK: {self.best_pck:.4f}")
        
        for epoch in range(start_epoch, epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_losses = []
            
            for i, batch in enumerate(pbar):
                if max_steps and i >= max_steps: 
                    break
                
                self.optimizer.zero_grad()
                
                src_img = batch['src_img'].to(self.device)
                trg_img = batch['trg_img'].to(self.device)
                src_kps = batch['src_kps'].to(self.device)
                trg_kps = batch['trg_kps'].to(self.device)
                n_pts = batch['n_pts'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    src_feat = self.model(src_img)
                    trg_feat = self.model(trg_img)
                    H_img, W_img = src_img.shape[2:]
                    
                    lossfn_input = {
                        'src_featmaps': src_feat,
                        'trg_featmaps': trg_feat,
                        'src_kps': src_kps,
                        'trg_kps': trg_kps,
                        'src_imgsize': (H_img, W_img),
                        'trg_imgsize': (H_img, W_img),
                        'npts': n_pts,
                        'softmax_temp': self.cfg.LOSS.SOFTMAX_TEMP,
                        'enable_l2_norm': True
                    }
                    loss = self.loss_fn(**lossfn_input)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_losses.append(loss.item())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{sum(epoch_losses)/len(epoch_losses):.4f}",
                    'best_pck': f"{self.best_pck:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.1e}"
                })
                
                global_step += 1
                
                if val_loader and eval_interval and global_step % eval_interval == 0:
                    print(f"\n--- Validation at step {global_step} ---")
                    val_pck = self.validate(val_loader, match_method=match_method, max_batch=max_val_batch)
                    
                    if val_pck > self.best_pck:
                        self.best_pck = val_pck
                        best_name = f"{filename_base}.pth"
                        best_path = os.path.join(checkpoint_dir, best_name)
                        self.save_checkpoint(best_path, epoch=epoch, weights_only=True)
                    
                    self.model.train()
            
            self.scheduler.step()
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}")
            
            last_path = os.path.join(checkpoint_dir, resume_name)
            self.save_checkpoint(last_path, epoch=epoch)

            if val_loader:
                print(f"\n--- End-of-epoch validation ---")
                val_pck = self.validate(val_loader, match_method=match_method, max_batch=max_val_batch)
                
                if val_pck > self.best_pck:
                    self.best_pck = val_pck
                    best_name = f"{filename_base}.pth"
                    best_path = os.path.join(checkpoint_dir, best_name)
                    self.save_checkpoint(best_path, epoch=epoch, weights_only=True)

        print(f"Training finished. Final Best PCK: {self.best_pck:.4f}")