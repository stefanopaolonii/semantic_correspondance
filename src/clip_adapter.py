import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE_PATH.parent.parent
CLIP_REPO_PATH = ROOT_DIR / "models" / "CLIP"
WEIGHTS_DIR = ROOT_DIR / "asset" / "weights" / "clip"

if str(CLIP_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(CLIP_REPO_PATH))

import clip

from .adapter import BackboneAdapter


class CLIPAdapter(BackboneAdapter):
    """
    Adapter for CLIP Vision Transformer backbone.  
    Extracts dense patch features from CLIP models for semantic correspondence.
    """
    
    MODEL_CONFIGS = {
        'vitb32': {
            'model_name': 'ViT-B/32',
            'checkpoint': 'ViT-B-32.pt',
            'feature_dim': 768,
            'patch_size': 32,
            'num_blocks': 12
        },
        'vitb16': {
            'model_name': 'ViT-B/16',
            'checkpoint': 'ViT-B-16.pt',
            'feature_dim': 768,
            'patch_size': 16,
            'num_blocks': 12
        },
        'vitl14': {
            'model_name': 'ViT-L/14',
            'checkpoint': 'ViT-L-14.pt',
            'feature_dim': 1024,
            'patch_size': 14,
            'num_blocks': 24
        },
        'vitb16_ft2': {
            'model_name': 'ViT-B/16',
            'checkpoint': 'trained/clip_vitb16_ft2.pth',
            'feature_dim': 768,
            'patch_size': 16,
            'num_blocks': 12
        },
        'vitb16_ft4': {
            'model_name': 'ViT-B/16',
            'checkpoint': 'trained/clip_vitb16_ft4.pth',
            'feature_dim': 768,
            'patch_size': 16,
            'num_blocks': 12
        },
        'vitb16_ft6': {
            'model_name': 'ViT-B/16',
            'checkpoint': 'trained/clip_vitb16_ft6.pth',
            'feature_dim': 768,
            'patch_size': 16,
            'num_blocks': 12
        },
    }

    def __init__(self, model_arch: str, fine_tune_layers: int = 0):
        super().__init__(model_name='clip', model_arch=model_arch, fine_tune_layers=fine_tune_layers)

    def _get_model_config(self, model_arch: str) -> Dict[str, Any]:
        if model_arch not in self.MODEL_CONFIGS:
            available = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"CLIP model '{model_arch}' not supported. Available: {available}")
        return self.MODEL_CONFIGS[model_arch]

    def _load_model_from_repo(self, model_arch: str) -> nn.Module:
        config = self._get_model_config(model_arch)
        ckpt_filename = config['checkpoint']
        checkpoint_path = WEIGHTS_DIR / ckpt_filename
        
        # Check if this is a fine-tuned model
        is_finetuned = 'trained/' in ckpt_filename
        
        if not checkpoint_path.exists():
            if is_finetuned:
                raise FileNotFoundError(
                    f"Fine-tuned checkpoint not found at {checkpoint_path}. "
                    f"Please ensure the trained weights are in the correct location."
                )
            
            print(f"Checkpoint not found at {checkpoint_path}")
            print(f"Attempting to download {config['model_name']}...")
            try:
                model, _ = clip.load(config['model_name'], device='cpu')
                return model.visual
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not load CLIP model. "
                    f"Please download weights first. Error: {e}"
                )
        
        print(f"Loading CLIP weights from: {checkpoint_path}")
        
        if is_finetuned:
            # Load fine-tuned model: first create base architecture, then load weights
            base_model, _ = clip.load(config['model_name'], device='cpu')
            visual_encoder = base_model.visual
            
            # Load the fine-tuned weights
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different possible state_dict formats
            if 'model_state_dict' in state_dict:
                visual_encoder.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                visual_encoder.load_state_dict(state_dict['state_dict'])
            else:
                visual_encoder.load_state_dict(state_dict)
            
            print(f"Successfully loaded fine-tuned weights from {checkpoint_path}")
            return visual_encoder
        else:
            # Load pretrained CLIP model
            model, _ = clip.load(checkpoint_path, device='cpu')
            return model.visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract dense feature map from input image."""
        B, C, H, W = x.shape
        
        # Validate that image dimensions are multiples of patch size
        assert H % self.patch_size == 0, \
            f"Input height {H} is not a multiple of patch size {self.patch_size}"
        assert W % self.patch_size == 0, \
            f"Input width {W} is not a multiple of patch size {self.patch_size}"
        
        original_size = (H, W)
        
        # Extract patch tokens before final projection
        patch_tokens = self._extract_patch_tokens(x)
        
        return self._tokens_to_dense_map(patch_tokens, original_size)

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from ViT, excluding CLS token."""
        # Patch embedding
        x = self.backbone.conv1(x)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid**2)
        x = x.permute(0, 2, 1)  # (B, grid**2, width)
        
        # Add CLS token
        x = torch.cat([
            self.backbone.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ), 
            x
        ], dim=1)  # (B, grid**2 + 1, width)
        
        # Interpolate positional embedding if needed
        pos_embed = self._interpolate_pos_encoding(x, x.shape[2], x.shape[2])
        x = x + pos_embed.to(x.dtype)
        
        # Pre-transformer layer norm
        x = self.backbone.ln_pre(x)
        
        # Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.backbone.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Post-transformer layer norm
        x = self.backbone.ln_post(x)
        
        # Extract only patch tokens (exclude CLS token at position 0)
        patch_tokens = x[:, 1:, :]  # (B, grid**2, width)
        
        return patch_tokens

    def _interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate positional embeddings for different resolutions."""
        pos_embed_orig = self.backbone.positional_embedding  # (N_orig+1, C) or (1, N_orig+1, C)
        
        # Handle shape: CLIP might have (N+1, C) instead of (1, N+1, C)
        if pos_embed_orig.ndim == 2:
            pos_embed_orig = pos_embed_orig.unsqueeze(0)
        
        N_input = x.shape[1]  # Number of tokens including CLS
        N_orig = pos_embed_orig.shape[1]
        
        # If sizes match, return original
        if N_input == N_orig:
            return pos_embed_orig.squeeze(0) if self.backbone.positional_embedding.ndim == 2 else pos_embed_orig
        
        # Split CLS and patch embeddings
        cls_pos_embed = pos_embed_orig[:, :1, :]  # (1, 1, C)
        patch_pos_embed = pos_embed_orig[:, 1:, :]  # (1, N_orig-1, C)
        
        # Calculate grid sizes
        N_orig_patches = N_orig - 1
        N_input_patches = N_input - 1
        
        grid_orig = int(N_orig_patches ** 0.5)
        grid_new = int(N_input_patches ** 0.5)
        
        assert grid_orig ** 2 == N_orig_patches, f"Original pos_embed not square: {N_orig_patches}"
        assert grid_new ** 2 == N_input_patches, f"Input tokens not square: {N_input_patches}"
        
        # Reshape patch embeddings to 2D grid
        C = patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, grid_orig, grid_orig, C)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)
        
        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(grid_new, grid_new),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, C)
        patch_pos_embed = patch_pos_embed.reshape(1, N_input_patches, C)
        
        # Combine CLS and patches
        pos_embed_new = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
        
        # Return in original format
        return pos_embed_new.squeeze(0) if self.backbone.positional_embedding.ndim == 2 else pos_embed_new

    def set_fine_tuning(self, num_layers: int):
        """Unlock the last N transformer blocks for fine-tuning."""
        all_blocks = self.backbone.transformer.resblocks
        total_blocks = len(all_blocks)
        
        if num_layers > 0:
            start_layer = total_blocks - num_layers
            for i in range(start_layer, total_blocks):
                block = all_blocks[i]
                for param in block.parameters():
                    param.requires_grad = True
            print(f"Unlocked CLIP layers from {start_layer} to {total_blocks-1}")

    def _tokens_to_dense_map(self, tokens: torch.Tensor, original_size: tuple) -> torch.Tensor:
        """Convert patch tokens to dense feature map."""
        B, N, C = tokens.shape
        H_in, W_in = original_size
        
        H_feat = H_in // self.patch_size
        W_feat = W_in // self.patch_size
        
        assert N == H_feat * W_feat, \
            f"Token count mismatch: expected {H_feat * W_feat}, got {N}"
        
        F_dense = tokens.transpose(1, 2).reshape(B, C, H_feat, W_feat)
        return F_dense
