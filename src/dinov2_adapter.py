import sys
import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE_PATH.parent.parent
DINO_REPO_PATH = ROOT_DIR / "models" / "dinov2"
WEIGHTS_DIR = ROOT_DIR / "asset" / "weights" / "dinov2"

if str(DINO_REPO_PATH) not in sys.path:
    sys.path.append(str(DINO_REPO_PATH))

from dinov2.hub.backbones import (
    dinov2_vits14, 
    dinov2_vitb14, 
    dinov2_vitl14
)

from .adapter import BackboneAdapter


class DINOv2Adapter(BackboneAdapter):
    """
    Adapter for DINOv2 Vision Transformer backbone.
    Extracts dense patch features from DINOv2 models for semantic correspondence.
    """
    
    MODEL_CONFIGS = {
        'vits14': {
            'arch_func': dinov2_vits14,          
            'checkpoint': 'dinov2_vits14_pretrain.pth', 
            'feature_dim': 384, 
            'patch_size': 14, 
            'num_blocks': 12
        },
        'vitb14': {
            'arch_func': dinov2_vitb14,           
            'checkpoint': 'dinov2_vitb14_pretrain.pth', 
            'feature_dim': 768, 
            'patch_size': 14, 
            'num_blocks': 12
        },
        'vitl14': {
            'arch_func': dinov2_vitl14,           
            'checkpoint': 'dinov2_vitl14_pretrain.pth', 
            'feature_dim': 1024, 
            'patch_size': 14, 
            'num_blocks': 24
        },
        'vits14_ft2': {
            'arch_func': dinov2_vits14,
            'checkpoint': 'trained/dinov2_vits14_ft2.pth',
            'feature_dim': 384,
            'patch_size': 14,
            'num_blocks': 12
        },
        'vits14_ft4': {
            'arch_func': dinov2_vits14,
            'checkpoint': 'trained/dinov2_vits14_ft4.pth',
            'feature_dim': 384,
            'patch_size': 14,
            'num_blocks': 12
        },
        'vits14_ft6': {
            'arch_func': dinov2_vits14,
            'checkpoint': 'trained/dinov2_vits14_ft6.pth',
            'feature_dim': 384,
            'patch_size': 14,
            'num_blocks': 12
        },
        'vitb14_ft2': {
            'arch_func': dinov2_vitb14,
            'checkpoint': 'trained/dinov2_vitb14_ft2.pth',
            'feature_dim': 768,
            'patch_size': 14,
            'num_blocks': 12
        },
        'vitb14_ft4': {
            'arch_func': dinov2_vitb14,
            'checkpoint': 'trained/dinov2_vitb14_ft4.pth',
            'feature_dim': 768,
            'patch_size': 14,
            'num_blocks': 12
        },
        'vitb14_ft6': {
            'arch_func': dinov2_vitb14,
            'checkpoint': 'trained/dinov2_vitb14_ft6.pth',
            'feature_dim': 768,
            'patch_size': 14,
            'num_blocks': 12
        }
    }
    
    def __init__(self, model_arch: str, fine_tune_layers: int = 0):
        super().__init__(model_name='dinov2', model_arch=model_arch, fine_tune_layers=fine_tune_layers)

    def _get_model_config(self, model_arch: str) -> Dict[str, Any]:
        if model_arch not in self.MODEL_CONFIGS:
            available = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"DINOv2 model '{model_arch}' not supported. Available: {available}")
        return self.MODEL_CONFIGS[model_arch]

    def _load_model_from_repo(self, model_arch: str) -> nn.Module:
        config = self._get_model_config(model_arch)
        model_arch_func = config['arch_func']
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
            else:
                raise FileNotFoundError(
                    f"Pretrained checkpoint not found at {checkpoint_path}. "
                    f"Please download the weights first."
                )
        
        print(f"Loading DINOv2 weights from: {checkpoint_path}")
        
        # Create model architecture
        model = model_arch_func(pretrained=False)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting dense patch features."""
        original_size = x.shape[2:]
        outs = self.backbone.forward_features(x)
        return self._tokens_to_dense_map(outs, original_size)

    def set_fine_tuning(self, num_layers: int):
        """Unlock the last N transformer blocks for fine-tuning."""
        all_blocks = self.backbone.blocks
        total_blocks = len(all_blocks)
        
        if num_layers > 0:
            start_layer = total_blocks - num_layers
            for i in range(start_layer, total_blocks):
                block = all_blocks[i]
                for param in block.parameters():
                    param.requires_grad = True
            print(f"Unlocked DINOv2 layers from {start_layer} to {total_blocks-1}")

    def _tokens_to_dense_map(self, outs: Dict[str, torch.Tensor], original_size: tuple) -> torch.Tensor:
        """Convert patch tokens to dense feature map."""
        patch_tokens = outs['x_norm_patchtokens']
        
        B, N, C = patch_tokens.shape
        H_in, W_in = original_size
        
        H_feat = H_in // self.patch_size
        W_feat = W_in // self.patch_size
        
        F_dense = patch_tokens.transpose(1, 2).reshape(B, C, H_feat, W_feat)
        return F_dense
