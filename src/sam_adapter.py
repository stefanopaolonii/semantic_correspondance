import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE_PATH.parent.parent
SAM_REPO_PATH = ROOT_DIR / "models" / "segment-anything"
WEIGHTS_DIR = ROOT_DIR / "asset" / "weights" / "sam"

if str(SAM_REPO_PATH) not in sys.path:
    sys.path.append(str(SAM_REPO_PATH))

from segment_anything.build_sam import (
    build_sam_vit_b,
    build_sam_vit_l,
    build_sam_vit_h
)

from .adapter import BackboneAdapter


class SAMAdapter(BackboneAdapter):
    """
    Adapter for SAM (Segment Anything Model) Vision Transformer backbone.
    Extracts dense features from SAM image encoder for semantic correspondence.
    """
    
    MODEL_CONFIGS = {
        'vitb16': {
            'build_func': build_sam_vit_b,
            'checkpoint': 'sam_vit_b_01ec64.pth',
            'feature_dim': 256,
            'patch_size': 16,
        },
        'vitl16': {
            'build_func': build_sam_vit_l,
            'checkpoint': 'sam_vit_l_0b3195.pth',
            'feature_dim': 256,
            'patch_size': 16,
        },
        'vith16': {
            'build_func': build_sam_vit_h,
            'checkpoint': 'sam_vit_h_4b8939.pth',
            'feature_dim': 256,
            'patch_size': 16,
        },
        'vitb16_ft2': {
            'build_func': build_sam_vit_b,
            'checkpoint': 'trained/sam_vitb_ft2.pth',
            'feature_dim': 256,
            'patch_size': 16,
        },
        'vitb16_ft4': {
            'build_func': build_sam_vit_b,
            'checkpoint': 'trained/sam_vitb_ft4.pth',
            'feature_dim': 256,
            'patch_size': 16,
        },
        'vitb16_ft6': {
            'build_func': build_sam_vit_b,
            'checkpoint': 'trained/sam_vitb_ft6.pth',
            'feature_dim': 256,
            'patch_size': 16,
        }
    }

    def __init__(self, model_arch: str, fine_tune_layers: int = 0):
        super().__init__(model_name='sam', model_arch=model_arch, fine_tune_layers=fine_tune_layers)
        self._original_pos_embed = None

    def _get_model_config(self, model_arch: str) -> Dict[str, Any]:
        if model_arch not in self.MODEL_CONFIGS:
            available = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"SAM model '{model_arch}' not supported. Available: {available}")
        return self.MODEL_CONFIGS[model_arch]

    def _load_model_from_repo(self, model_arch: str) -> nn.Module:
        config = self._get_model_config(model_arch)
        build_func = config['build_func']
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

        print(f"Loading SAM weights from: {checkpoint_path}")
        
        if not is_finetuned:
            # Load standard pretrained SAM checkpoint
            sam = build_func(checkpoint=str(checkpoint_path))
            return sam.image_encoder
        
        # Load fine-tuned model
        # First load base SAM model, then apply fine-tuned weights
        if 'vitb' in model_arch:
            base_checkpoint = WEIGHTS_DIR / 'sam_vit_b_01ec64.pth'
        elif 'vitl' in model_arch:
            base_checkpoint = WEIGHTS_DIR / 'sam_vit_l_0b3195.pth'
        elif 'vith' in model_arch:
            base_checkpoint = WEIGHTS_DIR / 'sam_vit_h_4b8939.pth'
        else:
            raise ValueError(f"Cannot determine base model for {model_arch}")
        
        print(f"Loading base SAM model from: {base_checkpoint}")
        sam_model = build_func(checkpoint=str(base_checkpoint))
        
        # Load fine-tuned weights into image_encoder
        print(f"Loading fine-tuned weights from: {checkpoint_path}")
        finetuned_state_dict = torch.load(checkpoint_path, map_location='cpu')
        sam_model.image_encoder.load_state_dict(finetuned_state_dict, strict=True)
        
        return sam_model.image_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with positional embedding interpolation."""
        B, C, H, W = x.shape
        
        # Validate that image dimensions are multiples of patch size
        assert H % self.patch_size == 0, \
            f"Input height {H} is not a multiple of patch size {self.patch_size}"
        assert W % self.patch_size == 0, \
            f"Input width {W} is not a multiple of patch size {self.patch_size}"
        
        # Apply positional encoding interpolation if needed
        self._interpolate_pos_encoding(x, W, H)
        
        features = self.backbone(x)
        return features
    
    def _interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int):
        """Interpolate positional encodings for arbitrary input resolutions."""
        # Cache original positional embedding on first call
        if self._original_pos_embed is None:
            self._original_pos_embed = self.backbone.pos_embed.clone()
        
        # Calculate number of patches from image dimensions
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        
        pos_embed = self._original_pos_embed
        
        # Verify SAM format: (1, grid_h, grid_w, embed_dim)
        if len(pos_embed.shape) != 4:
            return  # Silent fallback if unexpected format
            
        _, og_h, og_w, embed_dim = pos_embed.shape
        
        # Early return if dimensions already match
        if og_h == h0 and og_w == w0:
            self.backbone.pos_embed = torch.nn.Parameter(self._original_pos_embed.clone())
            return
        
        # Convert to float for interpolation
        previous_dtype = self.backbone.pos_embed.dtype
        pos_embed = pos_embed.float()
        
        # Reshape for interpolation: (1, og_h, og_w, dim) -> (1, dim, og_h, og_w)
        pos_embed_reshaped = pos_embed.permute(0, 3, 1, 2)
        
        # Perform bicubic interpolation
        pos_embed_new = F.interpolate(
            pos_embed_reshaped,
            size=(h0, w0),
            mode='bicubic',
            align_corners=False
        )
        
        # Verify output dimensions
        assert (h0, w0) == pos_embed_new.shape[-2:]
        
        # Reshape back: (1, dim, h0, w0) -> (1, h0, w0, dim)
        pos_embed_new = pos_embed_new.permute(0, 2, 3, 1)
        
        # Restore original dtype
        pos_embed_new = pos_embed_new.to(previous_dtype)
        
        # Update backbone's positional embedding
        self.backbone.pos_embed = torch.nn.Parameter(pos_embed_new)

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
            print(f"Unlocked SAM layers from {start_layer} to {total_blocks-1}")

    def _tokens_to_dense_map(self, outs: torch.Tensor, original_size: tuple) -> torch.Tensor:
        """Convert output to dense feature map."""
        return outs
