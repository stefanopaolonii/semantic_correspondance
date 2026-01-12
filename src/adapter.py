import torch
import torch.nn as nn
from typing import Iterator, Dict, Any
from abc import ABC, abstractmethod


class BackboneAdapter(nn.Module, ABC):
    
    def __init__(self, model_name: str, model_arch: str, fine_tune_layers: int = 0):
        super().__init__()
        self.model_name = model_name
        self.model_arch = model_arch
        self.fine_tune_layers = fine_tune_layers
        
        # Get model configuration
        config = self._get_model_config(model_arch)
        self.feature_dim = config['feature_dim']
        self.patch_size = config['patch_size']
        
        # Load pretrained backbone
        self.backbone = self._load_model_from_repo(model_arch)
        
        # Freeze all weights by default, then selectively unfreeze
        self._freeze_all_weights()
        if fine_tune_layers > 0:
            self.set_fine_tuning(fine_tune_layers)
    
    @abstractmethod
    def _get_model_config(self, model_arch: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _load_model_from_repo(self, model_arch: str) -> nn.Module:
        pass

    @abstractmethod
    def set_fine_tuning(self, num_layers: int):
        pass

    @abstractmethod
    def _tokens_to_dense_map(self, tokens: torch.Tensor, original_size: tuple) -> torch.Tensor:
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[2:]
        tokens = self.backbone(x)
        F_dense = self._tokens_to_dense_map(tokens, original_size)
        return F_dense

    def _freeze_all_weights(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_finetunable_params(self) -> Iterator[nn.Parameter]:
        return (p for p in self.backbone.parameters() if p.requires_grad)
