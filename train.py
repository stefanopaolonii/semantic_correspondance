import os
import sys
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import BASE_CONFIG
from src import DINOv2Adapter, DINOv3Adapter, CLIPAdapter, SAMAdapter
from dataset import SPairDataset, PFPascalDataset, PFWillowDataset
from utils.trainer import CorrespondenceTrainer
from utils.evaluator import CorrespondenceEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic correspondence model')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['dinov2', 'dinov3', 'clip', 'sam'],
                        help='Backbone model')
    parser.add_argument('--model_arch', type=str, required=True,
                        help='Model architecture (e.g., vits14, vitb16)')
    parser.add_argument('--fine_tune_layers', type=int, required=True,
                        help='Number of layers to fine-tune from the end')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['spair', 'pfpascal', 'pfwillow'],
                        help='Dataset name (default: from config)')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Dataset resolution (default: from model config)')
    parser.add_argument('--category', type=str, default='all',
                        help='Category to train on (default: all)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Evaluation arguments
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Evaluate every N steps (None = end of epoch only)')

    return parser.parse_args()


def get_config(args):
    cfg = BASE_CONFIG.clone()
    
    model_configs = {
        'dinov2': cfg.DINOV2,
        'dinov3': cfg.DINOV3,
        'clip': cfg.CLIP,
        'sam': cfg.SAM,
    }
    
    model_cfg = model_configs[args.model]
    
    cfg.DATASET.NAME = args.dataset if args.dataset is not None else cfg.DATASET.NAME
    cfg.DATASET.IMG_SIZE = args.resolution if args.resolution is not None else model_cfg.IMG_SIZE
    cfg.DATASET.MEAN = model_cfg.MEAN 
    cfg.DATASET.STD = model_cfg.STD 
    
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.LR = args.lr if args.lr is not None else cfg.TRAIN.LR
    cfg.TRAIN.EPOCHS = args.epochs if args.epochs is not None else cfg.TRAIN.EPOCHS
    
    return cfg


def get_model(model_name: str, model_arch: str, fine_tune_layers: int):
    model_map = {
        'dinov2': DINOv2Adapter,
        'dinov3': DINOv3Adapter,
        'clip': CLIPAdapter,
        'sam': SAMAdapter,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name](model_arch=model_arch, fine_tune_layers=fine_tune_layers)


def get_dataset(dataset_name: str, cfg, split: str, category: str = 'all'):
    dataset_map = {
        'spair': SPairDataset,
        'pfpascal': PFPascalDataset,
        'pfwillow': PFWillowDataset,
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_map[dataset_name](cfg, split=split, category=category)


def main():
    args = parse_args() 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = get_config(args)
    
    # Print configuration
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"  Model:            {args.model} ({args.model_arch})")
    print(f"  Fine-tune layers: {args.fine_tune_layers}")
    print(f"  Dataset:          {cfg.DATASET.NAME} (category: {args.category})")
    print(f"  Batch size:       {cfg.TRAIN.BATCH_SIZE}")
    print(f"  Epochs:           {cfg.TRAIN.EPOCHS}")
    print(f"  Learning rate:    {cfg.TRAIN.LR}")
    print(f"  Device:           {device}")
    print("=" * 80)
    
    train_dataset = get_dataset(cfg.DATASET.NAME, cfg, split='trn', category=args.category)
    val_dataset = get_dataset(cfg.DATASET.NAME, cfg, split='val', category=args.category)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    
    checkpoint_dir = Path(PROJECT_ROOT) / cfg.TRAIN.CHECKPOINT_DIR / args.model / 'trained'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(checkpoint_dir)
    
    model = get_model(args.model, args.model_arch, args.fine_tune_layers)
    model = model.to(device)
    
    evaluator = CorrespondenceEvaluator(cfg, device)
    trainer = CorrespondenceTrainer(
        model=model,
        device=device,
        cfg=cfg,
        evaluator=evaluator
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_interval=args.eval_interval,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == '__main__':
    main()
