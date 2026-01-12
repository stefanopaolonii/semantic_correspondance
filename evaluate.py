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
from utils.evaluator import CorrespondenceEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate semantic correspondence model')

    # Experimental arguments
    parser.add_argument('--phase', type=int, required=True,
                        choices=[1, 2, 3, 4],
                        help='Experiment phase')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['dinov2', 'dinov3', 'clip', 'sam'],
                        help='Backbone model')
    parser.add_argument('--model_arch', type=str, required=True,
                        help='Model architecture (e.g., vits14, vitb16)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['spair', 'pfpascal', 'pfwillow'],
                        help='Dataset name (default: from config)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Dataset resolution (default: from model config)')
    parser.add_argument('--category', type=str, default='all',
                        help='Category to evaluate (default: all)')
    
    # Evaluation arguments
    parser.add_argument('--match_method', type=str, default='argmax',
                        choices=['argmax', 'windowed_softargmax'],
                        help='Matching method: argmax or windowed_softargmax')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_batch', type=int, default=None,
                        help='Max batches to evaluate (None = all)')
    
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
    
    return cfg


def get_model(model_name: str, model_arch: str):
    model_map = {
        'dinov2': DINOv2Adapter,
        'dinov3': DINOv3Adapter,
        'clip': CLIPAdapter,
        'sam': SAMAdapter,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name](model_arch=model_arch, fine_tune_layers=0)


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
    
    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"  Model:        {args.model} ({args.model_arch})")
    print(f"  Dataset:      {args.dataset} ({args.split} split)")
    print(f"  Category:     {args.category}")
    print(f"  Resolution:   {args.resolution}")
    print(f"  Match method: {args.match_method}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Device:       {device}")
    print("=" * 80)
    
    dataset = get_dataset(cfg.DATASET.NAME, cfg, split=args.split, category=args.category)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = get_model(args.model, args.model_arch)
    
    model = model.to(device)
    model.eval()
    
    evaluator = CorrespondenceEvaluator(cfg, device)
    
    eval_img, eval_pt = evaluator.evaluate(
        model=model,
        loader=loader,
        match_method=args.match_method,
        max_batch=args.max_batch
    )
    
    print("RESULTS - PCK by Image")
    eval_img.print_summarize_result()
    
    print("RESULTS - PCK by Point")
    eval_pt.print_summarize_result()
    
    # Create results directory structure
    results_dir = Path(PROJECT_ROOT) / 'asset' / 'results' / args.model / f'phase{args.phase}' / args.model_arch / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
        
    path_img = results_dir / f'phase{args.phase}_{args.match_method}_image_report.txt'
    path_kpt = results_dir / f'phase{args.phase}_{args.match_method}_kpt_report.txt'
        
    eval_img.save_result(str(path_img))
    eval_pt.save_result(str(path_kpt))
        
    print(f"\nResults saved:")
    print(f"  Image report: {path_img}")
    print(f"  Point report: {path_kpt}")

if __name__ == '__main__':
    main()
