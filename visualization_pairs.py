import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import BASE_CONFIG
from src import DINOv2Adapter, DINOv3Adapter, CLIPAdapter, SAMAdapter
from dataset import SPairDataset, PFPascalDataset, PFWillowDataset
from utils.predictor import CorrespondencePredictor
from utils.evaluator import rescale_coords

KEYPOINT_COLORS = [
    '#00FF00',  # Green
    '#FF69B4',  # Pink
    '#00BFFF',  # Deep Sky Blue
    '#9400D3',  # Dark Violet
    '#FF4500',  # Orange Red
    '#FFD700',  # Gold
    '#00CED1',  # Dark Turquoise
    '#FF1493',  # Deep Pink
    '#32CD32',  # Lime Green
    '#8A2BE2',  # Blue Violet
    '#FF6347',  # Tomato
    '#4169E1',  # Royal Blue
    '#ADFF2F',  # Green Yellow
    '#DC143C',  # Crimson
    '#00FA9A',  # Medium Spring Green
]

MODEL_LABELS = {
    'dinov2': 'DINOv2',
    'dinov3': 'DINOv3',
    'sam': 'SAM',
    'clip': 'CLIP'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize semantic correspondence predictions'
    )
    
    # Sample selection
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the image pair to visualize')
    parser.add_argument('--max_keypoints', type=int, default=None,
                        help='Maximum number of keypoints to show (default: all)')
    parser.add_argument('--keypoint_indices', type=int, nargs='+', default=None,
                        help='Specific keypoint indices to visualize')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['dinov2', 'dinov3', 'clip', 'sam'],
                        help='Backbone model')
    parser.add_argument('--model_arch', type=str, required=True,
                        help='Model architecture (e.g., vits14, vitb16)')
    parser.add_argument('--model_arch_tuned', type=str, default=None,
                        help='Tuned model architecture for comparison (optional)')
    
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
                        help='Category to visualize (default: all)')
    
    # Prediction arguments
    parser.add_argument('--match_method', type=str, default='argmax',
                        choices=['argmax', 'windowed_softargmax'],
                        help='Matching method for baseline model')
    parser.add_argument('--match_method_tuned', type=str, default='windowed_softargmax',
                        choices=['argmax', 'windowed_softargmax'],
                        help='Matching method for tuned model')
    
    # Visualization options
    parser.add_argument('--show_gt', action='store_true', default=True,
                        help='Show ground truth markers')
    parser.add_argument('--no_gt', action='store_true',
                        help='Hide ground truth markers')
    parser.add_argument('--line_width', type=float, default=2.0,
                        help='Width of correspondence lines')
    parser.add_argument('--marker_size', type=int, default=10,
                        help='Size of keypoint markers')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Transparency of lines')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 12],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the figure')
    parser.add_argument('--no_title', action='store_true',
                        help='Remove title from figure')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save figure')
    parser.add_argument('--show', action='store_true',
                        help='Display figure interactively')
    
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


def get_predictions(model, predictor, src_img, trg_img, kp_src_all, method='argmax'):
    device = src_img.device
    
    with torch.no_grad():
        feat_src = model(src_img)
        feat_trg = model(trg_img)
        
        _, _, H_img, W_img = trg_img.shape
        _, _, H_feat, W_feat = feat_trg.shape
        
        predictions = []
        for i in range(len(kp_src_all)):
            kp_src = kp_src_all[i:i+1].unsqueeze(0).to(device)  # [1, 1, 2]
            
            kp_trg_feat = predictor.predict_with_image_coords(
                feat_src, feat_trg, kp_src, H_img, W_img, method=method
            )
            
            kp_trg_img = rescale_coords(kp_trg_feat, H_img, W_img, H_feat, W_feat)
            predictions.append(kp_trg_img[0, 0].numpy())
        
        return np.array(predictions)


def load_original_images(dataset, sample_idx, target_size=None):
    sample = dataset[sample_idx]
    
    # Construct paths
    cls_name = dataset.cls[dataset.cls_ids[sample_idx]]
    src_path = os.path.join(dataset.img_path, cls_name, sample['src_imname'])
    trg_path = os.path.join(dataset.img_path, cls_name, sample['trg_imname'])
    
    src_img_pil = Image.open(src_path).convert('RGB')
    trg_img_pil = Image.open(trg_path).convert('RGB')
    
    # Resize to target size if specified
    if target_size is not None:
        src_img_pil = src_img_pil.resize((target_size, target_size), Image.BILINEAR)
        trg_img_pil = trg_img_pil.resize((target_size, target_size), Image.BILINEAR)
    
    return src_img_pil, trg_img_pil


def visualize_correspondence_pair(
    src_img_pil,
    trg_img_pil,
    kp_src,
    kp_trg_gt,
    kp_trg_pred,
    kp_trg_pred_tuned=None,
    keypoint_indices=None,
    show_gt=True,
    line_width=2.0,
    marker_size=10,
    alpha=0.8,
    figsize=(8, 12),
    title=None,
    model_label='Model'
):
    """
    Create visualization of correspondence predictions.
    
    Args:
        src_img_pil: Source PIL image
        trg_img_pil: Target PIL image
        kp_src: Source keypoints [N, 2] in original image coords
        kp_trg_gt: Ground truth target keypoints [N, 2]
        kp_trg_pred: Predicted target keypoints [N, 2] (baseline)
        kp_trg_pred_tuned: Predicted target keypoints [N, 2] (tuned, optional)
        keypoint_indices: Which keypoints to visualize (None = all)
        show_gt: Whether to show ground truth
        line_width: Width of correspondence lines
        marker_size: Size of markers
        alpha: Line transparency
        figsize: Figure size
        title: Optional title
        model_label: Model name for legend
    
    Returns:
        matplotlib figure
    """
    # Convert images to numpy
    src_img = np.array(src_img_pil)
    trg_img = np.array(trg_img_pil)
    
    # Determine which keypoints to show
    n_kp = len(kp_src)
    if keypoint_indices is None:
        keypoint_indices = list(range(n_kp))
    
    # Get image dimensions
    src_h, src_w = src_img.shape[:2]
    trg_h, trg_w = trg_img.shape[:2]
    
    # Use the maximum width for both images (for centering)
    max_w = max(src_w, trg_w)
    total_h = src_h + trg_h
    
    # Calculate figure size to match aspect ratio
    aspect = max_w / total_h
    fig_h = figsize[1]
    fig_w = fig_h * aspect
    
    # Create figure with GridSpec for tight control
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # Create axes that span the full figure, stacked vertically
    # Heights proportional to image heights
    height_ratios = [src_h, trg_h]
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0, wspace=0,
                          left=0, right=1, top=0.95 if title else 1, bottom=0)
    
    ax_src = fig.add_subplot(gs[0])
    ax_trg = fig.add_subplot(gs[1])
    
    # Display images centered
    # Calculate centering offsets
    src_offset = (max_w - src_w) / 2
    trg_offset = (max_w - trg_w) / 2
    
    ax_src.imshow(src_img, extent=[src_offset, src_offset + src_w, src_h, 0])
    ax_trg.imshow(trg_img, extent=[trg_offset, trg_offset + trg_w, trg_h, 0])
    
    # Set axis limits to max width
    ax_src.set_xlim(0, max_w)
    ax_src.set_ylim(src_h, 0)
    ax_trg.set_xlim(0, max_w)
    ax_trg.set_ylim(trg_h, 0)
    
    ax_src.axis('off')
    ax_trg.axis('off')
    
    # Adjust keypoint coordinates for centering
    kp_src_adj = kp_src.copy()
    kp_src_adj[:, 0] += src_offset
    
    kp_trg_gt_adj = kp_trg_gt.copy() if kp_trg_gt is not None else None
    if kp_trg_gt_adj is not None:
        kp_trg_gt_adj[:, 0] += trg_offset
    
    kp_trg_pred_adj = kp_trg_pred.copy() if kp_trg_pred is not None else None
    if kp_trg_pred_adj is not None:
        kp_trg_pred_adj[:, 0] += trg_offset
    
    kp_trg_pred_tuned_adj = kp_trg_pred_tuned.copy() if kp_trg_pred_tuned is not None else None
    if kp_trg_pred_tuned_adj is not None:
        kp_trg_pred_tuned_adj[:, 0] += trg_offset
    
    # Plot keypoints
    for idx, kp_idx in enumerate(keypoint_indices):
        if kp_idx >= n_kp:
            continue
            
        color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]
        
        # Source keypoint (diamond marker)
        ax_src.plot(
            kp_src_adj[kp_idx, 0], kp_src_adj[kp_idx, 1],
            marker='D', markersize=marker_size,
            color=color, markeredgecolor='white', markeredgewidth=1.5,
            zorder=10
        )
        
        # Ground truth on target (+ marker with same color)
        if show_gt and kp_trg_gt_adj is not None:
            ax_trg.plot(
                kp_trg_gt_adj[kp_idx, 0], kp_trg_gt_adj[kp_idx, 1],
                marker='+', markersize=marker_size + 2,
                color=color, markeredgewidth=2.5,
                zorder=12
            )
        
        # Baseline prediction on target (X marker)
        if kp_trg_pred_adj is not None:
            ax_trg.plot(
                kp_trg_pred_adj[kp_idx, 0], kp_trg_pred_adj[kp_idx, 1],
                marker='x', markersize=marker_size,
                color=color, markeredgewidth=2.5,
                zorder=11
            )
        
        # Tuned prediction on target (circle marker)
        if kp_trg_pred_tuned_adj is not None:
            ax_trg.plot(
                kp_trg_pred_tuned_adj[kp_idx, 0], kp_trg_pred_tuned_adj[kp_idx, 1],
                marker='o', markersize=marker_size - 2,
                markerfacecolor='none', markeredgecolor=color, markeredgewidth=2.5,
                zorder=11
            )
    
    # Draw correspondence lines using ConnectionPatch
    for idx, kp_idx in enumerate(keypoint_indices):
        if kp_idx >= n_kp:
            continue
            
        color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]
        
        # Line from source to baseline prediction
        if kp_trg_pred_adj is not None:
            con = ConnectionPatch(
                xyA=(kp_src_adj[kp_idx, 0], kp_src_adj[kp_idx, 1]),
                xyB=(kp_trg_pred_adj[kp_idx, 0], kp_trg_pred_adj[kp_idx, 1]),
                coordsA="data", coordsB="data",
                axesA=ax_src, axesB=ax_trg,
                color=color, linewidth=line_width, alpha=alpha,
                linestyle='-' if kp_trg_pred_tuned_adj is None else '--',
                zorder=5
            )
            fig.add_artist(con)
        
        # Line from source to tuned prediction (solid)
        if kp_trg_pred_tuned_adj is not None:
            con_tuned = ConnectionPatch(
                xyA=(kp_src_adj[kp_idx, 0], kp_src_adj[kp_idx, 1]),
                xyB=(kp_trg_pred_tuned_adj[kp_idx, 0], kp_trg_pred_tuned_adj[kp_idx, 1]),
                coordsA="data", coordsB="data",
                axesA=ax_src, axesB=ax_trg,
                color=color, linewidth=line_width, alpha=alpha,
                linestyle='-',
                zorder=6
            )
            fig.add_artist(con_tuned)
    
    # Create legend elements
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markersize=8, label='Source Keypoint'),
    ]
    
    if show_gt:
        legend_elements.append(
            Line2D([0], [0], marker='+', color='gray', markersize=10,
                   linestyle='None', markeredgewidth=2, label='Ground Truth')
        )
    
    if kp_trg_pred is not None and kp_trg_pred_tuned is not None:
        legend_elements.extend([
            Line2D([0], [0], marker='x', color='gray', markersize=8,
                   linestyle='None', markeredgewidth=2, label=f'{model_label} Baseline'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='gray',
                   markersize=8, markerfacecolor='none', markeredgewidth=2,
                   label=f'{model_label} Tuned'),
        ])
    elif kp_trg_pred is not None:
        legend_elements.append(
            Line2D([0], [0], marker='x', color='gray', markersize=8,
                   linestyle='None', markeredgewidth=2, label=f'{model_label} Prediction')
        )
    
    # Place legend at top right, horizontal
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98 if title else 1.0),
        fontsize=8,
        framealpha=0.9,
        edgecolor='gray',
        ncol=len(legend_elements)
    )
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)
    
    return fig


def main():
    args = parse_args()
    
    if args.no_gt:
        args.show_gt = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = get_config(args)
    
    resolution = cfg.DATASET.IMG_SIZE
    model_label = MODEL_LABELS.get(args.model, args.model.upper())
    
    # Print configuration
    print("=" * 80)
    print("VISUALIZATION CONFIGURATION")
    print("=" * 80)
    print(f"  Model:          {args.model} ({args.model_arch})")
    if args.model_arch_tuned:
        print(f"  Model (tuned):  {args.model} ({args.model_arch_tuned})")
    print(f"  Dataset:        {cfg.DATASET.NAME} ({args.split} split)")
    print(f"  Category:       {args.category}")
    print(f"  Resolution:     {resolution}")
    print(f"  Sample index:   {args.sample_idx}")
    print(f"  Match method:   {args.match_method}")
    if args.model_arch_tuned:
        print(f"  Match (tuned):  {args.match_method_tuned}")
    print(f"  Device:         {device}")
    print("=" * 80)
    
    # Load dataset
    dataset = get_dataset(cfg.DATASET.NAME, cfg, split=args.split, category=args.category)
    
    if args.sample_idx >= len(dataset):
        raise ValueError(f"Sample index {args.sample_idx} out of range (max: {len(dataset) - 1})")
    
    sample = dataset[args.sample_idx]
    
    print(f"\nSample info:")
    print(f"  Category:     {sample.get('category', sample.get('pair_class', 'unknown'))}")
    print(f"  N keypoints:  {sample['n_pts']}")
    
    # Load images at model resolution for display
    src_img_pil, trg_img_pil = load_original_images(dataset, args.sample_idx, target_size=resolution)
    
    # Images are now at model resolution, no need to scale
    src_orig_w, src_orig_h = src_img_pil.size  # Will be (resolution, resolution)
    trg_orig_w, trg_orig_h = trg_img_pil.size  # Will be (resolution, resolution)
    
    # Get keypoints in model resolution (already correct, no scaling needed)
    kp_src_model = sample['src_kps'].cpu()  # [N, 2]
    kp_trg_gt_model = sample['trg_kps'].cpu()  # [N, 2]
    
    # No scaling needed since images are at model resolution
    kp_src = kp_src_model.numpy()
    kp_trg_gt = kp_trg_gt_model.numpy()
    
    # Prepare image tensors
    src_img_tensor = sample['src_img'].unsqueeze(0).to(device)
    trg_img_tensor = sample['trg_img'].unsqueeze(0).to(device)
    
    # Initialize predictor
    predictor = CorrespondencePredictor(cfg, device=device)
    
    # Get predictions from baseline model
    print(f"\nLoading model ({args.model_arch})...")
    model = get_model(args.model, args.model_arch)
    model = model.to(device).eval()
    
    pred_model = get_predictions(
        model, predictor, src_img_tensor, trg_img_tensor,
        kp_src_model, method=args.match_method
    )
    # No scaling needed since images are at model resolution
    kp_trg_pred = pred_model
    print(f"  Predictions computed ({args.match_method})")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Get predictions from tuned model (if specified)
    kp_trg_pred_tuned = None
    if args.model_arch_tuned:
        print(f"\nLoading tuned model ({args.model_arch_tuned})...")
        model_tuned = get_model(args.model, args.model_arch_tuned)
        model_tuned = model_tuned.to(device).eval()
        
        pred_tuned = get_predictions(
            model_tuned, predictor, src_img_tensor, trg_img_tensor,
            kp_src_model, method=args.match_method_tuned
        )
        # No scaling needed since images are at model resolution
        kp_trg_pred_tuned = pred_tuned
        print(f"  Predictions computed ({args.match_method_tuned})")
        
        del model_tuned
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Determine keypoints to visualize
    n_pts = sample['n_pts']
    if args.keypoint_indices is not None:
        keypoint_indices = [i for i in args.keypoint_indices if i < n_pts]
    elif args.max_keypoints is not None:
        keypoint_indices = list(range(min(args.max_keypoints, n_pts)))
    else:
        keypoint_indices = list(range(n_pts))
    
    print(f"\nVisualizing {len(keypoint_indices)} keypoints: {keypoint_indices}")
    
    # Generate title
    if args.no_title:
        title = None
    elif args.title:
        title = args.title
    else:
        category = sample.get('category', sample.get('pair_class', 'unknown'))
        title = f"{model_label} | {cfg.DATASET.NAME.upper()} | {category}"
    
    # Create visualization
    fig = visualize_correspondence_pair(
        src_img_pil=src_img_pil,
        trg_img_pil=trg_img_pil,
        kp_src=kp_src,
        kp_trg_gt=kp_trg_gt,
        kp_trg_pred=kp_trg_pred,
        kp_trg_pred_tuned=kp_trg_pred_tuned,
        keypoint_indices=keypoint_indices,
        show_gt=args.show_gt,
        line_width=args.line_width,
        marker_size=args.marker_size,
        alpha=args.alpha,
        figsize=tuple(args.figsize),
        title=title,
        model_label=model_label
    )
    
    # Save figure
    if args.save_path is None:
        save_dir = Path(PROJECT_ROOT) / 'asset' / 'visualizations' / f'sample_{args.sample_idx}'
        save_dir.mkdir(parents=True, exist_ok=True)
        category = sample.get('category', sample.get('pair_class', 'unknown'))
        mode = 'comparison' if args.model_arch_tuned else args.model_arch
        filename = f"{args.model}_{cfg.DATASET.NAME}_{category}_sample{args.sample_idx}_{mode}.png"
        args.save_path = save_dir / filename
    else:
        args.save_path = Path(args.save_path)
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(args.save_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {args.save_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()