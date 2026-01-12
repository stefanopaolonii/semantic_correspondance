import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
from .predictor import CorrespondencePredictor
from .pck_evaluator import PCKEvaluator


def rescale_coords(pred_feat, H_img, W_img, H_feat, W_feat):
    """Rescale coordinates from feature space to image space."""
    pred_img = pred_feat.clone()
    
    scale_x = W_img / W_feat
    scale_y = H_img / H_feat
    pred_img[..., 0] = (pred_img[..., 0] + 0.5) * scale_x
    pred_img[..., 1] = (pred_img[..., 1] + 0.5) * scale_y
    
    return pred_img.cpu()

class CorrespondenceEvaluator:
    def __init__(self, cfg_base, device):
        self.cfg_img = copy.deepcopy(cfg_base)
        self.cfg_img.EVALUATOR.BY = 'image'
        self.evaluator_img = PCKEvaluator(self.cfg_img)

        self.cfg_pt = copy.deepcopy(cfg_base)
        self.cfg_pt.EVALUATOR.BY = 'point'
        self.evaluator_pt = PCKEvaluator(self.cfg_pt)

        self.device = device
        self.alphas = cfg_base.EVALUATOR.ALPHA
        self.predictor = CorrespondencePredictor(cfg=cfg_base, device=self.device)

    def evaluate(self, model, loader, match_method='argmax', max_batch=None):
        model.eval()
        self.evaluator_img.method_options = [match_method]
        self.evaluator_img.clear_result()

        self.evaluator_pt.method_options = [match_method]
        self.evaluator_pt.clear_result()

        for alpha in self.alphas:
            key = f'{match_method}_pck{alpha}'
            if key not in self.evaluator_img.result:
                self.evaluator_img.result[key] = {'all': []}
            if key not in self.evaluator_pt.result:
                self.evaluator_pt.result[key] = {'all': []}

        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader), desc="Evaluating", total=len(loader)):
                if max_batch is not None and i >= max_batch:
                    break
                self._process_batch(model, batch, match_method)
                                        
        return self.evaluator_img, self.evaluator_pt

    def _process_batch(self, model, batch, match_method):
        """Process a batch of image pairs to compute semantic correspondences."""
        src_img = batch['src_img'].to(self.device)
        trg_img = batch['trg_img'].to(self.device)
        
        feat_src = model(src_img)
        feat_trg = model(trg_img)
        
        B_size = src_img.shape[0]
        H_img, W_img = src_img.shape[2:]
        H_feat, W_feat = feat_src.shape[2:]

        pred_kps_feat = self.predictor.predict_with_image_coords(
            featmap_src=feat_src, 
            featmap_trg=feat_trg, 
            kps_src_img=batch['src_kps'].to(self.device),
            H_img=H_img,
            W_img=W_img,
            method=match_method
        )
        
        pred_kps_img = rescale_coords(
            pred_kps_feat, H_img, W_img, H_feat, W_feat
        )

        self.evaluator_img.calculate_pck(
            batch['trg_kps'], 
            pred_kps_img,
            batch['n_pts'],
            batch['category'],
            batch['pckthres'], 
            match_method
        )
        self.evaluator_pt.calculate_pck(
            batch['trg_kps'],
            pred_kps_img,
            batch['n_pts'],
            batch['category'],
            batch['pckthres'],
            match_method
        )