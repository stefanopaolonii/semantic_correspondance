import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrespondencePredictor: 
    def __init__(self, cfg, device='cpu'):
        self.device = device
        self.cfg = cfg

    def l2_norm(self, feat, dim=-1):
        """L2 normalize features along a dimension."""
        return F.normalize(feat, p=2, dim=dim, eps=1e-6)

    def predict_argmax(self, sim_map):
        """Find maximum similarity position and return coordinates in feature map scale."""
        B, N, H, W = sim_map.shape
        sim_flat = sim_map.view(B, N, -1)
        
        indices = torch.argmax(sim_flat, dim=-1)
        
        y = (indices // W).float()
        x = (indices % W).float()
        
        return torch.stack([x, y], dim=-1)
    
    def predict_softargmax(self, sim_map):
        """Compute windowed soft-argmax with adaptive window centered on peak similarity."""
        temp = self.cfg.PREDICTOR.SOFTMAX_TEMP
        window_size = self.cfg.PREDICTOR.WINDOW_SIZE
            
        B, N, H, W = sim_map.shape
        
        masked_sim_map = self._apply_window_mask(sim_map, window_size)
        sim_flat = masked_sim_map.view(B, N, -1)
        
        prob = F.softmax(sim_flat / temp, dim=-1)
        prob = prob.view(B, N, H, W)
        
        y_vals = torch.arange(H, device=sim_map.device, dtype=torch.float32)
        x_vals = torch.arange(W, device=sim_map.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')
        
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)
        
        expected_x = torch.sum(prob * xx, dim=(2, 3))
        expected_y = torch.sum(prob * yy, dim=(2, 3))
        
        return torch.stack([expected_x, expected_y], dim=-1)

    def _apply_window_mask(self, sim_map, window_size):
        """Apply boolean mask centered on argmax, setting out-of-window values to -inf."""
        B, N, H, W = sim_map.shape
        device = sim_map.device
        
        sim_flat = sim_map.view(B, N, -1)
        max_indices = torch.argmax(sim_flat, dim=-1)
        
        max_x = (max_indices % W)
        max_y = (max_indices // W)
        
        offset_range = torch.arange(-window_size, window_size + 1, device=device)
        offset_y, offset_x = torch.meshgrid(offset_range, offset_range, indexing='ij')
        offset_x = offset_x.flatten()
        offset_y = offset_y.flatten()
        
        window_x = (max_x.unsqueeze(-1) + offset_x.view(1, 1, -1)).clamp(0, W - 1)
        window_y = (max_y.unsqueeze(-1) + offset_y.view(1, 1, -1)).clamp(0, H - 1)
        
        mask = torch.zeros((B, N, H, W), dtype=torch.bool, device=device)
        
        batch_indices = torch.arange(B, device=device)[:, None, None].expand(B, N, window_x.shape[-1])
        point_indices = torch.arange(N, device=device)[None, :, None].expand(B, N, window_x.shape[-1])
        
        mask[batch_indices, point_indices, window_y, window_x] = True
        
        masked_map = sim_map.clone()
        masked_map[~mask] = float('-inf')
        
        return masked_map
    
     
    def predict_with_image_coords(self, featmap_src, featmap_trg, kps_src_img, H_img, W_img, method='argmax', **kwargs):
        """Predict correspondences from image-scale coordinates, returning feature-scale coordinates."""
        feat_src = self.extract_feature_from_image_coords(featmap_src, kps_src_img, H_img, W_img)
        sim_map = self.compute_similarity(feat_src, featmap_trg)
        
        if method == 'argmax': 
            return self.predict_argmax(sim_map)
        elif method == 'windowed_softargmax':
            return self.predict_softargmax(sim_map)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract_feature_from_image_coords(self, featmap, kps_img, H_img, W_img):
        """Extract features from keypoints given in image-scale coordinates."""
        kps_norm = kps_img.clone()        
        kps_norm[..., 0] = ((kps_norm[..., 0]+0.5) / (W_img - 1)) * 2 - 1
        kps_norm[..., 1] = ((kps_norm[..., 1]+0.5) / (H_img - 1)) * 2 - 1
        
        kps_norm = kps_norm.unsqueeze(2)
        feat = F.grid_sample(featmap, kps_norm, align_corners=False, mode='bilinear', padding_mode='border')
        
        return feat.squeeze(-1).permute(0, 2, 1)
    
    def compute_similarity(self, feat_src, featmap_trg):
        """Compute cosine similarity map between source features and target feature map."""
        feat_src = self.l2_norm(feat_src, dim=-1)     
        featmap_trg = self.l2_norm(featmap_trg, dim=1)
        
        B, C, H, W = featmap_trg.shape
        featmap_flat = featmap_trg.view(B, C, -1)
        
        sim = torch.bmm(feat_src, featmap_flat)
        sim_map = sim.view(B, feat_src.shape[1], H, W)
        
        return sim_map