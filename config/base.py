from yacs.config import CfgNode as CN

_CN = CN()

_CN.PROJECT_ROOT = "."

# Dataset configuration
_CN.DATASET = CN()
_CN.DATASET.NAME = 'spair'
_CN.DATASET.ROOT = 'asset/dataset/'

# DINOv2 configuration
_CN.DINOV2 = CN()
_CN.DINOV2.IMG_SIZE = 518 
_CN.DINOV2.MEAN = [0.485, 0.456, 0.406]
_CN.DINOV2.STD = [0.229, 0.224, 0.225]

# DINOv3 configuration
_CN.DINOV3 = CN()
_CN.DINOV3.IMG_SIZE = 512  
_CN.DINOV3.MEAN = [0.485, 0.456, 0.406]
_CN.DINOV3.STD = [0.229, 0.224, 0.225]

# SAM configuration
_CN.SAM = CN()
_CN.SAM.IMG_SIZE = 1024  
_CN.SAM.MEAN = [0.485, 0.456, 0.406]
_CN.SAM.STD = [0.229, 0.224, 0.225]

# CLIP configuration
_CN.CLIP = CN()
_CN.CLIP.IMG_SIZE = 224 
_CN.CLIP.MEAN = [0.48145466, 0.4578275, 0.40821073]
_CN.CLIP.STD = [0.26862954, 0.26130258, 0.27577711]

# Evaluator configuration
_CN.EVALUATOR = CN()
_CN.EVALUATOR.ALPHA = (0.05, 0.1, 0.2)

# Predictor configuration
_CN.PREDICTOR = CN()
_CN.PREDICTOR.SOFTMAX_TEMP = 0.1
_CN.PREDICTOR.WINDOW_SIZE = 3      

# Training configuration
_CN.TRAIN = CN()
_CN.TRAIN.EPOCHS = 3
_CN.TRAIN.LR = 1e-5
_CN.TRAIN.WEIGHT_DECAY = 0.05
_CN.TRAIN.LR_MILESTONES = [1, 2]
_CN.TRAIN.LR_GAMMA = 0.5
_CN.TRAIN.CHECKPOINT_DIR = 'asset/weights/'

# Loss configuration
_CN.LOSS = CN()
_CN.LOSS.KERNEL_SIZE = 7
_CN.LOSS.SOFTMAX_TEMP = 0.03

BASE_CONFIG = _CN.clone()
