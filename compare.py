from omegaconf import OmegaConf
from dinov2.models import build_model_from_cfg
import torch
from icecream import ic

#default = OmegaConf.load('/home/nikola.jovisic.ivi/nj/dinov2/dinov2/configs/ssl_default_config.yaml')
#mammo = OmegaConf.load('/home/nikola.jovisic.ivi/nj/dinov2/dinov2/configs/train/mammo-config.yaml')
#cfg = OmegaConf.merge(default, mammo)

#student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
#pretrain = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

weights = torch.load('dinov2_vitb14_reg4_pretrain.pth')
test_weights = torch.load('test.pth')

ic(weights['register_tokens'])
ic(test_weights.keys())
#ic(weights.keys())
