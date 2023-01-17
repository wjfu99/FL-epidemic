import torch
import math
import numpy as np
def usr_emb_clip(usr_emb_upd, max_norm, norm_type=2.0):
    total_norm = torch.norm(usr_emb_upd, norm_type, dim=1)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    clip_coef_clamped = clip_coef_clamped.unsqueeze(1)
    # usr_emb_upd *= clip_coef_clamped
    usr_emb_upd = torch.mul(usr_emb_upd, clip_coef_clamped)
    return usr_emb_upd

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fl_dp(model, optimizer, **model_args):
    model_state = model.state_dict()
    scale = get_lr(optimizer) * model_args['fl_clip'] * math.sqrt(2 * math.log(1.25 / model_args['fl_delt'], math.e)) / \
            model_args['fl_eps']
    for para_name in model_state:
        if 'usr_emb' not in para_name:
            model_state[para_name] += scale * torch.randn_like(model_state[para_name])
    model.load_state_dict(model_state)