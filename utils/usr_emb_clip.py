import torch


def usr_emb_clip(usr_emb_upd, max_norm, norm_type=2.0):
    total_norm = torch.norm(usr_emb_upd, norm_type, dim=1)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    clip_coef_clamped = clip_coef_clamped.unsqueeze(1)
    # usr_emb_upd *= clip_coef_clamped
    usr_emb_upd = torch.mul(usr_emb_upd, clip_coef_clamped)
    return usr_emb_upd
