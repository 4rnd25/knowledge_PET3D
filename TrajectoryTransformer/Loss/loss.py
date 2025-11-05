"""
Created on Mar 05 2025 13:52

@author: ISAC - pettirsch
"""

import torch
import torch.nn as nn

def compute_loss(
    reconstructed,    # (B, T, 2)
    target,           # (B, T, 3)
    cls_logits,       # (B, n_classes)
    cls_targets,      # (B,)
    pad_mask,         # (B, T)
    input_mask,       # (B, T)
    cls_weight=0.3,
    ret_masked_error=False
):
    err = (reconstructed - target[:, :, 1:3]) ** 2     # (B, T, 2)
    pad_exp =  pad_mask.unsqueeze(-1).expand_as(err)  # True where padded
    inp_exp = (~input_mask.unsqueeze(-1)).expand_as(err)  # True where _un_masked_
    ignore  = pad_exp #| inp_exp                        # ignore padding OR un-masked slots

    valid_err  = err[~ignore]                         # now only masked & unpadded
    num_masked = (~pad_exp & inp_exp).sum().item()  # pads=False, masked=True
    print(f"[DEBUG] masked slots this batch = {num_masked}")
    recon_loss = valid_err.mean()

    rmse_m = torch.sqrt(recon_loss)

    cls_loss   = nn.CrossEntropyLoss()(cls_logits, cls_targets)

    # Weighted sum
    total_loss = recon_loss * 100 + cls_weight * cls_loss

    # debug printing
    print(
        f"recon_loss *100 = {100 * recon_loss.item():.4f}, "
        f"cls_loss *{cls_weight} = {cls_weight * cls_loss.item():.4f}"
    )

    if ret_masked_error:
        return total_loss, cls_loss, recon_loss, rmse_m
    else:
        return total_loss, cls_loss, recon_loss