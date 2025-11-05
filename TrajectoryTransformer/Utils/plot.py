"""
Created on Mar 05 2025 13:54

@author: ISAC - pettirsch
"""


import numpy as np
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt

def plot_one_batch(dataloader, model, epoch, device, output_dir):
    """
    Plots a batch of trajectories with reconstruction and masks,
    and annotates each subplot with the true (T) and predicted (P) class indices.
    """
    # grab one batch
    masked_batch, full_batch, pad_mask, input_mask, cls_targets = next(iter(dataloader))

    # run model
    reconstructed, cls_logits = model(masked_batch.to(device), pad_mask.to(device))

    # compute predicted class indices
    cls_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()
    cls_true = cls_targets.cpu().numpy()

    # move everything to numpy
    full_np   = full_batch.cpu().numpy()      # (B, T, 3)
    recon_np  = reconstructed.cpu().detach().numpy()  # (B, T, 2)
    pad_np    = pad_mask.cpu().numpy()        # (B, T)
    input_np  = input_mask.cpu().numpy()      # (B, T)

    plt.figure(figsize=(12, 8))
    num_samples = min(len(full_np), 8)

    for i in range(num_samples):
        plt.subplot(2, 4, i + 1)

        # integrate full ground-truth deltas to absolute coords
        full_deltas = full_np[i, :, 1:3]   # deltaX, deltaY
        full_xy     = np.cumsum(full_deltas, axis=0)

        # integrate reconstructed deltas at non-padded steps
        valid      = ~pad_np[i]
        rec_deltas = recon_np[i, valid]
        rec_xy     = np.cumsum(rec_deltas, axis=0)

        # plot GT and reconstruction
        plt.plot(full_xy[:, 0], full_xy[:, 1],
                 '-', linewidth=1.5, label='GT (full)')
        plt.plot(rec_xy[:, 0], rec_xy[:, 1],
                 '--', linewidth=1.5, label='Reconstructed')

        # highlight masked inputs
        masked_xy = full_xy[input_np[i]]
        plt.scatter(masked_xy[:, 0], masked_xy[:, 1],
                    color='k', s=30, label='Input-masked')

        # annotate with numeric classes
        tgt = cls_true[i]
        pred = cls_pred[i]
        plt.title(f"Sample {i+1}\nT: {tgt}    P: {pred}",
                  fontsize='small')

        plt.legend(loc='upper left', fontsize='x-small')
        plt.axis('equal')

    plt.suptitle(f"Full-Trajectory Reconstruction — Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()