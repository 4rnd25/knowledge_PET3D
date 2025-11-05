"""
Created on May 16 2025 07:57

@author: ISAC - pettirsch
"""

import torch

from TrajectoryTransformer.Model.model import TrajectoryTransformer
from TrajectoryTransformer.Utils.dataloader import to_ego_coordinates, normalize_time, compute_displacements


class LatentSpaceCalculator:
    def __init__(self, config, verbose = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TrajectoryTransformer(embed_dim=config['embeded_dim'], n_heads=config['n_heads'],
                                           n_layers=config['n_layers'], n_classes=config['n_classes'],
                                           max_len=config['max_len'])
        self.model.load_state_dict(torch.load(config['model_path']))
        self.model.eval()
        self.model.to(self.device)

        # Store maximum supported length from positional embedding
        self.max_positions = self.model.pos_embed.num_embeddings

        self.verbose = verbose


    def embed_single_trajectory(self, df):
        """
        df: one vehicle's full DataFrame
        returns:  (d,) numpy array
        """
        # 1) same preprocessing as in the Dataset
        traj = to_ego_coordinates(df.copy())
        traj = normalize_time(traj)
        traj = compute_displacements(traj)

        # 2) build the (1, T, 3) tensor and pad_mask
        data = torch.tensor(
            traj[['deltaT', 'deltaX', 'deltaY']].values,
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, T, 3)
        T = data.size(1)
        pad_mask = torch.zeros((1, T), dtype=torch.bool, device=self.device)

        # 3) ensure T does not exceed max_positions
        if T > self.max_positions:
            if self.verbose:
                print(f"[LatentSpaceCalculator] Truncating sequence length from {T} to {self.max_positions}")
            # truncate data and mask to max_positions
            data = data[:, :self.max_positions, :]
            pad_mask = pad_mask[:, :self.max_positions]
            T = self.max_positions

        # 3) run through encoder manually to get (1, T, D)
        with torch.no_grad():
            x = self.model.embed(data)  # (1, T, D)
            pos = torch.arange(T, device=self.device).unsqueeze(0).expand(1, T)
            x = x + self.model.pos_embed(pos)
            encoded = self.model.encoder(x, src_key_padding_mask=pad_mask)  # (1, T, D)

        # 4) mean‐pool ignoring padding (here none)
        #    if you ever pad shorter sequences, you can mask similarly
        z = encoded.mean(dim=1)  # (1, D)
        return z.squeeze(0).cpu().numpy()  # (D,)