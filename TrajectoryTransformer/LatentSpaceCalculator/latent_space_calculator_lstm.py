"""
Created on Oct 8 2025 10:57

@author: ISAC - pettirsch
"""

import torch
import torch.nn as nn

from TrajectoryTransformer.Model.trajectorylstm import TrajectoryLSTM
from TrajectoryTransformer.Utils.dataloader import to_ego_coordinates, normalize_time, compute_displacements


class LatentSpaceCalculatorLSTM:
    def __init__(self, config, verbose = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = True

        self.model = TrajectoryLSTM(n_classes=config['n_classes']).to(self.device)
        self.model.load_state_dict(torch.load(config['lstm_model_path']))
        self.max_len = int(config.get('max_len', 1200))
        # missing, unexpected = self.model.load_state_dict(state, strict=False)
        # if self.verbose:
        #     if missing:    print(f"[LSTM] Missing keys: {missing}")
        #     if unexpected: print(f"[LSTM] Unexpected keys: {unexpected}")

        self.model.eval()

        # cache output feature dim for pooling sanity checks
        self.out_dim = self.model.decoder.in_features  # equals hidden_dim*(2 if bidir else 1)

    def _to_features(self, x, pad_mask):
        """
        x: (B, T, 3) float32
        pad_mask: (B, T) bool, True = padding
        returns: (B, T, F) LSTM time features (unpadded)
        """
        lengths = (~pad_mask).sum(dim=1).cpu()
        x = self.model.embed(x)  # (B, T, embed_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.model.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (B, T_max, F)

        return out  # time features prior to decoder/classifier

    def embed_single_trajectory(self, df):
        """
        df: one vehicle's full DataFrame
        returns: (F,) numpy array latent (mean-pooled LSTM features)
        """
        # 1) same preprocessing as Dataset
        traj = to_ego_coordinates(df.copy())
        traj = normalize_time(traj)
        traj = compute_displacements(traj)

        seq = traj[['deltaT', 'deltaX', 'deltaY']].values
        # optional truncation if you trained with a max T
        if len(seq) > self.max_len:
            if self.verbose:
                print(f"[LatentSpaceCalculatorLSTM] Truncating sequence from {len(seq)} to {self.max_len}")
            seq = seq[:self.max_len]

        data = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, 3)
        T = data.size(1)
        pad_mask = torch.zeros((1, T), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            feats = self._to_features(data, pad_mask)  # (1, T, F)
            # mean-pool over valid timesteps (all valid here)
            z = feats.mean(dim=1)  # (1, F)

        z = z.squeeze(0).cpu().numpy()
        return z