import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, n_layers=2, n_classes=10, bidirectional=True):
        super().__init__()
        self.embed = nn.Linear(3, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.decoder = nn.Linear(lstm_out_dim, 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(lstm_out_dim, n_classes)
        )

    def forward(self, x, pad_mask):
        # pad_mask is (B, T), where True = padding
        lengths = (~pad_mask).sum(dim=1).cpu()
        x = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        recon = self.decoder(out)
        cls = self.classifier(out.permute(0, 2, 1))
        return recon, cls
