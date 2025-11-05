import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, n_layers=4, n_classes=10, max_len=1200):
        super().__init__()
        self.embed     = nn.Linear(3, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, batch_first=True)
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.decoder   = nn.Linear(embed_dim, 2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, x, pad_mask):
        B, T, _ = x.shape
        x = self.embed(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos)
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        recon = self.decoder(encoded)
        cls   = self.classifier(encoded.permute(0,2,1))
        return recon, cls