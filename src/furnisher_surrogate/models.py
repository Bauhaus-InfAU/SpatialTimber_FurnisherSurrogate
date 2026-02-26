"""CNN architecture for room score prediction.

Input: 3x64x64 rasterized room image + room_type index + tabular scalars.
Output: predicted score (0-100).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoomCNN(nn.Module):
    """CNN with tabular side-inputs for room score prediction.

    Image branch: 4 conv blocks → GlobalAvgPool → optional bottleneck
    Tabular branch: Embedding + scalars → optional FC
    FC head: cat(image_out + tabular_out) → FC layers → score

    v1 defaults: no bottleneck, raw concat (256+16+3=275)
    v2 tuning: image_bottleneck=64, tabular_hidden=32 → balanced (64+32=96)
    v3 tuning: + n_tabular=5, tabular_skip=True → tabular shortcut to output
    """

    def __init__(
        self,
        n_room_types: int = 9,
        embed_dim: int = 16,
        n_tabular: int = 3,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        fc_hidden: int = 128,
        dropout: float = 0.3,
        image_bottleneck: int | None = None,
        tabular_hidden: int | None = None,
        tabular_skip: bool = False,
    ):
        super().__init__()
        self.n_room_types = n_room_types
        self.embed_dim = embed_dim
        self.n_tabular = n_tabular
        self.channels = channels
        self.fc_hidden = fc_hidden
        self.dropout_rate = dropout

        # Image branch: 4 conv blocks
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Optional bottleneck: compress image features before concat
        image_out_dim = channels[-1]
        if image_bottleneck is not None:
            self.image_fc = nn.Sequential(
                nn.Linear(channels[-1], image_bottleneck),
                nn.ReLU(inplace=True),
            )
            image_out_dim = image_bottleneck
        else:
            self.image_fc = None

        # Tabular branch
        self.room_embed = nn.Embedding(n_room_types, embed_dim)
        tabular_raw_dim = embed_dim + n_tabular

        # Optional tabular FC: strengthen tabular signal before merge
        if tabular_hidden is not None:
            self.tabular_fc = nn.Sequential(
                nn.Linear(tabular_raw_dim, tabular_hidden),
                nn.ReLU(inplace=True),
            )
            tabular_out_dim = tabular_hidden
        else:
            self.tabular_fc = None
            tabular_out_dim = tabular_raw_dim

        # FC head
        fc_in = image_out_dim + tabular_out_dim
        self.head = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        # Optional skip connection: direct tabular → output pathway
        self.tabular_skip = tabular_skip
        if tabular_skip:
            self.skip_head = nn.Sequential(
                nn.Linear(tabular_raw_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

    def forward(
        self,
        image: torch.Tensor,
        room_type_idx: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: (B, 3, 64, 64) float32, pixel values in [0, 1]
            room_type_idx: (B,) int64, index into ROOM_TYPES
            tabular: (B, 3) float32, [area_standardized, door_rel_x, door_rel_y]

        Returns:
            (B, 1) predicted score
        """
        # Image branch
        x = self.conv(image)       # (B, 256, 4, 4)
        x = self.gap(x)            # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        if self.image_fc is not None:
            x = self.image_fc(x)   # (B, bottleneck)

        # Tabular branch
        emb = self.room_embed(room_type_idx)  # (B, 16)
        tab_raw = torch.cat([emb, tabular], dim=1)  # (B, embed+n_tabular)
        tab = self.tabular_fc(tab_raw) if self.tabular_fc is not None else tab_raw

        # Concat and predict
        combined = torch.cat([x, tab], dim=1)
        out = self.head(combined)  # (B, 1)

        # Optional skip: add direct tabular prediction
        if self.tabular_skip:
            out = out + self.skip_head(tab_raw)

        return out
