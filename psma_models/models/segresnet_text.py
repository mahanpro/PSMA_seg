import torch
import torch.nn as nn
from typing import Optional, Tuple
from monai.networks.blocks.convolutions import ResidualUnit
from .cross_attn import CrossAttention3D


def _stack_ru(n, c_in, c_out, stride=1):
    """
    Stack n ResidualUnits. First block may change channels/stride, rest keep same.
    """
    blocks = []
    for i in range(n):
        s = stride if i == 0 else 1
        cin = c_in if i == 0 else c_out
        blocks.append(
            ResidualUnit(
                spatial_dims=3,
                in_channels=cin,
                out_channels=c_out,
                strides=s,
                kernel_size=3,
                subunits=2,
                norm="INSTANCE",
                act="RELU",
                dropout=0.0,
            )
        )
    return nn.Sequential(*blocks)


class _UpBlockRU(nn.Module):
    """
    Upsample (transpose conv) -> concat skip -> ResidualUnit to fuse.
    """

    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.fuse = ResidualUnit(
            spatial_dims=3,
            in_channels=c_out + c_skip,
            out_channels=c_out,
            strides=1,
            kernel_size=3,
            subunits=2,
            norm="INSTANCE",
            act="RELU",
            dropout=0.0,
        )

    def forward(self, x, skip):
        x = self.up(x)
        # pad if shape off by 1-2 voxels due to odd divisions
        dd = skip.size(2) - x.size(2)
        hh = skip.size(3) - x.size(3)
        ww = skip.size(4) - x.size(4)
        if dd != 0 or hh != 0 or ww != 0:
            x = nn.functional.pad(x, [0, max(0, ww), 0, max(0, hh), 0, max(0, dd)])
        x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class SegResNetText(nn.Module):
    """
    SegResNet-like encoder-decoder with ResidualUnits (MONAI style) and optional
    text fusion via CrossAttention3D at each decoder level.

    - in_ch=2, out_ch=2 by default (CT+PET -> background/lesion)
    - init_filters controls base width (matches MONAI SegResNet)
    - blocks_down: list[int] (len=4) of RU stacks per down level
    - blocks_up:   list[int] (len=3) of RU stacks per up level (after upsample+fuse)

    When use_text=True, we insert CrossAttention3D after each up block (high→low).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        init_filters: int = 16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        txt_dim: int = 768,
        use_text: bool = False,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        fuse_fullres: bool = False,
    ):
        super().__init__()
        C = init_filters

        # Encoder (4 levels → /16)
        self.stem = ResidualUnit(
            3,
            in_channels,
            C,
            strides=1,
            kernel_size=3,
            subunits=2,
            norm="INSTANCE",
            act="RELU",
        )
        self.enc1 = _stack_ru(blocks_down[0], C, C, stride=1)  # C
        self.enc2 = _stack_ru(blocks_down[1], C, C * 2, stride=2)  # 2C
        self.enc3 = _stack_ru(blocks_down[2], C * 2, C * 4, stride=2)  # 4C
        self.enc4 = _stack_ru(blocks_down[3], C * 4, C * 8, stride=2)  # 8C

        # Bottleneck
        self.bottom = _stack_ru(1, C * 8, C * 16, stride=2)  # 16C

        # Decoder (x4 → back to /1)
        self.up4 = _UpBlockRU(C * 16, C * 8, C * 8)
        self.dec4 = _stack_ru(blocks_up[0], C * 8, C * 8, stride=1)

        self.up3 = _UpBlockRU(C * 8, C * 4, C * 4)
        self.dec3 = _stack_ru(blocks_up[1], C * 4, C * 4, stride=1)

        self.up2 = _UpBlockRU(C * 4, C * 2, C * 2)
        self.dec2 = _stack_ru(blocks_up[2], C * 2, C * 2, stride=1)

        self.up1 = _UpBlockRU(C * 2, C, C)
        self.dec1 = _stack_ru(1, C, C, stride=1)

        # Text fusion at decoder feature resolutions
        self.use_text = use_text
        if use_text:
            self.fuse4 = CrossAttention3D(
                C * 8, txt_dim, nhead=n_heads, dropout=attn_dropout
            )
            self.fuse3 = CrossAttention3D(
                C * 4, txt_dim, nhead=n_heads, dropout=attn_dropout
            )
            self.fuse2 = CrossAttention3D(
                C * 2, txt_dim, nhead=n_heads, dropout=attn_dropout
            )
            self.fuse1 = (
                CrossAttention3D(C, txt_dim, nhead=n_heads, dropout=attn_dropout)
                if fuse_fullres
                else None
            )
        else:
            self.fuse4 = self.fuse3 = self.fuse2 = self.fuse1 = None

        self.head = nn.Conv3d(C, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,  # (B, 2, D, H, W)
        txt_tokens: Optional[torch.Tensor] = None,  # (B, L, Ct)
        txt_mask: Optional[torch.Tensor] = None,  # (B, L) True==PAD
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Encoder
        s0 = self.stem(x)  # C
        s1 = self.enc1(s0)  # C
        s2 = self.enc2(s1)  # 2C
        s3 = self.enc3(s2)  # 4C
        s4 = self.enc4(s3)  # 8C
        b = self.bottom(s4)  # 16C

        # Decoder + optional cross-attn
        attn_out = None

        d4 = self.dec4(self.up4(b, s4))
        if self.use_text and self.fuse4 is not None:
            d4, a4 = self.fuse4(d4, txt_tokens, txt_mask, return_attn=return_attn)
            if return_attn:
                attn_out = a4

        d3 = self.dec3(self.up3(d4, s3))
        if self.use_text and self.fuse3 is not None:
            d3, a3 = self.fuse3(d3, txt_tokens, txt_mask, return_attn=return_attn)
            if return_attn:
                attn_out = a3

        d2 = self.dec2(self.up2(d3, s2))
        if self.use_text and self.fuse2 is not None:
            d2, a2 = self.fuse2(d2, txt_tokens, txt_mask, return_attn=return_attn)
            if return_attn:
                attn_out = a2

        d1 = self.dec1(self.up1(d2, s1))
        if self.use_text and self.fuse1 is not None:
            d1, attn_out = self.fuse1(d1, txt_tokens, txt_mask, return_attn=return_attn)

        logits = self.head(d1)
        return logits, attn_out
