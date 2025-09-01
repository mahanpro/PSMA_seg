import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .cross_attn import CrossAttention3D


def conv_block(c_in, c_out, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, k, s, p, bias=False),
        nn.InstanceNorm3d(c_out, affine=True),
        nn.SiLU(),
        nn.Conv3d(c_out, c_out, k, s, p, bias=False),
        nn.InstanceNorm3d(c_out, affine=True),
        nn.SiLU(),
    )


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.conv = conv_block(c_out + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diffD = skip.size(2) - x.size(2)
        diffH = skip.size(3) - x.size(3)
        diffW = skip.size(4) - x.size(4)
        x = nn.functional.pad(x, [0, diffW, 0, diffH, 0, diffD])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3DText(nn.Module):
    """
    UNet-like 3D model with optional cross-attention at each decoder stage.
    - in_ch=2 (CT, PET), out_ch=2 (background/foreground)
    - text tokens (B, L, Ct) are consumed once and shared across decoder levels via level-specific CrossAttention3D projections.
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 2,
        base_ch: int = 32,
        txt_dim: int = 768,
        use_text: bool = False,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_text = use_text

        C = base_ch
        self.enc1 = conv_block(in_ch, C)
        self.down1 = nn.Conv3d(C, C * 2, 2, 2)
        self.enc2 = conv_block(C * 2, C * 2)
        self.down2 = nn.Conv3d(C * 2, C * 4, 2, 2)
        self.enc3 = conv_block(C * 4, C * 4)
        self.down3 = nn.Conv3d(C * 4, C * 8, 2, 2)
        self.enc4 = conv_block(C * 8, C * 8)
        self.down4 = nn.Conv3d(C * 8, C * 16, 2, 2)

        self.bottleneck = conv_block(C * 16, C * 16)

        self.up4 = UpBlock(C * 16, C * 8, C * 8)
        self.up3 = UpBlock(C * 8, C * 4, C * 4)
        self.up2 = UpBlock(C * 4, C * 2, C * 2)
        self.up1 = UpBlock(C * 2, C, C)

        # optional text fusion blocks (one per decoder level)
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
            self.fuse1 = CrossAttention3D(
                C, txt_dim, nhead=n_heads, dropout=attn_dropout
            )
        else:
            self.fuse4 = self.fuse3 = self.fuse2 = self.fuse1 = None

        self.head = nn.Conv3d(C, out_ch, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,  # (B, 2, D, H, W)
        txt_tokens: Optional[torch.Tensor] = None,  # (B, L, Ct)
        txt_mask: Optional[torch.Tensor] = None,  # (B, L) True==PAD
        return_attn: bool = False,
    ):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        s4 = self.enc4(self.down3(s3))
        b = self.bottleneck(self.down4(s4))

        # Decoder + optional fusion
        d4 = self.up4(b, s4)
        if self.use_text and self.fuse4 is not None:
            d4, _ = self.fuse4(d4, txt_tokens, txt_mask, return_attn=return_attn)

        d3 = self.up3(d4, s3)
        if self.use_text and self.fuse3 is not None:
            d3, _ = self.fuse3(d3, txt_tokens, txt_mask, return_attn=return_attn)

        d2 = self.up2(d3, s2)
        if self.use_text and self.fuse2 is not None:
            d2, _ = self.fuse2(d2, txt_tokens, txt_mask, return_attn=return_attn)

        d1 = self.up1(d2, s1)
        if self.use_text and self.fuse1 is not None:
            d1, attn_l1 = self.fuse1(d1, txt_tokens, txt_mask, return_attn=return_attn)
        else:
            attn_l1 = None

        logits = self.head(d1)
        # For heatmaps, the last attention (at the highest resolution) is most interpretable
        return logits, attn_l1
