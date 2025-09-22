import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class CrossAttention3D(nn.Module):
    """
    Cross-attention that takes 3D image features as queries and text tokens as keys/values.
    Query:  B x Cq x D x H x W  -> flattened to B x N x Cq
    Text:   B x L x Ct (variable L, padded) -> projected to K,V with dim Cq
    """

    def __init__(self, c_img: int, c_txt: int, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.q_proj = nn.Conv3d(c_img, c_img, kernel_size=1, bias=False)
        self.k_proj = nn.Linear(c_txt, c_img, bias=False)
        self.v_proj = nn.Linear(c_txt, c_img, bias=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=c_img, num_heads=nhead, batch_first=True, dropout=dropout
        )
        self.out = nn.Conv3d(c_img, c_img, kernel_size=1)
        self.norm = nn.InstanceNorm3d(c_img, affine=True)
        self.act = nn.SiLU()
        # will be filled at forward if return_attn=True
        self._last_attn: Optional[torch.Tensor] = None

        self.txt_ln = nn.LayerNorm(c_txt)
        self.q_ln = nn.InstanceNorm3d(c_img, affine=True)

    def forward(
        self,
        x_img: torch.Tensor,  # (B, C, D, H, W)
        x_txt: torch.Tensor,  # (B, L, Ct)
        txt_mask: Optional[torch.Tensor] = None,  # (B, L) -> True for PAD positions
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, D, H, W = x_img.shape
        N = D * H * W

        # q = self.q_proj(x_img).flatten(2).transpose(1, 2)  # (B, N, C)
        # k = self.k_proj(x_txt)  # (B, L, C)
        # v = self.v_proj(x_txt)  # (B, L, C)

        x_txt = torch.nan_to_num(x_txt, nan=0.0, posinf=1e4, neginf=-1e4)
        x_txt = self.txt_ln(x_txt)  # (B, L, Ct) - stable token scale
        x_img = self.q_ln(x_img)  # (B, C, D, H, W) - tame query scale

        with torch.autocast(device_type=str(x_img.device).split(":")[0], enabled=False):
            q = self.q_proj(x_img.float()).flatten(2).transpose(1, 2)  # (B,N,C)
            k = self.k_proj(x_txt.float())  # (B,L,C)
            v = self.v_proj(x_txt.float())  # (B,L,C)
            out, attn = self.attn(
                q,
                k,
                v,
                key_padding_mask=txt_mask,  # True == PAD (ignored)
                need_weights=return_attn,
                average_attn_weights=False,
            )  # out: (B, N, C); attn: (B, N, L)
        out = out.to(x_img.dtype).transpose(1, 2).view(B, C, D, H, W)
        out = self.out(out)
        out = self.act(self.norm(out + x_img))  # residual

        if return_attn:
            # average across heads was already done by MHA with batch_first; attn is (B, N, L)
            self._last_attn = attn
            return out, attn
        return out, None

    def last_attention(self) -> Optional[torch.Tensor]:
        return self._last_attn
