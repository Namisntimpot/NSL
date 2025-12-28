# MIT License

# Copyright (c) 2022 Karl Stelzner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file comes from https://github.com/stelzner/srt
# Modified: use xformers to accelerate. no effect.

import torch
from einops import rearrange
from torch import nn

# try:
#     from xformers.ops import unbind
#     from xformers.components.attention import ScaledDotProduct
#     XFORMERS_AVAILABLE = True
# except ImportError:
#     XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, selfatt=True, kv_dim=None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.dim_head = dim_head
        
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        # if XFORMERS_AVAILABLE:
        #     self.attention = ScaledDotProduct()

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, z=None):
        '''
        x, z: batch n_samples c
        '''
        # if not XFORMERS_AVAILABLE:
        if z is None:
            # to_qkv, (b,n,c) -> (b,n,3*nhead*dim_per_head)
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)
        # qkv: b, n_samples, n_heads*dim_per_head
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (b h 1 n)  # q的n为1.

        attn = self.attend(dots)  # (b h n_sample n_sample)

        out = torch.matmul(attn, v)  # (b h n_sample n_sample) * (b h n_sample d)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
        
        # else:
        #     if z is None:
        #         B,N,C = x.shape
        #         qkv = self.to_qkv(x).reshape(B,N,3,self.heads,self.dim_head)
        #         q,k,v = unbind(qkv, 2)
        #     else:
        #         Bz, Nz, Cz = z.shape
        #         kv = self.to_kv(z).reshape(Bz, Nz, 2, self.heads, self.dim_head)
        #         k, v = unbind(kv, 2)
        #         Bx, Nx, Cx = x.shape
        #         q = self.to_q(x).reshape(Bx, Nx, self.heads, self.dim_head)
        #         B,N = max(Bx, Bz), max(Nx, Nz)
        #     out = self.attention(q, k, v)
        #     out = out.reshape(B, N, -1)[:,:1]
        #     return self.to_out(out)