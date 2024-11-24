# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from copy import deepcopy
from typing import List, Optional, Tuple
import math
from functools import partial
from sympy import flatten

import torch
import torch.nn as nn
from torch import Tensor, pixel_shuffle

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules import GELU

import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# from vit.vision_transformer import Conv3DCrossAttentionBlock

from .utils import trunc_normal_

from pdb import set_trace as st
# import apex
# from apex.normalization import FusedRMSNorm as RMSNorm
try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except:
    from dit.norm import RMSNorm

from torch.nn import LayerNorm

try:
    import xformers
    import xformers.ops
    from xformers.ops import memory_efficient_attention, unbind, fmha
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp, MemoryEfficientAttentionCutlassOp
    # from xformers.ops import RMSNorm

    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

from packaging import version
assert version.parse(torch.__version__) >= version.parse("2.0.0")
SDP_IS_AVAILABLE = True
# from torch.backends.cuda import SDPBackend, sdp_kernel
from torch.nn.attention import sdpa_kernel, SDPBackend


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0., 
                 enable_rmsnorm=False,
                 qk_norm=False,
                 no_flash_op=False,
                 enable_rope=False,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # https://github.com/huggingface/pytorch-image-models/blob/5dce71010174ad6599653da4e8ba37fd5f9fa572/timm/models/vision_transformer.py#L79C1-L80C78

        self.enable_rope = enable_rope
        # st()

        if enable_rope:
            self.q_norm = RMSNorm(dim, elementwise_affine=True) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(dim, elementwise_affine=True) if qk_norm else nn.Identity()
        else:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True) if qk_norm else nn.Identity()

        # if qk_norm:
        #     self.q_norm = LayerNorm(dim, eps=1e-5)
        #     self.k_norm = LayerNorm(dim, eps=1e-5)
        self.qk_norm = qk_norm
        self.no_flash_op = no_flash_op
        self.attn_mode = "torch"
         
        self.backend = SDPBackend.FLASH_ATTENTION # FA implemented by torch.

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x):
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
        #                           C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # return x, attn

        # https://github.com/Stability-AI/generative-models/blob/863665548f95ff827273948766a3f732ab01bc49/sgm/modules/attention.py#L179
        B, L, C = x.shape

        qkv = self.qkv(x)
        if self.attn_mode == "torch":
            qkv = rearrange(
                qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
            ).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            q, k = self.q_norm(q), self.k_norm(k)

            with sdpa_kernel([self.backend]): # new signature

                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                 
            del q, k, v
            x = rearrange(x, "B H L D -> B L (H D)")

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MemEffAttention(Attention):

    def forward(self, x: Tensor, attn_bias=None, freqs_cis=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x)

        dtype = qkv.dtype

        if self.enable_rope:
            assert freqs_cis is not None

            qkv = qkv.reshape(B, N, 3, C)
            q, k, v = unbind(qkv, 2)
            q, k = self.q_norm(q), self.k_norm(k) # do q-k norm on the full seq instead.

            st()
            q, k = Attention.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

            q = q.reshape(B, N, self.num_heads, C // self.num_heads)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads)

            q, k, v = map(
                lambda t: t.reshape(b, N, self.num_heads, C // self.num_heads)
                (q, k, v),
            )
            q, k = q.to(dtype), k.to(dtype)



        else:
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = unbind(qkv, 2)
            q, k = self.q_norm(q), self.k_norm(k)

        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias) # if not bf16, no flash-attn here.
        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=MemoryEfficientAttentionFlashAttentionOp) # force flash attention

        if self.no_flash_op: # F-A does not support large batch size? force cutlas?
            # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=MemoryEfficientAttentionCutlassOp) # force flash attention

            if version.parse(xformers.__version__) >= version.parse("0.0.21"):
                # NOTE: workaround for
                # https://github.com/facebookresearch/xformers/issues/845
                # def attn(max_bs, op):

                max_bs = 32768
                L = q.shape[0]
                n_batches = math.ceil(L / max_bs)
                x = list()
                for i_batch in range(n_batches):
                    batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                    x.append(
                        xformers.ops.memory_efficient_attention(
                            q[batch],
                            k[batch],
                            v[batch],
                            attn_bias=None,
                            # op=MemoryEfficientAttentionFlashAttentionOp,
                            # op=op,
                            op=MemoryEfficientAttentionCutlassOp,
                        )
                    )
                x = torch.cat(x, 0)
                # return x
                 
                # The cutlas implementation runs in 8396.681 microseconds
                # The Flash implementation runs in 19473.491 microseconds

                # max_bs = 32768
                # math_time = benchmark_torch_function_in_microseconds(attn, max_bs, MemoryEfficientAttentionCutlassOp)
                # print(f"The cutlas implementation runs in {math_time:.3f} microseconds")

                # max_bs = 32768 // 2 # works for flash attention
                # math_time = benchmark_torch_function_in_microseconds(attn, max_bs, MemoryEfficientAttentionFlashAttentionOp)
                # print(f"The Flash implementation runs in {math_time:.3f} microseconds")
                # st()
                # pass

        else: # will enable flash attention by default.
            # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=MemoryEfficientAttentionFlashAttentionOp) # force flash attention
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias) # force flash attention

        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffCrossAttention(MemEffAttention):
    # for cross attention, where context serves as k and v
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        del self.qkv
        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x: Tensor, context: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q = self.q(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        kv = self.kv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        k, v = unbind(kv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=MemoryEfficientAttentionFlashAttentionOp)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# https://github.com/IBM/CrossViT/blob/main/models/crossvit.py
class CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:,
                      0:1, ...]).reshape(B, 1, self.num_heads,
                                         C // self.num_heads).permute(
                                             0, 2, 1,
                                             3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N,
                               self.num_heads, C // self.num_heads).permute(
                                   0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N,
                               self.num_heads, C // self.num_heads).permute(
                                   0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(
            -2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(
            B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Conv3D_Aware_CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, group_size, N, C = x.shape  # B 3 N C
        p = int(N**0.5)  # patch size
        assert p**2 == N, 'check input dim, no [cls] needed here'
        assert group_size == 3, 'designed for triplane here'

        x = x.reshape(B, group_size, p, p, C)  # expand patch token dim

        # * init qkv
        # q = torch.empty(B * group_size * N,
        #                 1,
        #                 self.num_heads,
        #                 C // self.num_heads,
        #                 device=x.device).permute(0, 2, 1, 3)
        # k = torch.empty(B * group_size * N,
        #                 2 * p,
        #                 self.num_heads,
        #                 C // self.num_heads,
        #                 device=x.device).permute(0, 2, 1, 3)
        # v = torch.empty_like(k)

        q_x = torch.empty(
            B * group_size * N,
            1,
            # self.num_heads,
            # C // self.num_heads,
            C,
            device=x.device)
        k_x = torch.empty(
            B * group_size * N,
            2 * p,
            # self.num_heads,
            # C // self.num_heads,
            C,
            device=x.device)
        v_x = torch.empty_like(k_x)

        # ! refer to the following plane order
        # N, M, _ = coordinates.shape
        # xy_coords = coordinates[..., [0, 1]]
        # yz_coords = coordinates[..., [1, 2]]
        # zx_coords = coordinates[..., [2, 0]]
        # return torch.stack([xy_coords, yz_coords, zx_coords],
        #                 dim=1).reshape(N * 3, M, 2)

        index_i, index_j = torch.meshgrid(torch.arange(0, p),
                                          torch.arange(0, p),
                                          indexing='ij')  # 16*16
        index_mesh_grid = torch.stack([index_i, index_j], 0).to(
            x.device).unsqueeze(0).repeat_interleave(B,
                                                     0).reshape(B, 2, p,
                                                                p)  # B 2 p p.

        for i in range(group_size):
            q_x[B * i * N:B * (i + 1) * N] = x[:, i:i + 1].permute(
                0, 2, 3, 1, 4).reshape(B * N, 1, C)  # B 1 p p C -> B*N, 1, C

            # TODO, how to batchify gather ops?
            plane_yz = x[:, (i + 1) % group_size:(i + 1) % group_size +
                         1]  # B 1 p p C
            plane_zx = x[:, (i + 2) % group_size:(i + 2) % group_size + 1]

            assert plane_yz.shape == plane_zx.shape == (
                B, 1, p, p, C), 'check sub plane dimensions'

            pooling_plane_yz = torch.gather(
                plane_yz,
                dim=2,
                index=index_mesh_grid[:, 0:1].reshape(B, 1, N, 1, 1).expand(
                    -1, -1, -1, p,
                    C)).permute(0, 2, 1, 3, 4)  # B 1 256 16 C => B 256 1 16 C
            pooling_plane_zx = torch.gather(
                plane_zx,
                dim=3,
                index=index_mesh_grid[:, 1:2].reshape(B, 1, 1, N, 1).expand(
                    -1, -1, p, -1,
                    C)).permute(0, 3, 1, 2, 4)  # B 1 16 256 C => B 256 1 16 C

            k_x[B * i * N:B * (i + 1) *
                N] = v_x[B * i * N:B * (i + 1) * N] = torch.cat(
                    [pooling_plane_yz, pooling_plane_zx],
                    dim=2).reshape(B * N, 2 * p,
                                   C)  # B 256 2 16 C => (B*256) 2*16 C

            # q[B * i * N: B * (i+1) * N] = self.wq(q_x).reshape(B*N, 1, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
            # k[B * i * N: B * (i+1) * N] = self.wk(k_x).reshape(B*N, 2*p, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
            # v[B * i * N: B * (i+1) * N] = self.wv(v_x).reshape(B*N, 2*p, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)

        q = self.wq(q_x).reshape(B * group_size * N, 1,
                                 self.num_heads, C // self.num_heads).permute(
                                     0, 2, 1,
                                     3)  # merge num_heads into Batch dimention
        k = self.wk(k_x).reshape(B * group_size * N, 2 * p, self.num_heads,
                                 C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(v_x).reshape(B * group_size * N, 2 * p, self.num_heads,
                                 C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(
            -2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N, N=2p here
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(
            B * 3 * N, 1,
            C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)

        # reshape x back
        x = x.reshape(B, 3, N, C)

        return x


class xformer_Conv3D_Aware_CrossAttention(nn.Module):
    # https://github.dev/facebookresearch/dinov2
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        # https://pytorch.org/blog/accelerated-generative-diffusion-models/

        self.num_heads = num_heads
        self.wq = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.w_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.index_mesh_grid = None

    def forward(self, x, attn_bias=None):

        B, group_size, N, C = x.shape  # B 3 N C
        p = int(N**0.5)  # patch size
        assert p**2 == N, 'check input dim, no [cls] needed here'
        assert group_size == 3, 'designed for triplane here'

        x = x.reshape(B, group_size, p, p, C)  # expand patch token dim

        q_x = torch.empty(B * group_size * N, 1, C, device=x.device)
        context = torch.empty(B * group_size * N, 2 * p, C,
                              device=x.device)  # k_x=v_x

        if self.index_mesh_grid is None:  # further accelerate
            index_i, index_j = torch.meshgrid(torch.arange(0, p),
                                              torch.arange(0, p),
                                              indexing='ij')  # 16*16
            index_mesh_grid = torch.stack([index_i, index_j], 0).to(
                x.device).unsqueeze(0).repeat_interleave(B, 0).reshape(
                    B, 2, p, p)  # B 2 p p.
            self.index_mesh_grid = index_mesh_grid[0:1]
        else:
            index_mesh_grid = self.index_mesh_grid.clone().repeat_interleave(
                B, 0)
            assert index_mesh_grid.shape == (
                B, 2, p, p), 'check index_mesh_grid dimension'

        for i in range(group_size):
            q_x[B * i * N:B * (i + 1) * N] = x[:, i:i + 1].permute(
                0, 2, 3, 1, 4).reshape(B * N, 1, C)  # B 1 p p C -> B*N, 1, C

            # TODO, how to batchify gather ops?
            plane_yz = x[:, (i + 1) % group_size:(i + 1) % group_size +
                         1]  # B 1 p p C
            plane_zx = x[:, (i + 2) % group_size:(i + 2) % group_size + 1]

            assert plane_yz.shape == plane_zx.shape == (
                B, 1, p, p, C), 'check sub plane dimensions'

            pooling_plane_yz = torch.gather(
                plane_yz,
                dim=2,
                index=index_mesh_grid[:, 0:1].reshape(B, 1, N, 1, 1).expand(
                    -1, -1, -1, p,
                    C)).permute(0, 2, 1, 3, 4)  # B 1 256 16 C => B 256 1 16 C
            pooling_plane_zx = torch.gather(
                plane_zx,
                dim=3,
                index=index_mesh_grid[:, 1:2].reshape(B, 1, 1, N, 1).expand(
                    -1, -1, p, -1,
                    C)).permute(0, 3, 1, 2, 4)  # B 1 16 256 C => B 256 1 16 C

            context[B * i * N:B * (i + 1) * N] = torch.cat(
                [pooling_plane_yz, pooling_plane_zx],
                dim=2).reshape(B * N, 2 * p,
                               C)  # B 256 2 16 C => (B*256) 2*16 C

        # B, N, C = x.shape

        q = self.wq(q_x).reshape(B * group_size * N, 1, self.num_heads,
                                 C // self.num_heads)

        kv = self.w_kv(context).reshape(B * group_size * N, 2 * p, 2,
                                        self.num_heads, C // self.num_heads)
        k, v = unbind(kv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=MemoryEfficientAttentionFlashAttentionOp)
        x = x.transpose(1, 2).reshape([B * 3 * N, 1, C]).reshape(B, 3, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class xformer_Conv3D_Aware_CrossAttention_xygrid(
        xformer_Conv3D_Aware_CrossAttention):
    """implementation wise clearer, but yields identical results with xformer_Conv3D_Aware_CrossAttention
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                         proj_drop)

    def forward(self, x, attn_bias=None):

        B, group_size, N, C = x.shape  # B 3 N C
        p = int(N**0.5)  # patch size
        assert p**2 == N, 'check input dim, no [cls] needed here'
        assert group_size == 3, 'designed for triplane here'

        x = x.reshape(B, group_size, p, p, C)  # expand patch token dim

        q_x = torch.empty(B * group_size * N, 1, C, device=x.device)
        context = torch.empty(B * group_size * N, 2 * p, C,
                              device=x.device)  # k_x=v_x

        if self.index_mesh_grid is None:  # further accelerate
            index_u, index_v = torch.meshgrid(
                torch.arange(0, p), torch.arange(0, p),
                indexing='xy')  # ! switch to 'xy' here to match uv coordinate
            index_mesh_grid = torch.stack([index_u, index_v], 0).to(
                x.device).unsqueeze(0).repeat_interleave(B, 0).reshape(
                    B, 2, p, p)  # B 2 p p.
            self.index_mesh_grid = index_mesh_grid[0:1]
        else:
            index_mesh_grid = self.index_mesh_grid.clone().repeat_interleave(
                B, 0)
            assert index_mesh_grid.shape == (
                B, 2, p, p), 'check index_mesh_grid dimension'

        for i in range(group_size):
            q_x[B * i * N:B * (i + 1) * N] = x[:, i:i + 1].permute(
                0, 2, 3, 1, 4).reshape(B * N, 1, C)  # B 1 p p C -> B*N, 1, C

            # TODO, how to batchify gather ops?
            plane_yz = x[:, (i + 1) % group_size:(i + 1) % group_size +
                         1]  # B 1 p p C
            plane_zx = x[:, (i + 2) % group_size:(i + 2) % group_size + 1]

            assert plane_yz.shape == plane_zx.shape == (
                B, 1, p, p, C), 'check sub plane dimensions'

            pooling_plane_yz = torch.gather(
                plane_yz,
                dim=2,
                index=index_mesh_grid[:, 1:2].reshape(B, 1, N, 1, 1).expand(
                    -1, -1, -1, p,
                    C)).permute(0, 2, 1, 3, 4)  # B 1 256 16 C => B 256 1 16 C
            pooling_plane_zx = torch.gather(
                plane_zx,
                dim=3,
                index=index_mesh_grid[:, 0:1].reshape(B, 1, 1, N, 1).expand(
                    -1, -1, p, -1,
                    C)).permute(0, 3, 1, 2, 4)  # B 1 16 256 C => B 256 1 16 C

            context[B * i * N:B * (i + 1) * N] = torch.cat(
                [pooling_plane_yz, pooling_plane_zx],
                dim=2).reshape(B * N, 2 * p,
                               C)  # B 256 2 16 C => (B*256) 2*16 C

        # B, N, C = x.shape
        q = self.wq(q_x).reshape(B * group_size * N, 1, self.num_heads,
                                 C // self.num_heads)

        kv = self.w_kv(context).reshape(B * group_size * N, 2 * p, 2,
                                        self.num_heads, C // self.num_heads)
        k, v = unbind(kv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=MemoryEfficientAttentionFlashAttentionOp)
        x = x.transpose(1, 2).reshape([B * 3 * N, 1, C]).reshape(B, 3, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class xformer_Conv3D_Aware_CrossAttention_xygrid_withinC(
        xformer_Conv3D_Aware_CrossAttention_xygrid):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                         proj_drop)

    def forward(self, x, attn_bias=None):
        # ! split x: B N C into B 3 N C//3
        B, N, C = x.shape
        x = x.reshape(B, N, C // 3, 3).permute(0, 3, 1,
                                               2)  # B N C 3 -> B 3 N C
        x_out = super().forward(x, attn_bias)  # B 3 N C
        x_out = x_out.permute(0, 2, 3, 1)# B 3 N C -> B N C 3
        x_out = x_out.reshape(*x_out.shape[:2], -1) # B N C 3 -> B N C3
        return x_out.contiguous()

class self_cross_attn(nn.Module):
    def __init__(self, dino_attn, cross_attn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dino_attn = dino_attn
        self.cross_attn = cross_attn
    
    def forward(self, x_norm):
        y = self.dino_attn(x_norm) + x_norm
        return self.cross_attn(y) # will add x in the original code

# class RodinRollOutConv(nn.Module):
#     """implementation wise clearer, but yields identical results with xformer_Conv3D_Aware_CrossAttention
#     Use Group Conv
#     """

#     def __init__(self, in_chans, out_chans=None):
#         super().__init__()
#         # input: B 3C H W
#         if out_chans is None:
#             out_chans = in_chans

#         self.roll_out_convs = nn.Conv2d(in_chans,
#                                         out_chans,
#                                         kernel_size=3,
#                                         groups=3,
#                                         padding=1)

#     def forward(self, x):
#         return self.roll_out_convs(x)


class RodinRollOutConv3D(nn.Module):
    """implementation wise clearer, but yields identical results with xformer_Conv3D_Aware_CrossAttention
    """

    def __init__(self, in_chans, out_chans=None):
        super().__init__()
        if out_chans is None:
            out_chans = in_chans

        self.out_chans = out_chans // 3

        self.roll_out_convs = nn.Conv2d(in_chans,
                                        self.out_chans,
                                        kernel_size=3,
                                        padding=1)

    def forward(self, x):
        # todo, reshape before input?

        B, C3, p, p = x.shape  # B 3C H W
        C = C3 // 3
        group_size = C3 // C
        assert group_size == 3

        x = x.reshape(B, 3, C, p, p)

        roll_out_x = torch.empty(B, group_size * C, p, 3 * p,
                                 device=x.device)  # B, 3C, H, 3W

        for i in range(group_size):
            plane_xy = x[:, i]  # B C H W

            # TODO, simply do the average pooling?
            plane_yz_pooling = x[:, (i + 1) % group_size].mean(
                dim=-1, keepdim=True).repeat_interleave(
                    p, dim=-1)  # B C H W -> B C H 1 -> B C H W, reduce z dim
            plane_zx_pooling = x[:, (i + 2) % group_size].mean(
                dim=-2, keepdim=True).repeat_interleave(
                    p, dim=-2)  # B C H W -> B C 1 W -> B C H W, reduce z dim

            roll_out_x[..., i * p:(i + 1) * p] = torch.cat(
                [plane_xy, plane_yz_pooling, plane_zx_pooling],
                1)  # fill in the 3W dim

        x = self.roll_out_convs(roll_out_x)  # B C H 3W

        x = x.reshape(B, self.out_chans, p, 3, p)
        x = x.permute(0, 3, 1, 2, 4).reshape(B, 3 * self.out_chans, p,
                                             p)  # B 3C H W

        return x


class RodinRollOutConv3D_GroupConv(nn.Module):
    """implementation wise clearer, but yields identical results with xformer_Conv3D_Aware_CrossAttention
    """

    def __init__(self,
                 in_chans,
                 out_chans=None,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super().__init__()
        if out_chans is None:
            out_chans = in_chans

        self.roll_out_convs = nn.Conv2d(
            in_chans * 3,
            out_chans,
            kernel_size=kernel_size,
            groups=3,  # B 9C H W
            stride=stride,
            padding=padding)

    # @torch.autocast(device_type='cuda')
    def forward(self, x):
        # todo, reshape before input?

        B, C3, p, p = x.shape  # B 3C H W
        C = C3 // 3
        group_size = C3 // C
        assert group_size == 3

        x = x.reshape(B, 3, C, p, p)

        roll_out_x = torch.empty(B, group_size * C * 3, p, p,
                                 device=x.device)  # B, 3C, H, 3W

        for i in range(group_size):
            plane_xy = x[:, i]  # B C H W

            # # TODO, simply do the average pooling?
            plane_yz_pooling = x[:, (i + 1) % group_size].mean(
                dim=-1, keepdim=True).repeat_interleave(
                    p, dim=-1)  # B C H W -> B C H 1 -> B C H W, reduce z dim
            plane_zx_pooling = x[:, (i + 2) % group_size].mean(
                dim=-2, keepdim=True).repeat_interleave(
                    p, dim=-2)  # B C H W -> B C 1 W -> B C H W, reduce z dim

            roll_out_x[:, i * 3 * C:(i + 1) * 3 * C] = torch.cat(
                [plane_xy, plane_yz_pooling, plane_zx_pooling],
                1)  # fill in the 3W dim

            # ! directly cat, avoid intermediate vars
            # ? why OOM
            # roll_out_x[:, i * 3 * C:(i + 1) * 3 * C] = torch.cat(
            #     [
            #         x[:, i],
            #         x[:, (i + 1) % group_size].mean(
            #             dim=-1, keepdim=True).repeat_interleave(p, dim=-1),
            #         x[:, (i + 2) % group_size].mean(
            #             dim=-2, keepdim=True).repeat_interleave(
            #                 p, dim=-2
            #             )  # B C H W -> B C 1 W -> B C H W, reduce z dim
            #     ],
            #     1)  # fill in the 3C dim

        x = self.roll_out_convs(roll_out_x)  # B 3C H W

        return x


class RodinRollOut_GroupConv_noConv3D(nn.Module):
    """only roll out and do Conv on individual planes
    """

    def __init__(self,
                 in_chans,
                 out_chans=None,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super().__init__()
        if out_chans is None:
            out_chans = in_chans

        self.roll_out_inplane_conv = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            groups=3,  # B 3C H W
            stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.roll_out_inplane_conv(x)  # B 3C H W
        return x


# class RodinConv3D_SynthesisLayer_withact(nn.Module):
#     def __init__(self, in_chans, out_chans) -> None:
#         super().__init__()

#         self.act = nn.LeakyReLU(inplace=True)
#         self.conv = nn.Sequential(
#             RodinRollOutConv3D_GroupConv(in_chans, out_chans),
#             nn.LeakyReLU(inplace=True),
#         )

#         if in_chans != out_chans:
#             self.short_cut = RodinRollOutConv3D_GroupConv(in_chans, out_chans) # PSNR 13 first iteration.
#         else:
#             self.short_cut = None

# def forward(self, feats):

#     if self.short_cut is not None:
#         res_feats = self.short_cut(feats)
#     else:
#         res_feats = feats

#     # return res_feats + self.conv(feats)
#     feats = res_feats + self.conv(feats)
#     return self.act(feats) # as in resnet, add an act before return


class RodinConv3D_SynthesisLayer_mlp_unshuffle_as_residual(nn.Module):

    def __init__(self, in_chans, out_chans) -> None:
        super().__init__()

        self.act = nn.LeakyReLU(inplace=True)
        self.conv = nn.Sequential(
            RodinRollOutConv3D_GroupConv(in_chans, out_chans),
            nn.LeakyReLU(inplace=True),
        )

        self.out_chans = out_chans
        if in_chans != out_chans:
            # self.short_cut = RodinRollOutConv3D_GroupConv(in_chans, out_chans) # PSNR 13 first iteration.
            self.short_cut = nn.Linear(  # B 3C H W -> B 3C 4H 4W
                in_chans // 3,  # 144 / 3 = 48
                out_chans // 3 * 4 * 4,  # 32 * 16
                bias=True)  # decoder to pat

            # RodinRollOutConv3D_GroupConv(in_chans, out_chans) # PSNR 13 first iteration.
        else:
            self.short_cut = None

    def shortcut_unpatchify_triplane(self,
                                     x,
                                     p=None,
                                     unpatchify_out_chans=None):
        """separate triplane version; x shape: B (3*257) 768
        """

        assert self.short_cut is not None

        # B, L, C = x.shape
        B, C3, h, w = x.shape
        assert h == w
        L = h * w
        x = x.reshape(B, C3 // 3, 3, L).permute(0, 2, 3,
                                                1)  # (B, 3, L // 3, C)

        x = self.short_cut(x)

        p = h * 4

        x = x.reshape(shape=(B, 3, h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('ndhwpqc->ndchpwq',
                         x)  # nplanes, C order in the renderer.py
        x = x.reshape(shape=(B, 3 * self.out_chans, h * p, h * p))
        return x

    def forward(self, feats):

        if self.short_cut is not None:
            res_feats = self.shortcut_unpatchify_triplane(feats)
        else:
            res_feats = feats

        # return res_feats + self.conv(feats)

        feats = res_feats + self.conv(feats)
        return self.act(feats)  # as in resnet, add an act before return


# class RodinConv3D_SynthesisLayer(nn.Module):
#     def __init__(self, in_chans, out_chans) -> None:
#         super().__init__()

#         self.act = nn.LeakyReLU(inplace=True)
#         self.conv = nn.Sequential(
#             RodinRollOutConv3D_GroupConv(in_chans, out_chans),
#             nn.LeakyReLU(inplace=True),
#         )

#         if in_chans != out_chans:
#             self.short_cut = RodinRollOutConv3D_GroupConv(in_chans, out_chans) # PSNR 13 first iteration.
#         else:
#             self.short_cut = None

#     def forward(self, feats):

#         if self.short_cut is not None:
#             res_feats = self.short_cut(feats)
#         else:
#             res_feats = feats

#         # return res_feats + self.conv(feats)

#         feats = res_feats + self.conv(feats)
#         # return self.act(feats) # as in resnet, add an act before return
#         return feats # ! old behaviour, no act


# previous worked version
class RodinConv3D_SynthesisLayer(nn.Module):

    def __init__(self, in_chans, out_chans) -> None:
        super().__init__()
        # x2 SR + 1x1 Conv Residual BLK
        # self.conv3D = RodinRollOutConv3D(in_chans, out_chans)

        self.act = nn.LeakyReLU(inplace=True)
        self.conv = nn.Sequential(
            RodinRollOutConv3D_GroupConv(in_chans, out_chans),
            nn.LeakyReLU(inplace=True),
        )

        if in_chans != out_chans:
            self.short_cut = RodinRollOutConv3D_GroupConv(in_chans, out_chans)
        else:
            self.short_cut = None

    def forward(self, feats):
        feats_out = self.conv(feats)
        if self.short_cut is not None:
            # ! failed below
            feats_out = self.short_cut(
                feats
            ) + feats_out  # ! only difference here, no act() compared with baseline
            # feats_out = self.act(self.short_cut(feats)) + feats_out # ! only difference here, no act() compared with baseline
        else:
            feats_out = feats_out + feats
        return feats_out


class RodinRollOutConv3DSR2X(nn.Module):

    def __init__(self, in_chans, **kwargs) -> None:
        super().__init__()
        self.conv3D = RodinRollOutConv3D_GroupConv(in_chans)
        # self.conv3D = RodinRollOutConv3D(in_chans)
        self.act = nn.LeakyReLU(inplace=True)
        self.input_resolution = 224

    def forward(self, x):
        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        group_size = C3 // C

        assert group_size == 3
        # p = int(N**0.5)  # patch size
        # assert p**2 == N, 'check input dim, no [cls] needed here'
        assert group_size == 3, 'designed for triplane here'

        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x,
                                                size=(self.input_resolution,
                                                      self.input_resolution),
                                                mode='bilinear',
                                                align_corners=False,
                                                antialias=True)

        x = x + self.conv3D(x)

        return x


class RodinRollOutConv3DSR4X_lite(nn.Module):

    def __init__(self, in_chans, input_resolutiopn=256, **kwargs) -> None:
        super().__init__()
        self.conv3D_0 = RodinRollOutConv3D_GroupConv(in_chans)
        self.conv3D_1 = RodinRollOutConv3D_GroupConv(in_chans)

        self.act = nn.LeakyReLU(inplace=True)
        self.input_resolution = input_resolutiopn

    def forward(self, x):
        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        group_size = C3 // C

        assert group_size == 3
        # p = int(N**0.5)  # patch size
        # assert p**2 == N, 'check input dim, no [cls] needed here'
        assert group_size == 3, 'designed for triplane here'

        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x,
                                                size=(self.input_resolution,
                                                      self.input_resolution),
                                                mode='bilinear',
                                                align_corners=False,
                                                antialias=True)

        # ! still not convering, not bug here?
        # x = x + self.conv3D_0(x)
        # x = x + self.conv3D_1(x)

        x = x + self.act(self.conv3D_0(x))
        x = x + self.act(self.conv3D_1(x))

        # TODO: which is better, bilinear + conv or PixelUnshuffle?

        return x


# class RodinConv3D2X_lite_mlp_as_residual(nn.Module):
#     """lite 4X version, with MLP unshuffle to change the dimention
#     """
#     def __init__(self, in_chans, out_chans, input_resolution=256) -> None:
#         super().__init__()

#         self.act = nn.LeakyReLU(inplace=True)

#         self.conv3D_0 = RodinRollOutConv3D_GroupConv(in_chans, out_chans)
#         self.conv3D_1 = RodinRollOutConv3D_GroupConv(out_chans, out_chans)

#         self.act = nn.LeakyReLU(inplace=True)
#         self.input_resolution = input_resolution

#         self.out_chans = out_chans
#         if in_chans != out_chans: # ! only change the dimension
#             self.short_cut = nn.Linear( # B 3C H W -> B 3C 4H 4W
#                 in_chans//3, # 144 / 3 = 48
#                 out_chans//3, # 32 * 16
#                 bias=True)  # decoder to pat
#         else:
#             self.short_cut = None

#     def shortcut_unpatchify_triplane(self, x, p=None):
#         """separate triplane version; x shape: B (3*257) 768
#         """

#         assert self.short_cut is not None

#         # B, L, C = x.shape
#         B, C3, h, w = x.shape
#         assert h == w
#         L = h*w
#         x = x.reshape(B, C3//3, 3, L).permute(0,2,3,1) # (B, 3, L // 3, C_in)

#         x = self.short_cut(x) # B 3 L//3 C_out

#         x = x.permute(0,1,3,2) # B 3 C_out L//3
#         x = x.reshape(shape=(B, self.out_chans, h, w))

#         # directly resize to the target, no unpatchify here since no 3D ViT is included here
#         if w != self.input_resolution:
#             x = torch.nn.functional.interpolate(x, # 4X SR
#                                                 size=(self.input_resolution,
#                                                       self.input_resolution),
#                                                 mode='bilinear',
#                                                 align_corners=False,
#                                                 antialias=True)

#         return x

#     def forward(self, x):

#         # x: B 3 112*112 C
#         B, C3, p, p = x.shape  # after unpachify triplane
#         C = C3 // 3

#         if self.short_cut is not None:
#             res_feats = self.shortcut_unpatchify_triplane(x)
#         else:
#             res_feats = x

#         """following forward code copied from lite4x version
#         """
#         x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
#                                           p)  # B 3 C N -> B 3C h W

#         if x.shape[-1] != self.input_resolution:
#             x = torch.nn.functional.interpolate(x, # 4X SR
#                                                 size=(self.input_resolution,
#                                                       self.input_resolution),
#                                                 mode='bilinear',
#                                                 align_corners=False,
#                                                 antialias=True)

#         x = res_feats + self.act(self.conv3D_0(x))
#         x = x + self.act(self.conv3D_1(x))

#         return x


class RodinConv3D4X_lite_mlp_as_residual(nn.Module):
    """lite 4X version, with MLP unshuffle to change the dimention
    """

    def __init__(self,
                 in_chans,
                 out_chans,
                 input_resolution=256,
                 interp_mode='bilinear',
                 bcg_triplane=False) -> None:
        super().__init__()

        self.interp_mode = interp_mode

        self.act = nn.LeakyReLU(inplace=True)

        self.conv3D_0 = RodinRollOutConv3D_GroupConv(in_chans, out_chans)
        self.conv3D_1 = RodinRollOutConv3D_GroupConv(out_chans, out_chans)
        self.bcg_triplane = bcg_triplane
        if bcg_triplane:
            self.conv3D_1_bg = RodinRollOutConv3D_GroupConv(
                out_chans, out_chans)

        self.act = nn.LeakyReLU(inplace=True)
        self.input_resolution = input_resolution

        self.out_chans = out_chans
        if in_chans != out_chans:  # ! only change the dimension
            self.short_cut = nn.Linear(  # B 3C H W -> B 3C 4H 4W
                in_chans // 3,  # 144 / 3 = 48
                out_chans // 3,  # 32 * 16
                bias=True)  # decoder to pat
        else:
            self.short_cut = None

    def shortcut_unpatchify_triplane(self, x, p=None):
        """separate triplane version; x shape: B (3*257) 768
        """

        assert self.short_cut is not None

        B, C3, h, w = x.shape
        assert h == w
        L = h * w
        x = x.reshape(B, C3 // 3, 3, L).permute(0, 2, 3,
                                                1)  # (B, 3, L // 3, C_in)

        x = self.short_cut(x)  # B 3 L//3 C_out

        x = x.permute(0, 1, 3, 2)  # B 3 C_out L//3
        x = x.reshape(shape=(B, self.out_chans, h, w))

        # directly resize to the target, no unpatchify here since no 3D ViT is included here
        if w != self.input_resolution:
            x = torch.nn.functional.interpolate(
                x,  # 4X SR
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=True)

        return x

    def interpolate(self, feats):
        if self.interp_mode == 'bilinear':
            return torch.nn.functional.interpolate(
                feats,  # 4X SR
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=True)
        else:
            return torch.nn.functional.interpolate(
                feats,  # 4X SR
                size=(self.input_resolution, self.input_resolution),
                mode='nearest',
            )

    def forward(self, x):

        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3

        if self.short_cut is not None:
            res_feats = self.shortcut_unpatchify_triplane(x)
        else:
            res_feats = x
            if res_feats.shape[-1] != self.input_resolution:
                res_feats = self.interpolate(res_feats)
        """following forward code copied from lite4x version
        """
        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if x.shape[-1] != self.input_resolution:
            x = self.interpolate(x)

        x0 = res_feats + self.act(self.conv3D_0(x))  # the base feature
        x = x0 + self.act(self.conv3D_1(x0))
        if self.bcg_triplane:
            x_bcg = x0 + self.act(self.conv3D_1_bg(x0))
            return torch.cat([x, x_bcg], 1)
        else:
            return x


class RodinConv3D4X_lite_mlp_as_residual_litev2(
        RodinConv3D4X_lite_mlp_as_residual):

    def __init__(self,
                 in_chans,
                 out_chans,
                 num_feat=128,
                 input_resolution=256,
                 interp_mode='bilinear',
                 bcg_triplane=False) -> None:
        super().__init__(in_chans, out_chans, input_resolution, interp_mode,
                         bcg_triplane)

        self.conv3D_0 = RodinRollOutConv3D_GroupConv(in_chans, in_chans)
        self.conv_before_upsample = RodinRollOut_GroupConv_noConv3D(
            in_chans, num_feat * 3)
        self.conv3D_1 = RodinRollOut_GroupConv_noConv3D(
            num_feat * 3, num_feat * 3)
        self.conv_last = RodinRollOut_GroupConv_noConv3D(
            num_feat * 3, out_chans)
        self.short_cut = None

    def forward(self, x):

        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3

        # if self.short_cut is not None:
        #     res_feats = self.shortcut_unpatchify_triplane(x)
        # else:
        #     res_feats = x
        #     if res_feats.shape[-1] != self.input_resolution:
        #         res_feats = self.interpolate(res_feats)
        """following forward code copied from lite4x version
        """
        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        x = x + self.conv3D_0(x)  # the base feature
        x = self.act(self.conv_before_upsample(x))

        # if x.shape[-1] != self.input_resolution:
        x = self.conv_last(self.act(self.conv3D_1(self.interpolate(x))))

        return x


class RodinConv3D4X_lite_mlp_as_residual_lite(
        RodinConv3D4X_lite_mlp_as_residual):

    def __init__(self,
                 in_chans,
                 out_chans,
                 input_resolution=256,
                 interp_mode='bilinear') -> None:
        super().__init__(in_chans, out_chans, input_resolution, interp_mode)
        """replace the first Rodin Conv 3D with ordinary rollout conv to save memory
        """
        self.conv3D_0 = RodinRollOut_GroupConv_noConv3D(in_chans, out_chans)


class SR3D(nn.Module):
    # https://github.com/SeanChenxy/Mimic3D/blob/77d313656df3cd5536d2c4c5766db3a56208eea6/training/networks_stylegan2.py#L629
    # roll-out and apply two deconv/pixelUnshuffle layer

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class RodinConv3D4X_lite_mlp_as_residual_improved(nn.Module):

    def __init__(self,
                 in_chans,
                 num_feat,
                 out_chans,
                 input_resolution=256) -> None:
        super().__init__()

        assert in_chans == 4 * out_chans
        assert num_feat == 2 * out_chans
        self.input_resolution = input_resolution

        # refer to https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py#L750
        self.upscale = 4

        self.conv_after_body = RodinRollOutConv3D_GroupConv(
            in_chans, in_chans, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            RodinRollOutConv3D_GroupConv(in_chans, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.conv_up1 = RodinRollOutConv3D_GroupConv(num_feat, num_feat, 3, 1,
                                                     1)
        if self.upscale == 4:
            self.conv_up2 = RodinRollOutConv3D_GroupConv(
                num_feat, num_feat, 3, 1, 1)
        self.conv_hr = RodinRollOutConv3D_GroupConv(num_feat, num_feat, 3, 1,
                                                    1)
        self.conv_last = RodinRollOutConv3D_GroupConv(num_feat, out_chans, 3,
                                                      1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        """following forward code copied from lite4x version
        """
        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        # ? nearest or bilinear
        x = self.conv_after_body(x) + x
        x = self.conv_before_upsample(x)
        x = self.lrelu(
            self.conv_up1(
                torch.nn.functional.interpolate(
                    x,
                    scale_factor=2,
                    mode='nearest',
                    # align_corners=False,
                    # antialias=True
                )))
        if self.upscale == 4:
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(
                        x,
                        scale_factor=2,
                        mode='nearest',
                        # align_corners=False,
                        # antialias=True
                    )))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))

        assert x.shape[-1] == self.input_resolution

        return x


class RodinConv3D4X_lite_improved_lint_withresidual(nn.Module):

    def __init__(self,
                 in_chans,
                 num_feat,
                 out_chans,
                 input_resolution=256) -> None:
        super().__init__()

        assert in_chans == 4 * out_chans
        assert num_feat == 2 * out_chans
        self.input_resolution = input_resolution

        # refer to https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py#L750
        self.upscale = 4

        self.conv_after_body = RodinRollOutConv3D_GroupConv(
            in_chans, in_chans, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            RodinRollOutConv3D_GroupConv(in_chans, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.conv_up1 = RodinRollOutConv3D_GroupConv(num_feat, num_feat, 3, 1,
                                                     1)
        if self.upscale == 4:
            self.conv_up2 = RodinRollOutConv3D_GroupConv(
                num_feat, num_feat, 3, 1, 1)
        self.conv_hr = RodinRollOutConv3D_GroupConv(num_feat, num_feat, 3, 1,
                                                    1)
        self.conv_last = RodinRollOutConv3D_GroupConv(num_feat, out_chans, 3,
                                                      1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        """following forward code copied from lite4x version
        """
        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        # ? nearest or bilinear
        x = self.conv_after_body(x) + x
        x = self.conv_before_upsample(x)
        x = self.lrelu(
            self.conv_up1(
                torch.nn.functional.interpolate(
                    x,
                    scale_factor=2,
                    mode='nearest',
                    # align_corners=False,
                    # antialias=True
                )))
        if self.upscale == 4:
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(
                        x,
                        scale_factor=2,
                        mode='nearest',
                        # align_corners=False,
                        # antialias=True
                    )))
        x = self.conv_last(self.lrelu(self.conv_hr(x) + x))

        assert x.shape[-1] == self.input_resolution

        return x


class RodinRollOutConv3DSR_FlexibleChannels(nn.Module):

    def __init__(self,
                 in_chans,
                 num_out_ch=96,
                 input_resolution=256,
                 **kwargs) -> None:
        super().__init__()

        self.block0 = RodinConv3D_SynthesisLayer(in_chans,
                                                 num_out_ch)  # in_chans=48
        self.block1 = RodinConv3D_SynthesisLayer(num_out_ch, num_out_ch)

        self.input_resolution = input_resolution  # 64 -> 256 SR

    def forward(self, x):
        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        # group_size = C3 // C

        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x,
                                                size=(self.input_resolution,
                                                      self.input_resolution),
                                                mode='bilinear',
                                                align_corners=False,
                                                antialias=True)

        x = self.block0(x)
        x = self.block1(x)

        return x


# previous worked version
class RodinRollOutConv3DSR4X(nn.Module):
    # follow PixelUnshuffleUpsample

    def __init__(self, in_chans, **kwargs) -> None:
        super().__init__()
        # self.block0 = RodinConv3D_SynthesisLayer(in_chans, 96 * 2) # TODO, match the old behaviour now.
        # self.block1 = RodinConv3D_SynthesisLayer(96 * 2, 96)

        self.block0 = RodinConv3D_SynthesisLayer(in_chans, 96)
        self.block1 = RodinConv3D_SynthesisLayer(
            96, 96)  # baseline choice, validate with no LPIPS loss here

        self.input_resolution = 64  # 64 -> 256

    def forward(self, x):
        # x: B 3 112*112 C
        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        # group_size = C3 // C

        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x,
                                                size=(self.input_resolution,
                                                      self.input_resolution),
                                                mode='bilinear',
                                                align_corners=False,
                                                antialias=True)

        x = self.block0(x)
        x = self.block1(x)

        return x


class Upsample3D(nn.Module):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        super().__init__()

        m_convs = []
        m_pixelshuffle = []

        assert (scale & (scale - 1)) == 0, 'scale = 2^n'
        self.scale = scale

        for _ in range(int(math.log(scale, 2))):
            m_convs.append(
                RodinRollOutConv3D_GroupConv(num_feat, 4 * num_feat, 3, 1, 1))
            m_pixelshuffle.append(nn.PixelShuffle(2))

        self.m_convs = nn.ModuleList(m_convs)
        self.m_pixelshuffle = nn.ModuleList(m_pixelshuffle)

    # @torch.autocast(device_type='cuda')
    def forward(self, x):
        for scale_idx in range(int(math.log(self.scale, 2))):
            x = self.m_convs[scale_idx](x)  # B 3C H W
            # x =
            # B, C3, H, W = x.shape
            x = x.reshape(x.shape[0] * 3, x.shape[1] // 3, *x.shape[2:])
            x = self.m_pixelshuffle[scale_idx](x)
            x = x.reshape(x.shape[0] // 3, x.shape[1] * 3, *x.shape[2:])

        return x


class RodinConv3DPixelUnshuffleUpsample(nn.Module):

    def __init__(self,
                 output_dim,
                 num_feat=32 * 6,
                 num_out_ch=32 * 3,
                 sr_ratio=4,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.conv_after_body = RodinRollOutConv3D_GroupConv(
            output_dim, output_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            RodinRollOutConv3D_GroupConv(output_dim, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample3D(sr_ratio, num_feat)  # 4 time SR
        self.conv_last = RodinRollOutConv3D_GroupConv(num_feat, num_out_ch, 3,
                                                      1, 1)

    # @torch.autocast(device_type='cuda')
    def forward(self, x, input_skip_connection=True, *args, **kwargs):
        # x = self.conv_first(x)
        if input_skip_connection:
            x = self.conv_after_body(x) + x
        else:
            x = self.conv_after_body(x)

        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x


class RodinConv3DPixelUnshuffleUpsample_improvedVersion(nn.Module):

    def __init__(
        self,
        output_dim,
        num_out_ch=32 * 3,
        sr_ratio=4,
        input_resolution=256,
    ) -> None:
        super().__init__()

        self.input_resolution = input_resolution

        # self.conv_first = RodinRollOutConv3D_GroupConv(output_dim, num_out_ch,
        #                                               3, 1, 1)
        self.upsample = Upsample3D(sr_ratio, output_dim)  # 4 time SR
        self.conv_last = RodinRollOutConv3D_GroupConv(output_dim, num_out_ch,
                                                      3, 1, 1)

    def forward(self, x, bilinear_upsample=True):

        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        group_size = C3 // C

        assert group_size == 3, 'designed for triplane here'

        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if bilinear_upsample and x.shape[-1] != self.input_resolution:
            x_bilinear_upsample = torch.nn.functional.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=True)
            x = self.upsample(x) + x_bilinear_upsample
        else:
            # x_bilinear_upsample = x
            x = self.upsample(x)

        x = self.conv_last(x)

        return x


class RodinConv3DPixelUnshuffleUpsample_improvedVersion2(nn.Module):
    """removed nearest neighbour residual conenctions, add a conv layer residual conenction
    """

    def __init__(
        self,
        output_dim,
        num_out_ch=32 * 3,
        sr_ratio=4,
        input_resolution=256,
    ) -> None:
        super().__init__()

        self.input_resolution = input_resolution

        self.conv_after_body = RodinRollOutConv3D_GroupConv(
            output_dim, num_out_ch, 3, 1, 1)
        self.upsample = Upsample3D(sr_ratio, output_dim)  # 4 time SR
        self.conv_last = RodinRollOutConv3D_GroupConv(output_dim, num_out_ch,
                                                      3, 1, 1)

    def forward(self, x, input_skip_connection=True):

        B, C3, p, p = x.shape  # after unpachify triplane
        C = C3 // 3
        group_size = C3 // C

        assert group_size == 3, 'designed for triplane here'

        x = x.permute(0, 1, 3, 2).reshape(B, 3 * C, p,
                                          p)  # B 3 C N -> B 3C h W

        if input_skip_connection:
            x = self.conv_after_body(x) + x
        else:
            x = self.conv_after_body(x)

        x = self.upsample(x)
        x = self.conv_last(x)

        return x


class CLSCrossAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim,
                           hidden_features=mlp_hidden_dim,
                           act_layer=act_layer,
                           drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Conv3DCrossAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Conv3D_Aware_CrossAttention(dim,
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                attn_drop=attn_drop,
                                                proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim,
                           hidden_features=mlp_hidden_dim,
                           act_layer=act_layer,
                           drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Conv3DCrossAttentionBlockXformerMHA(Conv3DCrossAttentionBlock):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0,
                 attn_drop=0,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 has_mlp=False):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, has_mlp)
        # self.attn = xformer_Conv3D_Aware_CrossAttention(dim,
        self.attn = xformer_Conv3D_Aware_CrossAttention_xygrid(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)


class Conv3DCrossAttentionBlockXformerMHANested(
        Conv3DCrossAttentionBlockXformerMHA):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 has_mlp=False):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, has_mlp)
        """for in-place replaing the internal attn in Dino ViT.
        """

    def forward(self, x):
        Bx3, N, C = x.shape
        B, group_size = Bx3 // 3, 3
        x = x.reshape(B, group_size, N, C)  # in plane vit
        x = super().forward(x)
        return x.reshape(B * group_size, N,
                         C)  # to match the original attn size


class Conv3DCrossAttentionBlockXformerMHANested_withinC(
        Conv3DCrossAttentionBlockXformerMHANested):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0,
                 attn_drop=0,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 has_mlp=False):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, has_mlp)
        self.attn = xformer_Conv3D_Aware_CrossAttention_xygrid_withinC(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

    def forward(self, x):
        # basic TX attention forward function
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TriplaneFusionBlock(nn.Module):
    """4 ViT blocks + 1 CrossAttentionBlock
    """

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 cross_attention_blk=CLSCrossAttentionBlock,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        if use_fusion_blk:
            self.fusion = nn.ModuleList()

            # copied vit settings from https://github.dev/facebookresearch/dinov2
            nh = num_heads
            dim = embed_dim

            mlp_ratio = 4  # defined for all dino2 model
            qkv_bias = True
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            drop_path_rate = 0.3  # default setting
            attn_drop = proj_drop = 0.0
            qk_scale = None  # TODO, double check

            for d in range(self.num_branches):
                self.fusion.append(
                    cross_attention_blk(
                        dim=dim,
                        num_heads=nh,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        # drop=drop,
                        drop=proj_drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path_rate,
                        norm_layer=norm_layer,  # type: ignore
                        has_mlp=False))
        else:
            self.fusion = None

    def forward(self, x):
        # modified from https://github.com/IBM/CrossViT/blob/main/models/crossvit.py#L132
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'
        x = x.view(B * group_size, N, C)

        for blk in self.vit_blks:
            x = blk(x)  # B 3 N C

        if self.fusion is None:
            return x.view(B, group_size, N, C)

        # outs_b = x.view(B, group_size, N,
        #                 C).chunk(chunks=3,
        #                          dim=1)  # 3 * [B, 1, N//3, C] Tensors, for fusion

        outs_b = x.chunk(chunks=3,
                         dim=0)  # 3 * [B, N//3, C] Tensors, for fusion

        # only take the cls token out
        proj_cls_token = [x[:, 0:1] for x in outs_b]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat(
                (proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:,
                                                                        ...]),
                dim=1)
            tmp = self.fusion[i](tmp)
            # reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            reverted_proj_cls_token = tmp[:, 0:1, ...]
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]),
                            dim=1)
            outs.append(tmp)
        # outs = ? needs to merge back?
        outs = torch.stack(outs, 1)  # B 3 N C
        return outs


class TriplaneFusionBlockv2(nn.Module):
    """4 ViT blocks + 1 CrossAttentionBlock
    """

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlock,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        if use_fusion_blk:
            # self.fusion = nn.ModuleList()

            # copied vit settings from https://github.dev/facebookresearch/dinov2
            nh = num_heads
            dim = embed_dim

            mlp_ratio = 4  # defined for all dino2 model
            qkv_bias = True
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            drop_path_rate = 0.3  # default setting
            attn_drop = proj_drop = 0.0
            qk_scale = None  # TODO, double check

            # for d in range(self.num_branches):
            self.fusion = fusion_ca_blk(  # one fusion is enough
                dim=dim,
                num_heads=nh,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                # drop=drop,
                drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,  # type: ignore
                has_mlp=False)
        else:
            self.fusion = None

    def forward(self, x):
        # modified from https://github.com/IBM/CrossViT/blob/main/models/crossvit.py#L132
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'
        x = x.reshape(B * group_size, N, C)

        for blk in self.vit_blks:
            x = blk(x)  # B 3 N C

        if self.fusion is None:
            return x.reshape(B, group_size, N, C)

        x = x.reshape(B, group_size, N, C)  # .chunk(chunks=3,
        #  dim=1)  # 3 * [B, N//3, C] Tensors, for fusion
        return self.fusion(x)


class TriplaneFusionBlockv3(TriplaneFusionBlockv2):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHA,
                 *args,
                 **kwargs) -> None:
        super().__init__(vit_blks, num_heads, embed_dim, use_fusion_blk,
                         fusion_ca_blk, *args, **kwargs)


class TriplaneFusionBlockv4(TriplaneFusionBlockv3):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHA,
                 *args,
                 **kwargs) -> None:
        super().__init__(vit_blks, num_heads, embed_dim, use_fusion_blk,
                         fusion_ca_blk, *args, **kwargs)
        """OOM? directly replace the atten here
        """

        assert len(vit_blks) == 2
        # del self.vit_blks[1].attn
        del self.vit_blks[1].attn, self.vit_blks[1].ls1, self.vit_blks[1].norm1

    def ffn_residual_func(self, tx_blk, x: Tensor) -> Tensor:
        return tx_blk.ls2(
            tx_blk.mlp(tx_blk.norm2(x))
        )  # https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/layers/block.py#L86C1-L87C53

    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """
        assert self.fusion is not None

        B, group_size, N, C = x.shape  # has [cls] token in N
        x = x.reshape(B * group_size, N, C)  # in plane vit

        # in plane self attention
        x = self.vit_blks[0](x)

        # 3D cross attention blk + ffn
        x = x + self.fusion(x.reshape(B, group_size, N, C)).reshape(
            B * group_size, N, C)
        x = x + self.ffn_residual_func(self.vit_blks[1], x)
        return x.reshape(B, group_size, N, C)


class TriplaneFusionBlockv4_nested(nn.Module):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        assert use_fusion_blk

        assert len(vit_blks) == 2

        # ! replace vit_blks[1] attn layer with 3D aware attention
        del self.vit_blks[
            1].attn  # , self.vit_blks[1].ls1, self.vit_blks[1].norm1

        # copied vit settings from https://github.dev/facebookresearch/dinov2
        nh = num_heads
        dim = embed_dim

        mlp_ratio = 4  # defined for all dino2 model
        qkv_bias = True
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_path_rate = 0.3  # default setting
        attn_drop = proj_drop = 0.0
        qk_scale = None  # TODO, double check

        self.vit_blks[1].attn = fusion_ca_blk(  # one fusion is enough
            dim=dim,
            num_heads=nh,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            # drop=drop,
            drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,  # type: ignore
            has_mlp=False)

    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'
        x = x.reshape(B * group_size, N, C)

        for blk in self.vit_blks:
            x = blk(x)  # B 3 N C

        # TODO, avoid the reshape overhead?
        return x.reshape(B, group_size, N, C)


class TriplaneFusionBlockv4_nested_init_from_dino(nn.Module):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested,
                 init_from_dino=True,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        assert use_fusion_blk

        assert len(vit_blks) == 2

        # copied vit settings from https://github.dev/facebookresearch/dinov2
        nh = num_heads
        dim = embed_dim

        mlp_ratio = 4  # defined for all dino2 model
        qkv_bias = True
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_path_rate = 0.3  # default setting
        attn_drop = proj_drop = 0.0
        qk_scale = None  # TODO, double check

        attn_3d = fusion_ca_blk(  # one fusion is enough
            dim=dim,
            num_heads=nh,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            # drop=drop,
            drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,  # type: ignore
            has_mlp=False)

        # ! initialize 3dattn from dino attn
        if init_from_dino:
            merged_qkv_linear = self.vit_blks[1].attn.qkv
            attn_3d.attn.proj.load_state_dict(
                self.vit_blks[1].attn.proj.state_dict())

            # Initialize the Q, K, and V linear layers using the weights of the merged QKV linear layer
            attn_3d.attn.wq.weight.data = merged_qkv_linear.weight.data[:
                                                                        dim, :]
            attn_3d.attn.w_kv.weight.data = merged_qkv_linear.weight.data[
                dim:, :]

            # Optionally, you can initialize the biases as well (if your QKV linear layer has biases)
            if qkv_bias:
                attn_3d.attn.wq.bias.data = merged_qkv_linear.bias.data[:dim]
                attn_3d.attn.w_kv.bias.data = merged_qkv_linear.bias.data[dim:]

        del self.vit_blks[1].attn
        # ! assign
        self.vit_blks[1].attn = attn_3d

    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'
        x = x.reshape(B * group_size, N, C)

        for blk in self.vit_blks:
            x = blk(x)  # B 3 N C

        # TODO, avoid the reshape overhead?
        return x.reshape(B, group_size, N, C)


class TriplaneFusionBlockv4_nested_init_from_dino_lite(nn.Module):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=None,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        assert use_fusion_blk

        assert len(vit_blks) == 2

        # copied vit settings from https://github.dev/facebookresearch/dinov2
        nh = num_heads
        dim = embed_dim

        mlp_ratio = 4  # defined for all dino2 model
        qkv_bias = True
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_path_rate = 0.3  # default setting
        attn_drop = proj_drop = 0.0
        qk_scale = None  # TODO, double check

        attn_3d = xformer_Conv3D_Aware_CrossAttention_xygrid_withinC(  # ! raw 3D attn layer
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop)

        del self.vit_blks[1].attn
        # ! assign
        self.vit_blks[1].attn = attn_3d

    def forward(self, x):
        """x: B N C, where N = H*W tokens. Just raw ViT forward pass
        """

        # ! move the below to the front of the first call
        B, N, C = x.shape  # has [cls] token in N

        for blk in self.vit_blks:
            x = blk(x)  # B N C

        return x

class TriplaneFusionBlockv4_nested_init_from_dino_lite_merge(nn.Module):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=None,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.vit_blks = vit_blks

        assert use_fusion_blk
        assert len(vit_blks) == 2

        # copied vit settings from https://github.dev/facebookresearch/dinov2
        nh = num_heads
        dim = embed_dim
        qkv_bias = True
        attn_drop = proj_drop = 0.0
        qk_scale = None  # TODO, double check

        if False: # abla
            for blk in self.vit_blks:
                attn_3d = xformer_Conv3D_Aware_CrossAttention_xygrid_withinC(  # ! raw 3D attn layer
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop)
                blk.attn = self_cross_attn(blk.attn, attn_3d)

    def forward(self, x):
        """x: B N C, where N = H*W tokens. Just raw ViT forward pass
        """

        # ! move the below to the front of the first call
        B, N, C = x.shape  # has [cls] token in N

        for blk in self.vit_blks:
            x = blk(x)  # B N C

        return x

class TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C(TriplaneFusionBlockv4_nested_init_from_dino_lite_merge):
    # on roll out + B 3L C
    def __init__(self, vit_blks, num_heads, embed_dim, use_fusion_blk=True, fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested, init_from_dino=True, *args, **kwargs) -> None:
        super().__init__(vit_blks, num_heads, embed_dim, use_fusion_blk, fusion_ca_blk, init_from_dino, *args, **kwargs)


    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # ! move the below to the front of the first call

        # B, N, C = x.shape  # has [cls] token in N
        B, group_size, N, C = x.shape  # has [cls] token in N
        x = x.reshape(B, group_size*N, C)

        for blk in self.vit_blks:
            x = blk(x)  # B N C

        x = x.reshape(B, group_size, N, C) # outer loop tradition

        return x

class TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout(TriplaneFusionBlockv4_nested_init_from_dino_lite_merge):
    # roll out + B 3L C
    def __init__(self, vit_blks, num_heads, embed_dim, use_fusion_blk=True, fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested, init_from_dino=True, *args, **kwargs) -> None:
        super().__init__(vit_blks, num_heads, embed_dim, use_fusion_blk, fusion_ca_blk, init_from_dino, *args, **kwargs)


    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # ! move the below to the front of the first call

        # B, N, C = x.shape  # has [cls] token in N
        B, group_size, N, C = x.shape  # has [cls] token in N
        x = x.reshape(B*group_size, N, C)
        x = self.vit_blks[0](x)

        x = x.reshape(B,group_size*N, C)
        x = self.vit_blks[1](x)

        x = x.reshape(B, group_size, N, C) # outer loop tradition

        return x


class TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_add3DAttn(TriplaneFusionBlockv4_nested_init_from_dino):
    # no roll out + 3D Attention
    def __init__(self, vit_blks, num_heads, embed_dim, use_fusion_blk=True, fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested, init_from_dino=True, *args, **kwargs) -> None:
        super().__init__(vit_blks, num_heads, embed_dim, use_fusion_blk, fusion_ca_blk, init_from_dino, *args, **kwargs)


    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        B, group_size, N, C = x.shape  # has [cls] token in N
        x = x.reshape(B, group_size*N, C)
        x = self.vit_blks[0](x) # B 3 L C

        # ! move the below to the front of the first call
        x = x.reshape(B, group_size, N, C).reshape(B*group_size, N, C)
        x = self.vit_blks[1](x) # has 3D attention
        return x.reshape(B, group_size, N, C)

        return x


class TriplaneFusionBlockv5_ldm_addCA(nn.Module):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        assert use_fusion_blk

        assert len(vit_blks) == 2

        # ! rather than replacing, add a 3D attention block after.
        # del self.vit_blks[
        #     1].attn  # , self.vit_blks[1].ls1, self.vit_blks[1].norm1
        self.norm_for_atten_3d = deepcopy(self.vit_blks[1].norm1)

        # copied vit settings from https://github.dev/facebookresearch/dinov2
        nh = num_heads
        dim = embed_dim

        mlp_ratio = 4  # defined for all dino2 model
        qkv_bias = True
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_path_rate = 0.3  # default setting
        attn_drop = proj_drop = 0.0
        qk_scale = None  # TODO, double check

        self.attn_3d = xformer_Conv3D_Aware_CrossAttention_xygrid(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop)

    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'

        flatten_token = lambda x: x.reshape(B * group_size, N, C)
        unflatten_token = lambda x: x.reshape(B, group_size, N, C)

        x = flatten_token(x)
        x = self.vit_blks[0](x)

        x = unflatten_token(x)
        x = self.attn_3d(self.norm_for_atten_3d(x)) + x

        x = flatten_token(x)
        x = self.vit_blks[1](x)

        return unflatten_token(x)


class TriplaneFusionBlockv6_ldm_addCA_Init3DAttnfrom2D(
        TriplaneFusionBlockv5_ldm_addCA):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested,
                 *args,
                 **kwargs) -> None:
        super().__init__(vit_blks, num_heads, embed_dim, use_fusion_blk,
                         fusion_ca_blk, *args, **kwargs)

    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'

        flatten_token = lambda x: x.reshape(B * group_size, N, C)
        unflatten_token = lambda x: x.reshape(B, group_size, N, C)

        x = flatten_token(x)
        x = self.vit_blks[0](x)

        x = unflatten_token(x)
        x = self.attn_3d(self.norm_for_atten_3d(x)) + x

        x = flatten_token(x)
        x = self.vit_blks[1](x)

        return unflatten_token(x)


class TriplaneFusionBlockv5_ldm_add_dualCA(nn.Module):

    def __init__(self,
                 vit_blks,
                 num_heads,
                 embed_dim,
                 use_fusion_blk=True,
                 fusion_ca_blk=Conv3DCrossAttentionBlockXformerMHANested,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.num_branches = 3  # triplane
        self.vit_blks = vit_blks

        assert use_fusion_blk

        assert len(vit_blks) == 2

        # ! rather than replacing, add a 3D attention block after.
        # del self.vit_blks[
        #     1].attn  # , self.vit_blks[1].ls1, self.vit_blks[1].norm1
        self.norm_for_atten_3d_0 = deepcopy(self.vit_blks[0].norm1)
        self.norm_for_atten_3d_1 = deepcopy(self.vit_blks[1].norm1)

        # copied vit settings from https://github.dev/facebookresearch/dinov2
        nh = num_heads
        dim = embed_dim

        mlp_ratio = 4  # defined for all dino2 model
        qkv_bias = True
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_path_rate = 0.3  # default setting
        attn_drop = proj_drop = 0.0
        qk_scale = None  # TODO, double check

        self.attn_3d_0 = xformer_Conv3D_Aware_CrossAttention_xygrid(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop)

        self.attn_3d_1 = deepcopy(self.attn_3d_0)

    def forward(self, x):
        """x: B 3 N C, where N = H*W tokens
        """

        # self attention, by merging the triplane channel into B for parallel computation

        # ! move the below to the front of the first call
        B, group_size, N, C = x.shape  # has [cls] token in N
        assert group_size == 3, 'triplane'

        flatten_token = lambda x: x.reshape(B * group_size, N, C)
        unflatten_token = lambda x: x.reshape(B, group_size, N, C)

        x = flatten_token(x)
        x = self.vit_blks[0](x)

        x = unflatten_token(x)
        x = self.attn_3d_0(self.norm_for_atten_3d_0(x)) + x

        x = flatten_token(x)
        x = self.vit_blks[1](x)

        x = unflatten_token(x)
        x = self.attn_3d_1(self.norm_for_atten_3d_1(x)) + x

        return unflatten_token(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim,
        self.attn = MemEffAttention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, C, L -> B, L, C
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self,
                 img_size=[224],
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 patch_embedding=True,
                 cls_token=True,
                 pixel_unshuffle=False,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        # if norm_layer == 'nn.LayerNorm':
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if patch_embedding:
            self.patch_embed = PatchEmbed(img_size=img_size[0],
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches
            self.img_size = self.patch_embed.img_size
        else:
            self.patch_embed = None
            self.img_size = img_size[0]
            num_patches = (img_size[0] // patch_size) * (img_size[0] //
                                                         patch_size)

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        if cls_token:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # if pixel_unshuffle:
        #     self.decoder_pred = nn.Linear(embed_dim,
        #                                 patch_size**2 * out_chans,
        #                                 bias=True)  # decoder to patch

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(
            h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(2, -1, dim)

        if self.cls_token is not None:
            class_pos_embed = self.pos_embed[:, 0]
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                             dim=1)
        return patch_pos_embed

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]  # return spatial feature maps, not the [CLS] token
        # return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size,
                              embed_dim=192,
                              depth=12,
                              num_heads=3,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # type: ignore
        **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model


vits = vit_small
vitb = vit_base
