from typing import List, Optional, Tuple

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from packaging import version
from pdb import set_trace as st

from ldm.modules.diffusionmodules.util import checkpoint

# from torch.nn import LayerNorm
try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except:
    from dit.norm import RMSNorm


# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp, MemoryEfficientAttentionCutlassOp
# from xformers.ops import RMSNorm, fmha, rope_padded
# import apex
# from apex.normalization import FusedRMSNorm as RMSNorm

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    # from torch.backends.cuda import SDPBackend, sdp_kernel
    from torch.nn.attention import sdpa_kernel, SDPBackend

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdpa_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        # backend=None,
        backend=SDPBackend.FLASH_ATTENTION, # FA implemented by torch.
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        # with sdpa_kernel(**BACKEND_MAP[self.backend]):
        with sdpa_kernel([self.backend]): # new signature
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)

# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context=None, mask=None):
#         h = self.heads

#         q = self.to_q(x)
#         context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
#         attn = sim.softmax(dim=-1)

#         out = einsum('b i j, b j d -> b i d', attn, v)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         return self.to_out(out)


# class BasicTransformerBlock(nn.Module):
#     def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
#         super().__init__()
#         self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
#         self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
#                                     heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.checkpoint = checkpoint

#     def forward(self, x, context=None):
#         return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

#     def _forward(self, x, context=None):
#         x = self.attn1(self.norm1(x)) + x
#         x = self.attn2(self.norm2(x), context=context) + x
#         x = self.ff(self.norm3(x)) + x
#         return x


try:
    # from xformers.triton import FusedLayerNorm as LayerNorm
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

from typing import Optional, Any

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, enable_rmsnorm=False, qk_norm=False, no_flash_op=False, enable_rope=False, qk_norm_fullseq=False,):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.") 
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)

        self.enable_rope = enable_rope

        # if enable_rmsnorm:
        # self.q_rmsnorm = RMSNorm(query_dim, eps=1e-5)
        # self.k_rmsnorm = RMSNorm(context_dim, eps=1e-5)

        if qk_norm_fullseq: # as in lumina
            self.q_norm = RMSNorm(inner_dim, elementwise_affine=True) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(inner_dim, elementwise_affine=True) if qk_norm else nn.Identity()
        else:
            self.q_norm = RMSNorm(self.dim_head, elementwise_affine=True) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(self.dim_head, elementwise_affine=True) if qk_norm else nn.Identity()

        # if not qk_norm:
        #     logpy.warn(
        #         f"No QK Norm activated, wish you good luck..."
        #     )

        # self.enable_rmsnorm = enable_rmsnorm

        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        if no_flash_op:
            self.attention_op = MemoryEfficientAttentionCutlassOp # force flash attention
        else:
            self.attention_op: Optional[Any] = None # enable 

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
        assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
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
            freqs_cis = MemoryEfficientCrossAttention.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)


    def forward(self, x, context=None, freqs_cis=None, mask=None):

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)

        v = self.to_v(context)

        dtype = q.dtype

        b, _, _ = q.shape
        if self.enable_rope:
            q, k = self.q_norm(q), self.k_norm(k) # for stable amp training

            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                # .reshape(b * self.heads, t.shape[1], self.dim_head)
                .reshape(b, self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            assert freqs_cis is not None
            q, k = MemoryEfficientCrossAttention.apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q, k = q.to(dtype), k.to(dtype)
            pass

        else:
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )
            q, k = self.q_norm(q), self.k_norm(k) # for stable amp training

        # actually compute the attention, what we cannot get enough of
        # out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        # op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)



class JointMemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv_t = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_qkv_i = nn.Linear(query_dim, inner_dim, bias=False)

        # self.to_k = nn.Linear(context_dim*2, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim*2, inner_dim, bias=False)
        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None
        # self.attention_op: Optional[Any] = MemoryEfficientAttentionFlashAttentionOp

        # TODO, add later for stable AMP training.
        # self.rms_norm_t_q = RMSNorm(args.dim, eps=args.norm_eps)
        # self.rms_norm_t_k = RMSNorm(args.dim, eps=args.norm_eps)
        # self.rms_norm_i_q = RMSNorm(args.dim, eps=args.norm_eps)
        # self.rms_norm_i_k = RMSNorm(args.dim, eps=args.norm_eps)


    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        return self._forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



class BasicTransformerBlock3D(BasicTransformerBlock):

    def forward(self, x, context=None, num_frames=1):
        # return checkpoint(self._forward, (x, context, num_frames), self.parameters(), self.checkpoint)
        return self._forward(x, context, num_frames) # , self.parameters(), self.checkpoint

    def _forward(self, x, context=None, num_frames=1):
        x = rearrange(x, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = rearrange(x, "b (f l) c -> (b f) l c", f=num_frames).contiguous()
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer3D(nn.Module):
    ''' 3D self-attention ''' 
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        elif context_dim is None:
            context_dim = [None] * depth

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock3D(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, num_frames=1):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], num_frames=num_frames)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in