import math
import random
import kiui
from kiui.op import recenter
import torchvision
import torchvision.transforms.v2
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
from pdb import set_trace as st

import kornia
import numpy as np
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (ByT5Tokenizer, CLIPTextModel, CLIPTokenizer,
                          T5EncoderModel, T5Tokenizer)

from ...modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from ...modules.diffusionmodules.model import Encoder
from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.diffusionmodules.util import (extract_into_tensor,
                                              make_beta_schedule)
from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...util import (append_dims, autocast, count_params, default,
                     disabled_train, expand_dims_like, instantiate_from_config)

from dit.dit_models_xformers import CaptionEmbedder, approx_gelu, t2i_modulate


class AbstractEmbModel(nn.Module):

    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel,
                             batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(self,
                batch: Dict,
                force_zero_embeddings: Optional[List] = None) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key
                                                       is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(
                        *[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                if embedder.input_key in ('caption', 'img'):
                    out_key = f'{embedder.input_key}_{self.OUTPUT_DIM2KEYS[emb.dim()]}'
                elif emb.dim()==3 and emb.shape[-1] == 3:
                    out_key = 'fps-xyz'
                else:
                    out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (expand_dims_like(
                        torch.bernoulli(
                            (1.0 - embedder.ucg_rate) *
                            torch.ones(emb.shape[0], device=emb.device)),
                        emb,
                    ) * emb)
                if (hasattr(embedder, "input_key")
                        and embedder.input_key in force_zero_embeddings):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb),
                                                self.KEY2CATDIM[out_key.split('_')[1]])
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0 # ! force no drop during inference
        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc,
                  force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class InceptionV3(nn.Module):
    """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
    port with an additional squeeze at the end"""

    def __init__(self, normalize_input=False, **kwargs):
        super().__init__()
        from pytorch_fid import inception

        kwargs["resize_input"] = True
        self.model = inception.InceptionV3(normalize_input=normalize_input,
                                           **kwargs)

    def forward(self, inp):
        outp = self.model(inp)

        if len(outp) == 1:
            return outp[0].squeeze()

        return outp


class IdentityEncoder(AbstractEmbModel):

    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):

    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs, ), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):

    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(self,
                 version="google/t5-v1_1-xxl",
                 device="cuda",
                 max_length=77,
                 freeze=True
                 ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(self,
                 version="google/byt5-base",
                 device="cuda",
                 max_length=77,
                 freeze=True
                 ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens,
                                   output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (x[torch.arange(x.shape[0]),
               text.argmax(dim=-1)] @ self.model.text_projection)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (self.model.transformer.grad_checkpointing
                    and not torch.jit.is_scripting()):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (self.model.transformer.grad_checkpointing
                    and not torch.jit.is_scripting()):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        # arch="ViT-H-14",
        # version="laion2b_s32b_b79k",
        arch="ViT-L-14",
        # version="laion2b_s32b_b82k",
        version="openai",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        init_device=None,
        inp_size=224,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(default(init_device, "cpu")),
            pretrained=version,
        )
        del model.transformer
        self.inp_size = inp_size
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (
            not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer("mean",
                             torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                             persistent=False)
        self.register_buffer("std",
                             torch.Tensor([0.26862954, 0.26130258,
                                           0.27577711]),
                             persistent=False)
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens
        self.interpolate_offset = 0.0
        self.patch_size = 14
        npatch = (self.inp_size // self.patch_size) ** 2
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/configs/eval/vitl14_reg4_pretrain.yaml#L5
        self.interpolate_antialias = True

        if self.inp_size != 224:
            self.model.visual.positional_embedding = torch.nn.Parameter(self.interpolate_pos_encoding(npatch, self.inp_size, self.inp_size) )

    # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179
    def interpolate_pos_encoding(self, npatch, w, h):
        dim = self.model.visual.positional_embedding.shape[-1]

        # previous_dtype = x.dtype
        previous_dtype = torch.float32
        # npatch = x.shape[1] - 1

        pos_embed = self.model.visual.positional_embedding.float().unsqueeze(0)
        N = pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.model.visual.positional_embedding

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        # dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)[0]

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (self.inp_size, self.inp_size),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (torch.bernoulli(
                (1.0 - self.ucg_rate) *
                torch.ones(z.shape[0], device=z.device))[:, None] * z)
            if tokens is not None:
                tokens = (expand_dims_like(
                    torch.bernoulli(
                        (1.0 - self.ucg_rate) *
                        torch.ones(tokens.shape[0], device=tokens.device)),
                    tokens,
                ) * tokens)
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (torch.bernoulli(
                (1.0 - self.ucg_rate) *
                torch.ones(x.shape[0], x.shape[1], 1, device=x.device)) * x)
            if tokens is not None:
                tokens = rearrange(tokens,
                                   "(b n) t d -> b t (n d)",
                                   n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message.")
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)


# dino-v2 embedder
class FrozenDinov2ImageEmbedder(AbstractEmbModel):
    """
    Uses the Dino-v2 for low-level image embedding
    """

    def __init__(
        self,
        arch="vitl",
        version="dinov2",  # by default
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        output_cls=False,
        init_device=None,
        inp_size=224,
    ):
        super().__init__()

        self.model = torch.hub.load(
            f'facebookresearch/{version}',
            '{}_{}{}_reg'.format(
                version, f'{arch}', '14'
            ),  # with registers better performance. vitl and vitg similar. Since fixed, load the best one.
            pretrained=True).to(torch.device(default(init_device, "cpu")))
        
        # print(self.model)

        # ! frozen
        # self.tokenizer.requires_grad_(False)
        # self.tokenizer.eval()

        # assert freeze # add adaLN here
        self.inp_size = inp_size
        if freeze:
            self.freeze()

        # self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (
            not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length

        self.antialias = antialias

        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L41
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        self.register_buffer("mean",
                             torch.Tensor(IMAGENET_DEFAULT_MEAN),
                             persistent=False)
        self.register_buffer("std",
                             torch.Tensor(IMAGENET_DEFAULT_STD),
                             persistent=False)

        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        # self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens  # output
        self.output_cls = output_cls
        # self.output_tokens = False

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            # (224, 224),
            (self.inp_size, self.inp_size),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _model_forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def encode_with_vision_transformer(self, img, **kwargs):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            # assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)

        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L326
        if not self.output_cls:
            return self._model_forward(
                img, is_training=True,
                **kwargs)['x_norm_patchtokens']  # to return spatial tokens

        else:
            dino_ret_dict = self._model_forward(
                img, is_training=True)  # to return spatial tokens
            x_patchtokens, x_norm_clstoken = dino_ret_dict[
                'x_norm_patchtokens'], dino_ret_dict['x_norm_clstoken']

            return x_norm_clstoken, x_patchtokens

    @autocast
    def forward(self, image, no_dropout=False, **kwargs):
        tokens = self.encode_with_vision_transformer(image, **kwargs)
        z = None
        if self.output_cls:
            # z, tokens = z[0], z[1]
            z, tokens = tokens[0], tokens[1]
            z = z.to(image.dtype)
        tokens = tokens.to(image.dtype)  # ! return spatial tokens only
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            if z is not None:
                z = (torch.bernoulli(
                    (1.0 - self.ucg_rate) *
                    torch.ones(z.shape[0], device=z.device))[:, None] * z)
            tokens = (expand_dims_like(
                torch.bernoulli(
                    (1.0 - self.ucg_rate) *
                    torch.ones(tokens.shape[0], device=tokens.device)),
                tokens,
            ) * tokens)
        if self.output_cls:
            return tokens, z
        else:
            return tokens


class FrozenDinov2ImageEmbedderMVPlucker(FrozenDinov2ImageEmbedder):

    def __init__(
        self,
        arch="vitl",
        version="dinov2",  # by default
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        output_cls=False,
        init_device=None,
        # mv cond settings
        n_cond_frames=4,  # numebr of condition views
        enable_bf16=False,
        modLN=False,
        aug_c=False,
        inp_size=224,
    ):
        super().__init__(
            arch,
            version,
            device,
            max_length,
            freeze,
            antialias,
            ucg_rate,
            unsqueeze_dim,
            repeat_to_max_len,
            num_image_crops,
            output_tokens,
            output_cls,
            init_device,
            inp_size=inp_size,
        )
        self.n_cond_frames = n_cond_frames
        self.dtype = torch.bfloat16 if enable_bf16 else torch.float32
        self.enable_bf16 = enable_bf16
        self.aug_c = aug_c

        # ! proj c_cond to features

        self.reso_encoder = inp_size
        orig_patch_embed_weight = self.model.patch_embed.state_dict()

        # ! 9-d input
        with torch.no_grad():
            new_patch_embed = PatchEmbed(img_size=224,
                                        patch_size=14,
                                        in_chans=9,
                                        embed_dim=self.model.embed_dim)
            # zero init first
            nn.init.constant_(new_patch_embed.proj.weight, 0)
            nn.init.constant_(new_patch_embed.proj.bias, 0)
            # load pre-trained first 3 layers weights, bias into the new patch_embed

            new_patch_embed.proj.weight[:, :3].copy_(orig_patch_embed_weight['proj.weight'])
            new_patch_embed.proj.bias[:].copy_(orig_patch_embed_weight['proj.bias'])

        self.model.patch_embed = new_patch_embed  # xyz in the front
        # self.scale_jitter_aug = torchvision.transforms.v2.ScaleJitter(target_size=(self.reso_encoder, self.reso_encoder), scale_range=(0.5, 1.5))

    @autocast
    def scale_jitter_aug(self, x):
        inp_size = x.shape[2]
        # aug_size = torch.randint(low=50, high=100, size=(1,)) / 100 * inp_size
        aug_size = int(max(0.5, random.random()) * inp_size)
        # st()
        x = torch.nn.functional.interpolate(x,
                                            size=aug_size,
                                            mode='bilinear', 
                                            antialias=True)
        x = torch.nn.functional.interpolate(x,size=inp_size,
                                            mode='bilinear', antialias=True)
        return x

    @autocast
    def gen_rays(self, c):
        # Generate rays
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32, device=c.device) + 0.5,
            torch.arange(self.w, dtype=torch.float32, device=c.device) + 0.5,
            indexing='ij')

        # normalize to 0-1 pixel range
        yy = yy / self.h
        xx = xx / self.w

        # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
            0], intrinsics[4]
        # cx *= self.w
        # cy *= self.h

        # f_x = f_y = fx * h / res_raw
        # c2w = torch.from_numpy(c2w).float()
        c2w = c2w.float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        # st()
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    @autocast
    def get_plucker_ray(self, c):
        rays_plucker = []
        for idx in range(c.shape[0]):
            rays_o, rays_d = self.gen_rays(c[idx])
            rays_plucker.append(
                torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                          dim=-1).permute(2, 0, 1))  # [h, w, 6] -> 6,h,w
        rays_plucker = torch.stack(rays_plucker, 0)
        return rays_plucker

    @autocast
    def _model_forward(self, x, plucker_c, *args, **kwargs):

        with torch.cuda.amp.autocast(dtype=self.dtype, enabled=True):
            x = torch.cat([x, plucker_c], dim=1).to(self.dtype)
            return self.model(x, **kwargs)

    def preprocess(self, x):
        # add gaussian noise and rescale augmentation

        if self.ucg_rate > 0.0:

            # 1 means maintain the input x
            enable_drop_flag = torch.bernoulli(
                (1.0 - self.ucg_rate) *
                torch.ones(x.shape[0], device=x.device))[:, None, None, None] # broadcast to B,1,1,1

            # * add random downsample & upsample
            # rescaled_x = self.downsample_upsample(x)
            # torchvision.utils.save_image(x, 'tmp/x.png', normalize=True, value_range=(-1,1))
            x_aug = self.scale_jitter_aug(x)
            # torchvision.utils.save_image(x_aug, 'tmp/rescale-x.png', normalize=True, value_range=(-1,1))

            # x_aug = x * enable_drop_flag + (1-enable_drop_flag) * x_aug

            # * guassian noise jitter
            # force linear_weight > 0.24
            # linear_weight = torch.max(enable_drop_flag, torch.max(torch.rand_like(enable_drop_flag), 0.25 * torch.ones_like(enable_drop_flag), dim=0, keepdim=True), dim=0, keepdim=True)
            gaussian_jitter_scale, jitter_lb = torch.rand_like(enable_drop_flag), 0.8 * torch.ones_like(enable_drop_flag)           
            gaussian_jitter_scale = torch.where(gaussian_jitter_scale>jitter_lb, gaussian_jitter_scale, jitter_lb)

            # torchvision.utils.save_image(x, 'tmp/aug-x.png', normalize=True, value_range=(-1,1))
            x_aug = gaussian_jitter_scale * x_aug + (1 - gaussian_jitter_scale) * torch.randn_like(x).clamp(-1,1)

            x_aug = x * enable_drop_flag + (1-enable_drop_flag) * x_aug
            # torchvision.utils.save_image(x_aug, 'tmp/final-x.png', normalize=True, value_range=(-1,1))

        # st()

        return super().preprocess(x)

    def random_rotate_c(self, c):

        intrinsics, c2ws = c[16:], c[:16].reshape(4, 4)

        # https://github.com/TencentARC/InstantMesh/blob/34c193cc96eebd46deb7c48a76613753ad777122/src/data/objaverse.py#L195

        degree = np.random.uniform(-math.pi * 0.25, math.pi * 0.25)

        # random rotation along z axis
        if random.random() > 0.5:
            rot = torch.tensor([
                [np.cos(degree), -np.sin(degree), 0, 0],
                [np.sin(degree), np.cos(degree), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).to(c2ws)
        else:
            # random rotation along y axis
            rot = torch.tensor([
                [np.cos(degree), 0, np.sin(degree),  0],
                [0, 1, 0, 0],
                [-np.sin(degree), 0, np.cos(degree), 0],
                [0, 0, 0, 1],
            ]).to(c2ws)

        c2ws = torch.matmul(rot, c2ws)

        return torch.cat([c2ws.reshape(-1), intrinsics])

    @autocast
    def forward(self, img_c, no_dropout=False):


        mv_image, c = img_c['img'], img_c['c']

        if self.aug_c:
            for idx_b in range(c.shape[0]):
                for idx_v in range(c.shape[1]):
                    if random.random() > 0.8:
                        c[idx_b, idx_v] = self.random_rotate_c(c[idx_b, idx_v])

        # plucker_c = self.get_plucker_ray(
        #     rearrange(c[:, 1:1 + self.n_cond_frames], "b t ... -> (b t) ..."))
        plucker_c = self.get_plucker_ray(
            rearrange(c[:, :self.n_cond_frames], "b t ... -> (b t) ..."))
        
        # plucker_c = torch.ones_like(plucker_c)
        # plucker_c = torch.zeros_like(plucker_c)

        # mv_image_tokens = super().forward(mv_image[:, 1:1 + self.n_cond_frames],
        mv_image_tokens = super().forward(mv_image[:, :self.n_cond_frames],
                                plucker_c=plucker_c, 
                                no_dropout=no_dropout)

        mv_image_tokens = rearrange(mv_image_tokens,
                            "(b t) ... -> b t ...",
                            t=self.n_cond_frames)

        return mv_image_tokens

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_HW,
                              stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class FrozenDinov2ImageEmbedderMV(FrozenDinov2ImageEmbedder):

    def __init__(
        self,
        arch="vitl",
        version="dinov2",  # by default
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        output_cls=False,
        init_device=None,
        # mv cond settings
        n_cond_frames=4,  # numebr of condition views
        enable_bf16=False,
        modLN=False,
        inp_size=224,
    ):
        super().__init__(
            arch,
            version,
            device,
            max_length,
            freeze,
            antialias,
            ucg_rate,
            unsqueeze_dim,
            repeat_to_max_len,
            num_image_crops,
            output_tokens,
            output_cls,
            init_device,
            inp_size=inp_size,
        )
        self.n_cond_frames = n_cond_frames
        self.dtype = torch.bfloat16 if enable_bf16 else torch.float32
        self.enable_bf16 = enable_bf16

        # ! proj c_cond to features

        hidden_size = self.model.embed_dim  # 768 for vit-b

        # self.cam_proj = CaptionEmbedder(16, hidden_size,
        self.cam_proj = CaptionEmbedder(25, hidden_size, act_layer=approx_gelu)

        # ! single-modLN
        self.model.modLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))

        # zero-init modLN
        nn.init.constant_(self.model.modLN_modulation[-1].weight, 0)
        nn.init.constant_(self.model.modLN_modulation[-1].bias, 0)

        # inject modLN to dino block
        for block in self.model.blocks:
            block.scale_shift_table = nn.Parameter(torch.zeros(
                4, hidden_size))  # zero init also

            # torch.randn(4, hidden_size) / hidden_size**0.5)

    def _model_forward(self, x, *args, **kwargs):
        # re-define model forward, finetune dino-v2.
        assert self.training

        # ? how to send in camera
        # c = 0 # placeholder
        # ret = self.model.forward_features(*args, **kwargs)

        with torch.cuda.amp.autocast(dtype=self.dtype, enabled=True):

            x = self.model.prepare_tokens_with_masks(x, masks=None)

            B, N, C = x.shape
            # TODO how to send in c
            # c = torch.ones(B, 25).to(x) # placeholder
            c = kwargs.get('c')
            c = self.cam_proj(c)
            cond = self.model.modLN_modulation(c)

            # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/block.py#L89
            for blk in self.model.blocks:  # inject modLN

                shift_msa, scale_msa, shift_mlp, scale_mlp = (
                    blk.scale_shift_table[None] +
                    cond.reshape(B, 4, -1)).chunk(4, dim=1)

                def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
                    # return blk.ls1(blk.attn(blk.norm1(x), attn_bias=attn_bias))
                    return blk.ls1(
                        blk.attn(
                            t2i_modulate(blk.norm1(x), shift_msa, scale_msa)))

                def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
                    # return blk.ls2(blk.mlp(blk.norm2(x)))
                    return blk.ls2(
                        t2i_modulate(blk.mlp(blk.norm2(x)), shift_mlp,
                                     scale_mlp))

                x = x + blk.drop_path1(
                    attn_residual_func(x))  # all drop_path identity() here.
                x = x + blk.drop_path2(ffn_residual_func(x))

            x_norm = self.model.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            # "x_norm_regtokens": x_norm[:, 1 : self.model.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:,
                                         self.model.num_register_tokens + 1:],
            # "x_prenorm": x,
            # "masks": masks,
        }

    @autocast
    def forward(self, img_c, no_dropout=False):

        # if self.enable_bf16:
        #     with th.cuda.amp.autocast(dtype=self.dtype,
        #                               enabled=True):
        # mv_image = super().forward(mv_image[:, 1:1+self.n_cond_frames].to(torch.bf16))
        # else:
        mv_image, c = img_c['img'], img_c['c']

        # ! use zero c here, ablation. current verison wrong.
        # c = torch.zeros_like(c)

        # ! frame-0 as canonical here.

        mv_image = super().forward(mv_image[:, :self.n_cond_frames],
                                   c=rearrange(c[:, :self.n_cond_frames],
                                               "b t ... -> (b t) ...",
                                               t=self.n_cond_frames), 
                                               no_dropout=no_dropout)

        mv_image = rearrange(mv_image,
                             "(b t) ... -> b t ...",
                             t=self.n_cond_frames)

        return mv_image


class FrozenCLIPT5Encoder(AbstractEmbModel):

    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="cuda",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version,
                                               device,
                                               max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version,
                                           device,
                                           max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SpatialRescaler(nn.Module):

    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
        wrap_video=False,
        kernel_size=1,
        remap_output=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate,
                                    mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video

    def forward(self, x):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> b t c h w")
            x = rearrange(x, "b t c h w -> (b t) c h w")

        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)
            x = rearrange(x, "b t c h w -> b c t h w")
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class LowScaleEncoder(nn.Module):

    def __init__(
        self,
        model_config,
        linear_start,
        linear_end,
        timesteps=1000,
        max_noise_level=250,
        output_size=64,
        scale_factor=1.0,
    ):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end)
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps, ) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (alphas_cumprod.shape[0] == self.num_timesteps
                ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev",
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod",
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod",
                             to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod",
                             to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod",
                             to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                          t, x_start.shape) * noise)

    def forward(self, x):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(0,
                                    self.max_noise_level, (x.shape[0], ),
                                    device=x.device).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z,
                                                size=self.out_size,
                                                mode="nearest")
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb,
                        "(b d) d2 -> b (d d2)",
                        b=b,
                        d=dims,
                        d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):

    def __init__(self,
                 weight: float = 1.0,
                 flatten_output: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) -> Tuple[Dict, torch.Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        return log, z


class VideoPredictionEmbedderWithEncoder(AbstractEmbModel):

    def __init__(
        self,
        n_cond_frames: int,
        n_copies: int,
        encoder_config: dict,
        sigma_sampler_config: Optional[dict] = None,
        sigma_cond_config: Optional[dict] = None,
        is_ae: bool = False,
        scale_factor: float = 1.0,
        disable_encoder_autocast: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.encoder = instantiate_from_config(encoder_config)
        self.sigma_sampler = (instantiate_from_config(sigma_sampler_config)
                              if sigma_sampler_config is not None else None)
        self.sigma_cond = (instantiate_from_config(sigma_cond_config)
                           if sigma_cond_config is not None else None)
        self.is_ae = is_ae
        self.scale_factor = scale_factor
        self.disable_encoder_autocast = disable_encoder_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def forward(
        self, vid: torch.Tensor
    ) -> Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, dict],
            Tuple[Tuple[torch.Tensor, torch.Tensor], dict],
    ]:
        if self.sigma_sampler is not None:
            b = vid.shape[0] // self.n_cond_frames
            sigmas = self.sigma_sampler(b).to(vid.device)
            if self.sigma_cond is not None:
                sigma_cond = self.sigma_cond(sigmas)
                sigma_cond = repeat(sigma_cond,
                                    "b d -> (b t) d",
                                    t=self.n_copies)
            sigmas = repeat(sigmas, "b -> (b t)", t=self.n_cond_frames)
            noise = torch.randn_like(vid)
            vid = vid + noise * append_dims(sigmas, vid.ndim)

        with torch.autocast("cuda", enabled=not self.disable_encoder_autocast):
            n_samples = (self.en_and_decode_n_samples_a_time
                         if self.en_and_decode_n_samples_a_time is not None
                         else vid.shape[0])
            n_rounds = math.ceil(vid.shape[0] / n_samples)
            all_out = []
            for n in range(n_rounds):
                if self.is_ae:
                    out = self.encoder.encode(vid[n * n_samples:(n + 1) *
                                                  n_samples])
                else:
                    out = self.encoder(vid[n * n_samples:(n + 1) * n_samples])
                all_out.append(out)

        vid = torch.cat(all_out, dim=0)
        vid *= self.scale_factor

        vid = rearrange(vid,
                        "(b t) c h w -> b () (t c) h w",
                        t=self.n_cond_frames)
        vid = repeat(vid, "b 1 c h w -> (b t) c h w", t=self.n_copies)

        return_val = (vid, sigma_cond) if self.sigma_cond is not None else vid

        return return_val


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):

    def __init__(
        self,
        open_clip_embedding_config: Dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def forward(self, vid):
        vid = self.open_clip(vid)
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)

        return vid


class FrozenOpenCLIPImageMVEmbedder(AbstractEmbModel):
    # for multi-view 3D diffusion condition. Only extract the first frame
    def __init__(
        self,
        open_clip_embedding_config: Dict,
        # n_cond_frames: int,
        # n_copies: int,
    ):
        super().__init__()

        # self.n_cond_frames = n_cond_frames
        # self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def forward(self, vid, no_dropout=False):
        # st()
        vid = self.open_clip(vid[:, 0, ...], no_dropout=no_dropout)
        # vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        # vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)

        return vid

# process PCD

# raw scaling
class PCD_Scaler(AbstractEmbModel):
    """
    just scale the input pcd
    TODO, add some rand noise?
    """

    def __init__(
        self,
        scaling_factor=0.45,
        perturb_pcd_scale=0.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.perturb_pcd_scale = perturb_pcd_scale

    @autocast
    def forward(self, pcd, **kwargs):
        if self.perturb_pcd_scale > 0:
            t = torch.rand(pcd.shape[0], 1, 1).to(pcd) * self.perturb_pcd_scale
            pcd = pcd + t * torch.randn_like(pcd)
            pcd = pcd.clip(-0.45, 0.45) # avoid scaling xyz too large.
        pcd = pcd / self.scaling_factor
        return pcd



# raw scaling
class PCD_Scaler_perChannel(AbstractEmbModel):
    """
    scale the input pcd to unit std
    """

    def __init__(
        self,
        scaling_factor=[0.14593576, 0.15753542, 0.18873914],
    ):
        super().__init__()
        self.scaling_factor = np.array(scaling_factor)

    @autocast
    def forward(self, pcd, **kwargs):
        return pcd / self.scaling_factor