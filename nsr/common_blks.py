

# https://github.com/sxyu/pixel-nerf/blob/master/src/model/resnetfc.py
from torch import nn
import torch

from vit.vision_transformer import Mlp, DropPath


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """
    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, init_as_zero=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        if init_as_zero:
            nn.init.zeros_(self.fc_0.weight)
        else:
            nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            # nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        # with profiler.record_function("resblock"):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx




# Resnet Blocks
class ResnetBlockFCViT(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """
    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, init_as_zero=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        if init_as_zero:
            nn.init.zeros_(self.fc_0.weight)
        else:
            nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            # nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        # with profiler.record_function("resblock"):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim,
#                               num_heads=num_heads,
#                               qkv_bias=qkv_bias,
#                               qk_scale=qk_scale,
#                               attn_drop=attn_drop,
#                               proj_drop=drop)
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim,
#                        hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer,
#                        drop=drop)

#     def forward(self, x, return_attention=False):
#         y, attn = self.attn(self.norm1(x))
#         if return_attention:
#             return attn
#         x = x + self.drop_path(y)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x




class ResMlp(nn.Module):
    def __init__(self,
                 
                size_in,
                size_out=None,
                size_h=None,
                drop=0.,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        # Attributes
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.norm1 = norm_layer(size_in) # ? how to use

        self.mlp = Mlp(in_features=size_in, 
                       out_features=size_out,
                       act_layer=act_layer,
                       drop=drop)

        # Residual shortcuts
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            self.norm2 = norm_layer(size_in)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        dx = self.mlp(self.norm1(x))

        if self.shortcut is not None:
            x_s = self.shortcut(self.norm2(x))
        else:
            x_s = x

        return x_s + self.drop_path(dx)