import numpy as np
import torch

from modules.diffusionmodules.model import SimpleDecoder, Decoder

# decoder = SimpleDecoder(4,32)
# https://github.com/CompVis/stable-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml
decoder = Decoder(
    ch=64,
    out_ch=32,
    ch_mult=(1, 2),
    num_res_blocks=2,
    # num_res_blocks=1,
    dropout=0.0,
    attn_resolutions=(),
    z_channels=4,
    resolution=64,
    in_channels=3,
).cuda()

input_tensor = torch.randn(1,4,32,32,).cuda()
# input_tensor = torch.randn(
#     1,
#     96,
#     32,
#     32,
# )

print(decoder(input_tensor).shape)
