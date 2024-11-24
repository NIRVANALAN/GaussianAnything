import torch
import torch.nn as nn
import torchvision


EPS = 1e-7

class ConfNet(nn.Module):
    def __init__(self, cin=3, cout=1, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        # ! only the symmetric confidence is required
        # out_net1 = [
        #     nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
        #     nn.GroupNorm(16, nf),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
        #     nn.GroupNorm(16, nf),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
        #     # nn.Conv2d(nf, 1, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
        #     nn.Softplus()
        # ]
        # self.out_net1 = nn.Sequential(*out_net1)

        # ! for perceptual loss
        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()
                    # nn.Sigmoid()
                    ]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        # return self.out_net1(out)
        return self.out_net2(out)
        # return self.out_net1(out), self.out_net2(out)