# ------------------------------------------------------------------------------
# https://github.dev/HRNet/HigherHRNet-Human-Pose-Estimation
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import logging

import torch
import torch.nn as nn
from pdb import set_trace as st

logger = logging.getLogger(__name__)


class HeatmapGenerator():
    def __init__(self, heatmap_size, num_joints=68, sigma=2):
        self.heatmap_size = heatmap_size
        # self.image_size = image_size
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.heatmap_size / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # def __call__(self, joints, image_size: np.ndarray):
    def __call__(self, joints, image_size: int):
        """generate heatmap gt from joints

        Args:
            joints (np.ndarray): N,3

        Returns:
            hms: N,H,W
        """
        hms = np.zeros((self.num_joints, self.heatmap_size, self.heatmap_size),
                       dtype=np.float32)
        sigma = self.sigma

        # feat_stride = image_size / [self.heatmap_size, self.heatmap_size]
        feat_stride = image_size / self.heatmap_size
        for idx, pt in enumerate(joints):
            # for idx, pt in enumerate(p):
            if pt[2] > 0:
                # x = int(pt[0] / feat_stride[0] + 0.5)
                # y = int(pt[1] / feat_stride[1] + 0.5) # normalize joints to heatmap size
                x = int(pt[0] / feat_stride + 0.5)
                y = int(pt[1] / feat_stride +
                        0.5)  # normalize joints to heatmap size
                if x < 0 or y < 0 or \
                    x >= self.heatmap_size or y >= self.heatmap_size:
                    continue

                ul = int(np.round(x - 3 * sigma - 1)), int(
                    np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(
                    np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.heatmap_size) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.heatmap_size) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.heatmap_size)
                aa, bb = max(0, ul[1]), min(br[1], self.heatmap_size)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd],
                                                    self.g[a:b, c:d])
        return hms


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask=None):
        # todo, add mask
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        if mask is not None:
            loss = loss * mask[:, None, :, :].expand_as(pred)
        # loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        loss = loss.mean()
        # loss = loss.mean(dim=3).mean(dim=2).sum(dim=1)
        return loss
