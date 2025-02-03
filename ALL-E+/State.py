import numpy as np
import sys
import cv2
from utils import batch_psnr, normalize, init_logger_ipol, variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import torch

import os
from models import FFDNet
from torch.autograd import Variable
import matplotlib.image as mpimg
from PIL import Image


class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range
        # self.net = model

    def reset(self, x):
        self.image = x
        self.raw = x * 255
        self.raw[np.where(self.raw <= 2)] = 3

    def step(self, act):
        neutral = 9
        move = act.astype(np.float32)
        moves = (move - neutral) / 18
        moved_image = np.zeros(self.image.shape, dtype=np.float32)
        # de = move[:, 3:, :, :]
        r = self.image[:, 0, :, :]
        g = self.image[:, 1, :, :]
        b = self.image[:, 2, :, :]
        moved_image[:, 0, :, :] = r + (moves[:, 0, :, :]) * r * (1 - r)
        moved_image[:, 1, :, :] = g + (0.1 * moves[:, 1, :, :] + 0.9 * moves[:, 0, :, :]) * g * (1 - g)
        moved_image[:, 2, :, :] = b + (0.1 * moves[:, 2, :, :] + 0.9 * moves[:, 0, :, :]) * b * (1 - b)
        self.image = 0.8 * moved_image + 0.2 * self.image
