"""
Synthetic data for evaluation my model
"""

import os
import sys
sys.path.append("../..")
from typing import List, Dict
import argparse

import numpy as np
# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from camerasource.utils.dataset import get_vector
from camerasource.utils.visulaize import plot_vec_2d, plot_vec_3d

# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace









if __name__ == "__main__":
    print(__file__)
    
    ref_vec = np.random.randn(3)
    vec = get_vector(ref_vec=ref_vec, dim=3, theta=np.pi/10, mag=2, cross_product=True)
    vec_stack = np.vstack((ref_vec, vec))
    fig = plot_vec_3d(vec_stack=vec_stack, gt_mags=np.array([0, 1]))
    print(2)
    fig.savefig("img.png")
    