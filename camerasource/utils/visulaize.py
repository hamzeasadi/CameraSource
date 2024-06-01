"""
docs
"""

from typing import List, Dict
import os
import sys
sys.path.append("../..")

import numpy as np
from matplotlib import pyplot as plt


# pylint: disable=trailing-whitespace
# pylint: disable=missing-function-docstring

COLORS = ['b', 'r', 'g', 'cyan', 'yellow', 'brown']

def plot_vec_2d(vec_stack:np.ndarray, gt_mags:np.ndarray):
    
    num_vecs = vec_stack.shape[0]
    max_mag = 2 * np.max(vec_stack)
    
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim([-max_mag, max_mag])
    ax.set_ylim([-max_mag, max_mag])
    for vec_idx in range(num_vecs):
        vec = vec_stack[vec_idx]
        color = COLORS[gt_mags[vec_idx]]
        ax.quiver(0, 0, vec[0], vec[1], color=color, units='xy', scale=1)
    plt.close()
    return fig


def plot_vec_3d(vec_stack:np.ndarray, gt_mags:np.ndarray):
    
    num_vecs = vec_stack.shape[0]
    max_mag = 2 * np.max(vec_stack)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([-max_mag, max_mag])
    ax.set_ylim([-max_mag, max_mag])
    ax.set_zlim([-max_mag, max_mag])
    for vec_idx in range(num_vecs):
        vec = vec_stack[vec_idx]
        color = COLORS[gt_mags[vec_idx]]
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color)
    plt.close()
    return fig
    


if __name__ == "__main__":
    print(__file__)
    