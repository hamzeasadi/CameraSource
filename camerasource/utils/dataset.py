
from typing import List, Dict
import os
import sys
sys.path.append("../..")


import numpy as np



# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace

def unit_vec(vector:np.ndarray):
    return vector / np.linalg.norm(vector)


def angle_between(vec1:np.ndarray, vec2:np.ndarray, degree:bool):
    v1_u = unit_vec(vector=vec1)
    v2_u = unit_vec(vector=vec2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if degree:
        return (180/np.pi)*angle
    
    return angle


def get_prependecular(ref_vec:np.ndarray, dim:int, cross_product:bool):
    
    if cross_product:
        assert dim==3, f"cross product just avilable for dim 3 but you provide {dim}"
        rnd_vec = np.random.randn(dim)
        perp_vec = np.cross(ref_vec, rnd_vec)
        return perp_vec
    
    rnd_idx = np.random.randint(low=0, high=dim)
    rnd_vec = np.random.randn(dim)
    rnd_vec[rnd_idx] = 0
    rnd_idx_value = np.dot(rnd_vec, ref_vec)
    rnd_vec[rnd_idx] = -rnd_idx_value / (ref_vec[rnd_idx] + 1e-10)
    
    return rnd_vec


def get_vector(ref_vec:np.ndarray, dim:int, theta:float, 
               mag:float, cross_product:bool):
    
    vec_perp = get_prependecular(ref_vec=ref_vec, dim=dim, cross_product=cross_product)
    theta_vec = unit_vec(ref_vec) * np.cos(theta) + unit_vec(vec_perp) * np.sin(theta)
    
    return mag*theta_vec
    







if __name__ == "__main__":
    print(__file__)