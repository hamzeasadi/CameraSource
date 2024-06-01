"""
Genral utilities functions
"""
import pickle
from typing import Dict, Any
import json
from dataclasses import dataclass
import os
import sys
sys.path.append("../..")


import numpy as np


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace


def crtdir(path:str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(file_path:str):
    """
    doc
    """
    with open(file_path, mode="r",encoding="utf-8") as jfile:
        data = json.load(jfile)
    return data


def save_json(data:Dict, save_path:str, file_name:str):
    """
    doc
    """
    crtdir(save_path)
    file_path:str = os.path.join(save_path, file_name)
    with open(file_path, mode='w', encoding='utf-8') as jfile:
        json.dump(data, jfile, ensure_ascii=False, indent=4)


def load_pickle(file_path:str):
    """
    doc
    """
    with open(file_path, mode="rb") as pklfile:
        data = pickle.load(pklfile)
    return data


def save_pickle(data:Any, save_path:str, file_name:str):
    """
    doc
    """
    crtdir(save_path)
    file_path:str = os.path.join(save_path, file_name)
    with open(file_path, mode="wb") as pklfile:
        pickle.dump(obj=data, file=pklfile)


def load_npy(file_path:str):
    """
    doc
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        data = np.load(file_path, allow_pickle=True)
        
    return data


def save_npy(data:Any, save_path:str, file_name:str):
    """
    doc
    """
    crtdir(save_path)
    file_path:str = os.path.join(save_path, file_name)
    np.save(file_path, arr=data, allow_pickle=True)


def save_data(data:Any, save_path:str, file_name:str):
    """
    doc
    """
    ext = os.path.splitext(file_name)[-1].strip()
    if ext == "npy":
        save_npy(data=data, save_path=save_path, file_name=file_name)
    elif ext == "pkl":
        save_pickle(data=data, save_path=save_path, file_name=file_name)
    elif ext == "json":
        save_json(data=data, save_path=save_path, file_name=file_name)
    else:
        raise NotImplementedError(f"This method can not save file with {ext}. \
                                  supporting extensions are:npy, json, pkl")


def load_data(file_path):
    """
    doc
    """
    base_name = os.path.basename(file_path)
    ext = os.path.splitext(base_name)[-1].strip()
    if ext == "npy":
        data = load_npy(file_path=file_path)
    elif ext == "pkl":
        data = load_pickle(file_path=file_path)
    elif ext == "json":
        data = load_json(file_path=file_path)
    else:
        raise NotImplementedError(f"This method can not load file with {ext}. \
            supporting extensions are:npy, json, pkl")

    return data



@dataclass
class Paths:
    project_root:str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path:str = os.path.join(project_root, "data")
    
    dataset_path:str = os.path.join(data_path, "dataset")
    model_path:str = os.path.join(data_path, "model")
    report_path:str = os.path.join(data_path, "report")
    config_path:str = os.path.join(data_path, "config")
    
    def crtdir(self, path:str):
        crtdir(path)

    def init_paths(self):
        for path_name, path_dir in self.__dict__.items():
            if os.path.expanduser("~") in path_dir:
                self.crtdir(path=path_dir)







if __name__ == "__main__":
    print(__file__)
    paths = Paths()
    paths.init_paths()
    