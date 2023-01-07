import os
import numpy as np
import torch
import random



# seed intialization
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

root = os.getcwd()

paths = dict(
    root=root, data_root=os.path.join(root, 'data'), images=os.path.join(root, 'data', 'images'), 
    src_data=os.path.join(root, 'data', 'src_data'), model=os.path.join(root, 'data', 'model'),
    train=os.path.join(root, 'data', 'images', 'train'), test=os.path.join(root, 'data', 'images', 'test')
)

constlayer = dict(ks=5, scale=1, outch=3)

def creatdir(path: str):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f"{path} is already exist!!!!")

    




def main():
    pass


if __name__ == '__main__':
    main()


