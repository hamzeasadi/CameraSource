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
    src_data=os.path.join(root, 'data', 'src_data'), model=os.path.join(root, 'data', 'model')
)

constlayer = dict(ks=5, scale=1, outch=8)

class Conf():
    """
    doc
    """
    




def main():
    pass


if __name__ == '__main__':
    main()


