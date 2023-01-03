import os, random
import conf as cfg
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms



# seed intialization
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)






def bgr2graycoord(img):

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grayimg.shape

    channelx = np.ones(shape=(h, w))
    for i in range(h):
        channelx[i, :] = i*channelx[i, :]
    channelx = 2*(channelx/h) - 1

    channely = np.ones(shape=(h, w))
    for i in range(w):
        channely[:, i] = i*channely[:, i]
    channely = 2*(channely/w) - 1
    
    img[:, :, 0] = grayimg
    img[:, :, 1] = channelx
    img[:, :, 2] = channely

    return img
    

trf = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=[125/255], std=[36/255])])
def imgpatchs(img):
    H, W = 224, 224
    h, w, c = img.shape
    graytrf = trf(img[:, :, 0])
    img[:, :, 0] = graytrf.numpy()
    numh = 2*int(h/H) - 1
    numw = 2*int(w/W) - 1
    patches = []
    for i in range(numh):
        hi = i*int(H/2)
        for j in range(numw):
            wi = j*int(W/2)
            patch = img[hi:hi+H, wi:wi+W, :]
            patches.append(patch)
    return patches


def extractallpatches(src_path, trg_path):

    srcfolders = os.listdir(src_path)
    try:
        srcfolders.remove('.DS_Store')
    except Exception as e:
        print(f'{e}')

    for srcfolder in srcfolders:
        srcfolderpath = os.path.join(src_path, srcfolder)
        trgtrainfolderpath = os.path.join(cfg.paths['train'], srcfolder)
        trgtestfolderpath = os.path.join(cfg.paths['test'], srcfolder)
        srcfolderfiles = os.listdir(srcfolderpath)
        numsrcfiles = len(srcfolderfiles)
        train_size = int(0.8*numsrcfiles)

        cfg.creatdir(trgtrainfolderpath)
        cfg.creatdir(trgtestfolderpath)
        i=0

        for cnt, srcfile in enumerate(srcfolderfiles):
            srcimgpath = os.path.join(srcfolderpath, srcfile)
            srcimg = cv2.imread(srcimgpath)
            coordimg = bgr2graycoord(srcimg)
            coordpatches = imgpatchs(coordimg)

            if cnt<train_size:
                for patch in coordpatches:
                    patchname = f'patch_{i}.png'
                    patchpath = os.path.join(trgtrainfolderpath, patchname)
                    cv2.imwrite(filename=patchpath, img=patch)
                    i+=1
            else:
                for patch in coordpatches:
                    patchname = f'patch_{i}.png'
                    patchpath = os.path.join(trgtestfolderpath, patchname)
                    cv2.imwrite(filename=patchpath, img=patch)
                    i+=1


def main():
    srcpath = cfg.paths['src_data']
    trgpath = cfg.paths['images']
    # imgpath = os.path.join(cfg.paths['src_data'], 'Truck Rear', '2022-09-07_090505_Truck Transfer Lane Left_b189212_1_0.jpeg')
    # bgrimg = cv2.imread(imgpath)
    # coordimg = bgr2graycoord(bgrimg)
    # imgpatchs(coordimg)
    extractallpatches(src_path=srcpath, trg_path=trgpath)



if __name__ == '__main__':
    main()