import os
import conf as cfg
from torchvision.datasets import ImageFolder
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


def extract_patches(srcpath, trgpath):
    W, H = 256, 256
    folders = os.listdir(srcpath)
    try:
        folders.remove('.DS_Store')
    except Exception as e:
        print("is removed")

    for folder in folders:
        folderpath = os.path.join(srcpath, folder)
        trgfolderpath = os.path.join(trgpath, folder)
        try:
            os.makedirs(trgfolderpath)
        except Exception as e:
            print(" is already exist")

        files = os.listdir(folderpath)
        i = 0

        for file in files:
            filepath = os.path.join(folderpath, file)
            img = cv2.imread(filepath)
            h, w, c = img.shape
            ch, cw = int(h/2), int(w/2)
            dh, dw = int(H/2), int(W/2)
            crop = img[ch-dh:ch+dh, cw-dw:cw+dw, :]
            cropg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            fname = f"crop_{i}.jpg"
            fpath= os.path.join(trgfolderpath, fname)
            cv2.imwrite(filename=fpath, img=cropg)


            i+=1
        # break



t = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[126.14/255], std=[34.95/255])])

dataset = ImageFolder(root=cfg.paths['images'], transform=t)

def createdb(dataset: Dataset, batch_size=64, train_percent=0.85):
    l = len(dataset)
    train_size = int(l*train_percent)
    test_size = l - train_size
    train, test = random_split(dataset=dataset, lengths=[train_size, test_size])
    train_loader = DataLoader(dataset=train, batch_size=batch_size)
    test_loader = DataLoader(dataset=test, batch_size=batch_size)

    return train_loader, test_loader

trainl, testl = createdb(dataset=dataset, batch_size=64)

def main():
    # extract_patches(srcpath=cfg.paths['src_data'], trgpath=cfg.paths['images'])
    batch = next(iter(trainl))
    print(batch[0].shape)
    print(batch[1])

if __name__ == '__main__':
    main()