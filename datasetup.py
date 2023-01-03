import os
import conf as cfg
from torchvision.datasets import ImageFolder
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


t = transforms.Compose([transforms.ToTensor()])

traindataset = ImageFolder(root=cfg.paths['train'], transform=t)
testdataset = ImageFolder(root=cfg.paths['test'], transform=t)

def createdb(dataset: Dataset, batch_size=256, train_percent=0.80):
    l = len(dataset)
    train_size = int(l*train_percent)
    test_size = l - train_size
    train, test = random_split(dataset=dataset, lengths=[train_size, test_size])
    train_loader = DataLoader(dataset=train, batch_size=batch_size)
    test_loader = DataLoader(dataset=test, batch_size=batch_size)

    return train_loader, test_loader

trainl = DataLoader(dataset=traindataset, batch_size=256, shuffle=True)
vall, testl = createdb(dataset=testdataset, batch_size=256)

def main():
    # extract_patches(srcpath=cfg.paths['src_data'], trgpath=cfg.paths['images'])
    for X, Y in testl:
        print(X.shape)

if __name__ == '__main__':
    main()