import numpy as np
from utils import KeepTrack
import conf as cfg
from datasetup import trainl, testl
from camsrc0 import model
import engine
import argparse
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
# parser.add_argument('--val', action=argparse.BooleanOptionalAction)
parser.add_argument('--epoch', '-e', type=int, required=False, metavar='epoch', default=1)

args = parser.parse_args()


def train(net, train_loader, val_loader, opt, criterion, epochs, minerror, modelname:str):

    kt = KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        train_loss = engine.train_step(model=net, data=train_loader, criterion=criterion, optimizer=opt)
        val_loss = engine.val_step(model=net, data=val_loader, criterion=criterion)
        if val_loss < minerror:
            minerror = val_loss
            kt.save_ckp(model=net, opt=opt, epoch=epoch, minerror=val_loss, fname=modelname)
            
        print(f"train_loss={train_loss} val_loss={minerror}")
    


def main():
    model_name = f"residual_0.pt"
    keeptrack = KeepTrack(path=cfg.paths['model'])
    Net = model.ConstConv(lcnf=cfg.constlayer)
    Net.to(dev)
    opt = optim.Adam(params=Net.parameters(), lr=3e-4)
    # criteria = OrthoLoss()
    criteria = nn.CrossEntropyLoss()
    # dataset = Datad(path=cfg.paths['trg_parch'])
    # train_data, test_data = createds(dataset=dataset, batch_size=64)
    minerror = np.inf
    if args.train:
        train(net=Net, train_loader=trainl, val_loader=testl, opt=opt, criterion=criteria, epochs=100, minerror=minerror, modelname=model_name)

    if args.test:
        # model_name = f"source_0.pt"
        state = keeptrack.load_ckp(fname=model_name)
        Net.load_state_dict(state['model'], strict=False)
        print(f"min error is {state['minerror']} which happen at epoch {state['epoch']}")
        engine.test_step(model=Net, data=testl, criterion=criteria)



if __name__ == '__main__':
    main()