import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary

import os, sys
sys.path.append(os.pardir)
import conf as cfg


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelBase(nn.Module):
    def __init__(self, name, created_time):
        super(ModelBase, self).__init__()
        self.name = name
        self.created_time = created_time

    def copy_params(self, state_dict):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

    def boost_params(self, scale=1.0):
        if scale == 1.0:
            return self.state_dict()
        for (name, param) in self.state_dict().items():
            self.state_dict()[name].copy_((scale * param).clone())
        return self.state_dict()

    # self - x
    def sub_params(self, x):
        own_state = self.state_dict()
        for (name, param) in x.items():
            if name in own_state:
                own_state[name].copy_(own_state[name] - param)

    # self + x
    def add_params(self, x):
        a = self.state_dict()
        for (name, param) in x.items():
            if name in a:
                a[name].copy_(a[name] + param)


class ConstConv(ModelBase):
    """
    doc
    """
    def __init__(self, lcnf: dict, name='constlayer', created_time=None):
        super().__init__(name=name, created_time=created_time)
        self.lcnf = lcnf
        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[lcnf['outch'], 1, lcnf['ks'], lcnf['ks']]), requires_grad=True)
        self.fx = self.feat_ext()
        self.coords = self.coordinates()



    def coordinates(self):
        x_coord = torch.zeros(size=(256, 256), device=dev)
        y_coord = torch.zeros(size=(256, 256), device=dev)
        for i in range(256):
            x_coord[i, :] = i
            y_coord[:, i] = i
        x = 2*(x_coord/255) - 1
        y = 2*(y_coord/255) - 1
        x.unsqueeze_(dim=0)
        y.unsqueeze_(dim=0)
        z = torch.cat((x, y), dim=0)
        # z.unsqueeze_(dim=0)
        return z

    def add_pos(self, batch):
        Z = []
        for i in range(batch.shape[0]):
            z = torch.cat((batch[i], self.coords))
            Z.append(z.unsqueeze_(dim=0))
        return torch.cat(tensors=Z, dim=0)

    def normalize(self):
        cntrpxl = int(self.lcnf['ks']/2)
        centeral_pixels = (self.const_weight[:, 0, cntrpxl, cntrpxl])
        for i in range(self.lcnf['outch']):
            sumed = (self.const_weight.data[i].sum() - centeral_pixels[i])/self.lcnf['scale']
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, cntrpxl, cntrpxl] = -self.lcnf['scale']

    def feat_ext(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=self.lcnf['outch']+2, out_channels=96, kernel_size=7, stride=2), nn.BatchNorm2d(num_features=96),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride=1), nn.BatchNorm2d(num_features=64),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1), nn.BatchNorm2d(num_features=64),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1), nn.BatchNorm2d(num_features=128),
            nn.Tanh(), nn.AvgPool2d(kernel_size=9, stride=2),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), nn.BatchNorm2d(num_features=128),
            # nn.Tanh()#, nn.AvgPool2d(kernel_size=7, stride=2)
            nn.Flatten(), nn.Linear(in_features=128, out_features=6)
        )

        return layer

    def forward(self, x):
        self.normalize()
        noise = F.conv2d(x, self.const_weight, padding='same')
        x = self.add_pos(noise)
        x = self.fx(x)
        return x 





def main():
    x = torch.randn(size=[15, 1, 256, 256])
    model = ConstConv(lcnf=cfg.constlayer)
    # summary(model, input_size=[1, 1, 256, 256])
    out = model(x)
    print(out.shape)
    # x = torch.randn(size=(2, 20, 20))
    # y = torch.randn(size=(5, 8, 20, 20))
    # # z = torch.stack(tensors=(x, y), dim=1)
    # Z = []
    # for i in range(y.shape[0]):
    #     # print(y[i].shape)
    #     z = torch.cat(tensors=(y[i], x))
    #     Z.append(z.unsqueeze_(dim=0))

    # zz = torch.cat(tensors=Z, dim=0)
    # print(zz.shape)
    


    # # print(z.shape)    


if __name__ == '__main__':
    main()