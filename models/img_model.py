import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from config import opt
from .basic_module import BasicModule, RegGRU, RMSNorm, BottleStack, RGA_Module
import torch.nn.init as init

import timm


class IMG(BasicModule):
    def __init__(self, hidden_dim, output_dim, dropout, num_class, adj_file):
        super(IMG, self).__init__()
        self.module_name = 'IMG_module'
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop_out = dropout
        self.features = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True, num_classes=opt.hidden_dim)
        self.fully_c = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(True),
            nn.Linear(in_features=8192, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        self.hash_module = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
            )

        self.weight_init()

    def freezen(self):
        for name, para in self.features.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.hash_module:
            initializer(m)
        for m in self.fully_c:
            initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        fe = self.features(x.to(x.device, non_blocking=True)).squeeze()
        if opt.is_dual_card:
            fe = fe.to("cuda:0", non_blocking=True)

        f_m = self.fully_c(fe)
        hid = self.hash_module(f_m)
        x_code = torch.tanh(hid)
        return f_m, hid, x_code.squeeze()

    def generate_img_code(self, i):
        f_i = self.features(i).squeeze()
        if opt.is_dual_card:
            f_i = f_i.to("cuda:0", non_blocking=True)

        f_i = self.fully_c(f_i)
        hid = self.hash_module(f_i.detach())
        f_i = torch.tanh(hid)
        return f_i.squeeze()


if __name__ == '__main__':
    # model = IMG(1, 1, 1, 1, 1)
    model = IMG(hidden_dim=2048, output_dim=64, dropout=False, num_class=1, adj_file=1)
    print(model)
