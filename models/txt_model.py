import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from .basic_module import BasicModule, RegGRU, RMSNorm
import torch.nn.init as init
from config import opt
from .cbam import LKA_Attention_1d

class TXT(BasicModule):
    def __init__(self, nfeat, hidden_dim, output_dim, dropout, num_class, adj_file):
        super(TXT, self).__init__()
        self.drop_out = dropout
        self.nfeat = nfeat
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.module_name = 'TXT_module'

        self.conv_stage = nn.Sequential(
            nn.Conv1d(in_channels=nfeat, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1)
            )
        self.conv_stage2 =  nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1)
        )


        if self.drop_out:
            self.fully_c = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=8192, bias=True),
                nn.BatchNorm1d(8192),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(in_features=8192, out_features=hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True)
            )
        else:
            self.fully_c = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=8192, bias=True),
                nn.BatchNorm1d(8192),
                nn.ReLU(True),
                nn.Linear(in_features=8192, out_features=hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True)
            )

        self.hash_module = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True),
        )
        self.weight_init()

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
        x = x.view(x.size(0), self.nfeat, -1)
        f_e = self.conv_stage(x).squeeze()
        f_i = self.fully_c(f_e)
        f_e = f_i.view(f_i.size(0), self.hidden_dim, -1)
        f_e = self.conv_stage2(f_e).squeeze()

        hid = self.hash_module(f_e).squeeze()
        x_code = torch.tanh(hid)
        return f_i, hid, x_code.squeeze()

    def generate_txt_code(self, x):
        x = x.view(x.size(0), self.nfeat, -1)
        f_x = self.conv_stage(x).squeeze()
        f_x = self.fully_c(f_x)
        
        f_x = f_x.view(f_x.size(0), self.hidden_dim, -1)
        f_x = self.conv_stage2(f_x).squeeze()
        hid = self.hash_module(f_x.detach())
        h_x = torch.tanh(hid)
        return h_x.squeeze()


if __name__ == "__main__":
    pass
