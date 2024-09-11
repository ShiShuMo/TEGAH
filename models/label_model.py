import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from .basic_module import BasicModule, CustFlatten, CustUnFlatten
from .cbam import LKA_Attention_1d
import torch.nn.init as init

class mult_scale_att(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0., dualition=2, dim=128):
        super(mult_scale_att, self).__init__()
        self.embd_dim = hidden_dim
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(in_channels=self.dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mod = dualition
        self.scale = hidden_dim ** -0.5
        self.q = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=self.dim, kernel_size=3, stride=1, padding=1, bias=qkv_bias)
        self.k = nn.Conv1d(in_channels=hidden_dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1, bias=qkv_bias)
        self.v = nn.Conv1d(in_channels=hidden_dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1, bias=qkv_bias)

    def forward(self, x1, x2, x3, x4):
        B, c, N = x1.shape
        C = self.dim
        q = self.q(torch.cat([x3, x4], dim=1)).reshape(B, N, self.num_heads * self.mod, C // (self.num_heads * self.mod)).permute(0, 2, 3, 1)
        k = self.k(x1).reshape(B, N, self.num_heads * self.mod, C // (self.num_heads * self.mod)).permute(0, 2, 3, 1)
        v = self.v(x2).reshape(B, N, self.num_heads * self.mod, C // (self.num_heads * self.mod)).permute(0, 2, 3, 1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.proj_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LABEL(BasicModule):
    def __init__(self, label_dim, hidden_dim, output_dim):
        super(LABEL, self).__init__()
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.att = mult_scale_att(hidden_dim=hidden_dim)
        self.module_name = 'LABEL_module'

        self.conv_stage = nn.Sequential(
            nn.Conv1d(in_channels=label_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            # LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1)
            )
        
        self.conv_stage2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            # LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1)
            )

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=10, padding=1),
            nn.ReLU(True),
            LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=5, padding=1),
            nn.ReLU(True),
            LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=3, padding=1),
            nn.ReLU(True),
            LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=1),
            nn.ReLU(True),
            LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1)
        )

        self.feature = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=8192, bias=True),
            nn.LayerNorm(8192),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8192, out_features=hidden_dim, bias=True),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        ) if label_dim == 80 else nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=8192, kernel_size=3, stride=1, padding=1),
            CustFlatten(),
            nn.LayerNorm(8192),
            CustUnFlatten(),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=8192, out_channels=self.hidden_dim, kernel_size=3, stride=1, padding=1),
            CustFlatten(),
            nn.LayerNorm(self.hidden_dim),
            CustUnFlatten(),
            nn.ReLU(inplace=True),
        )

        self.hash_module = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True),
        )

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.hash_module:
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
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.conv_stage(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)

        x = self.att(x1, x2, x3, x4).squeeze()
        f_e = self.feature(x)
        hid = self.hash_module(f_e)
        x_code = torch.tanh(hid)
        return f_e, hid, x_code.squeeze()


if __name__ == "__main__":
    pass
