import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class LSKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=(1, 5), padding=2, groups=dim)
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=(5, 1), padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=(1, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv_spatial2 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=(7, 1), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv1(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return u * attn


class LSKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.activation = nn.ReLU(True)
        self.spatial_gating_unit = LSKA(dim=d_model)
        self.proj_2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class LKA_1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention_1d(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv1d(d_model, d_model, 1)
        self.activation = nn.ReLU(True)
        self.spatial_gating_unit = LKA_1d(d_model)
        self.proj_2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.ReLU(True)
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
