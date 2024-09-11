import torch
from torch import einsum
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.init as init
from einops import rearrange
import torch.nn.functional as F
import os
from torch.nn.modules.loss import _WeightedLoss

class CustFlatten(nn.Module):
    def __init__(self):
        super(CustFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CustUnFlatten(nn.Module):
    def __init__(self):
        super(CustUnFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),  x.size(1), -1)

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)

# RGA注意力机制
class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module,self).__init__()
        self.in_channel=in_channel  # C=256
        self.in_spatial=in_spatial  # H*W=64*32=2048
        self.inter_channel=in_channel//cha_ratio    # C//8=256//8=32
        self.inter_spatial=in_spatial//spa_ratio    # (H*W)//8=2048//8=256
        self.use_spatial=use_spatial    # 是否使用RGA-S
        self.use_channel=use_channel    # 是否使用RGA-C
        if self.use_spatial:
            # 定义5个卷积
            # (8*256*64*32)--(8*32*64*32)
            self.theta_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*256*64*32)--(8*32*64*32)
            self.phi_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*4096*64*32)--(8*256*64*32)
            self.gg_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial*2,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*256*64*32)--(8*32*64*32)
            self.gx_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*257*64*32)--(8*1*64*32)
            num_channel_s=1+self.inter_spatial
            self.W_spatial=nn.Sequential(
                    nn.Conv2d(in_channels=num_channel_s,
                            out_channels=num_channel_s//down_ratio,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(num_channel_s//down_ratio),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=num_channel_s//down_ratio,
                            out_channels=1,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(1)
            )
        if self.use_channel:
            # 定义5个卷积
            # (8*2048*256*1)--(8*256*256*1)
            self.theta_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*2048*256*1)--(8*256*256*1)
            self.phi_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*512*256*1)--(8*32*256*1)
            self.gg_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel*2,
                            out_channels=self.inter_channel,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_channel),
                    nn.ReLU()
            )
            # (8*2048*256*1)--(8*256*256*1)
            self.gx_channel=nn.Sequential(
                    nn.Conv2d(in_channels=self.in_spatial,
                            out_channels=self.inter_spatial,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.inter_spatial),
                    nn.ReLU()
            )
            # (8*33*256*1)--(8*1*256*1)
            num_channel_c=1+self.inter_channel
            self.W_channel=nn.Sequential(
                    nn.Conv2d(in_channels=num_channel_c,
                            out_channels=num_channel_c//down_ratio,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(num_channel_c//down_ratio),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=num_channel_c//down_ratio,
                            out_channels=1,
                            kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(1)
            )
    def forward(self,x):
        # input x:(8,256,64,32)
        b,c,h,w=x.size()    # 8,256,64,32
        if self.use_spatial:
            theta_xs=self.theta_spatial(x)  # 8*32*64*32
            phi_xs=self.phi_spatial(x)      # 8*32*64*32
            theta_xs=theta_xs.view(b,self.inter_channel,-1) # 8*32*2048
            theta_xs=theta_xs.permute(0,2,1)                # 8*2048*32
            phi_xs=phi_xs.view(b,self.inter_channel,-1)     # 8*32*2048
            Gs=torch.matmul(theta_xs,phi_xs)                # 8*2048*2048
            Gs_in=Gs.permute(0,2,1).view(b,h*w,h,w)         # 8*2048*64*32
            Gs_out=Gs.view(b,h*w,h,w)                       # 8*2048*64*32
            Gs_joint=torch.cat((Gs_in,Gs_out),1)            # 8*4096*64*32
            Gs_joint=self.gg_spatial(Gs_joint)              # 8*256*64*32
            g_xs=self.gx_spatial(x)                         # 8*32*64*32
            g_xs=torch.mean(g_xs,dim=1,keepdim=True)        # 8*1*64*32
            ys=torch.cat((g_xs,Gs_joint),1)                 # 8*257*64*32
            w_ys=self.W_spatial(ys)                         # 8*1*64*32
            if not self.use_channel:
                out=torch.sigmoid(w_ys.expand_as(x))*x      # 8*256*64*32
                return out
            else:
                x=torch.sigmoid(w_ys.expand_as(x))*x        # 8*256*64*32
        if self.use_channel:
            xc=x.view(b,c,-1).permute(0,2,1).unsqueeze(-1)              # 8*2048*256*1
            theta_xc=self.theta_channel(xc).squeeze(-1).permute(0,2,1)  # 8*256*256
            phi_xc=self.phi_channel(xc).squeeze(-1) # 8*256*256
            Gc=torch.matmul(theta_xc,phi_xc)        # 8*256*256
            Gc_in=Gc.permute(0,2,1).unsqueeze(-1)   # 8*256*256*1
            Gc_out=Gc.unsqueeze(-1)                 # 8*256*256*1
            Gc_joint=torch.cat((Gc_in,Gc_out),1)    # 8*512*256*1
            Gc_joint=self.gg_channel(Gc_joint)      # 8*32*256*1
            g_xc=self.gx_channel(xc)                # 8*256*256*1
            g_xc=torch.mean(g_xc,dim=1,keepdim=True)    # 8*1*256*1
            yc=torch.cat((g_xc,Gc_joint),1)             # 8*33*256*1
            w_yc=self.W_channel(yc).transpose(1,2)      # 8*1*256*1--8*256*1*1
            out=torch.sigmoid(w_yc)*x                   # 8*256*64*32
            return out

## 继承_WeightedLoss类
class SmoothingBCELossWithLogits(_WeightedLoss):
	def __init__(self, weight=None, reduction='mean', smoothing=0.1):
		super(SmoothingBCELossWithLogits, self).__init__(weight=weight, reduction=reduction)
		self.smoothing = smoothing
		self.weight  = weight
		self.reduction = reduction

	@staticmethod
	def _smooth(targets, n_labels, smoothing=0.0):
		assert 0 <= smoothing < 1
		with torch.no_grad():
			targets = targets  * (1 - smoothing) + 0.5 * smoothing
		return targets

	def forward(self, inputs, targets):
		targets = self._smooth(targets, inputs.size(-1), self.smoothing)
		loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weights)

		if self.reduction == 'sum':
			loss = loss.item()
		elif self.reduction == 'mean':
			loss = loss.mean()
		return loss
		

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class Wasserstein(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps=0.01, max_iter=100, reduction='sum'):
        super(Wasserstein, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).to(x.device, non_blocking=True)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(x.device, non_blocking=True)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(x.device, non_blocking=True)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# positional embeddings

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out

class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head

        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias = False),
            nn.BatchNorm2d(attn_dim_in),
            activation,
            Attention(
                dim = attn_dim_in,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x = x + shortcut
        return self.activation(x)

# main bottle stack

class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        fmap_size = pair(fmap_size)

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample

            fmap_divisor = (2 if downsample and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        return self.net(x)

def deepnorm(x, x1, a):
   return LayerNorm(x*a + x1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TXT_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num_layers=8):
        super(TXT_GRU, self).__init__()
        # 定义GRU
        self.rnn = nn.GRU(hidden_dim, hidden_dim, hidden_num_layers)
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            RMSNorm(hidden_dim),
            nn.PReLU(hidden_dim)
        )
        # 定义回归层网络，输入的特征维度等于GRU的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            RMSNorm(hidden_dim),
            nn.PReLU(hidden_dim)
        )
        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.reg:
            initializer(m) 
        for m in self.fc:
            initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        m = self.fc(x)
        r, _ = self.rnn(m)
        p = r + m
        x = self.reg(p) + m
        return x.squeeze()

class RegGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num_layers=8):
        super(RegGRU, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim // 2),
            RMSNorm(hidden_dim // 2),
            nn.PReLU(hidden_dim // 2)
        )
        # 定义GRU
        self.rnn = nn.GRU(input_dim // 2, hidden_dim // 2, hidden_num_layers)
        # 定义回归层网络，输入的特征维度等于GRU的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim),
            RMSNorm(hidden_dim),
            nn.PReLU(hidden_dim)
            # nn.ReLU(True)
        )
        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.reg:
            initializer(m)
        for m in self.fc:
            initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        m = self.fc(x)
        # m = self.reg(x)
        r, _ = self.rnn(m)
        att1 = torch.sigmoid(r)
        att2 = torch.sigmoid(m)
        p = m * att2 + m * att1
        x = self.reg(p) + x
        return x.squeeze()


class BasicModule(nn.Module):
    """
    封装nn.Module,主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            # print(t.load(path, map_location=lambda storage, loc: storage).keys())
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            # print(t.load(path, map_location=lambda storage, loc: storage).keys())
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        """
        保存模型,默认使用"模型名字+时间"作为文件名
        """
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, name))
        # print('save success!')
        return name

    def forward(self, *input):
        pass

