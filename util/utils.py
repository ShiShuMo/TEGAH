import torch
from PIL import Image
import torch.nn as nn
import numpy as np
import visdom
import time
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import functional as F
from config import opt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import textwrap
import torch.nn.init as init

from logging import getLogger

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

logger = getLogger()

@torch.jit.script
def quantizationLoss(hashrepresentations_bs, hashcodes_bs):
    batch_size, bit = hashcodes_bs.shape
    quantization_loss = torch.sum(torch.pow(hashcodes_bs - hashrepresentations_bs, 2)) / (batch_size * bit)
    return quantization_loss

@torch.jit.script
def multilabelsimilarityloss_KL(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    a1 = labelsSimilarity - hashrepresentationsSimilarity
    a2 = torch.log((1e-5 + labelsSimilarity) / (1e-5 + hashrepresentationsSimilarity))
    # print(a1.shape)
    # print(a2.shape)
    # torch.mul(a1, a2)
    # KLloss = (torch.sum(a1) * torch.sum(a2)) / ( num_train * batch_size)
    # KLloss = (torch.sum(a1) + torch.sum(a2)) / ( num_train * batch_size)
    KLloss2 = torch.sum(torch.relu(labelsSimilarity - hashrepresentationsSimilarity)) / (num_train * batch_size)
    KLloss3 = torch.sum(torch.relu(hashrepresentationsSimilarity - labelsSimilarity)) / (num_train * batch_size)
    MSEloss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (num_train * batch_size)
    # KLloss =  KLloss + 0.5 * KLloss2 + 0.5 * KLloss3 + MSEloss
    KLloss = 0.5 * KLloss2 + 0.5 * KLloss3 + MSEloss
    # print('KLloss1 = %4.4f, KLloss2 = %4.4f'%(KLloss1 , KLloss2))
    return KLloss

@torch.jit.script
def multilabelsimilarityloss_MSE(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    MSEloss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (num_train * batch_size)

    return MSEloss

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # sim: {0, 1}^{mxn}
    num_query = query_label.shape[0]
    map = 0.
    GND = (query_label.mm(retrieval_label.t()) > 0).type(torch.float).squeeze().cuda(non_blocking=True)
    if k is None:
        k = retrieval_label.shape[0]
    for iter in tqdm(range(num_query)):
        # gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze().cuda(non_blocking=True)
        gnd = GND[iter, :].cuda(non_blocking=True)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        # count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        count = torch.arange(1, total + 1).type(torch.float).cuda(non_blocking=True)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex).cuda(non_blocking=True)
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_topK(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''
        self.env = env
    
    def save(self):
        self.vis.save([self.env])
 
    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / tindex)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
