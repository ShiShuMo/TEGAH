import os
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import numpy as np
from torch.nn import functional as Func
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
import scipy.io as scio
from models.img_model import IMG
from models.txt_model import TXT
from models.label_model import LABEL
from models.basic_module import Wasserstein
import timm
from triplet_loss import *
from util.utils import calc_map_k, pr_curve, p_topK, Visualizer, ContrastiveLoss, Encoder, Decoder, pairwise_loss, CrossModalLoss, multilabelsimilarityloss_KL, quantizationLoss
from datasets.data_handler import load_data
import time
import pickle
from torch.optim import lr_scheduler, Adam, SGD, AdamW, RAdam, NAdam
from models.gat import GAT_MODEL
import argparse 

is_distributedDataParallel = True
is_dp = False
parser = argparse.ArgumentParser()
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--ddp", type=bool, default=True)
parser.add_argument("--dp", type=bool, default=False)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--flag", type=str, default='mir')
parser.add_argument("--bit", type=int, default=64)
parser.add_argument("--ngpus", type=int, default=2)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--dual_card", type=bool, default=False)
parser.add_argument("--gpu_id", type=str, default="2,3")
# parser.add_argument("--batch", type=int, default=80 * parser.parse_args().ngpus)
parser.add_argument("--batch", type=int, default=80)
args = parser.parse_args()

#参数
opt.flag = args.flag
opt.bit = args.bit
opt.batch_size = args.batch
opt.accumulation_steps = args.accumulation_steps
if args.flag == 'coco':
    opt.accumulation_steps = 1
opt.is_dp = args.dp
opt.is_distributedDataParallel = args.ddp
amp = False
is_pt = True

if args.dual_card:
    opt.is_dual_card = True

if args.ddp and args.ngpus != 1 and args.ngpus in (2, 3, 4):
    is_distributedDataParallel = True
else:
    current_rank = 0
    world_size = 1
    opt.rank = 0

if opt.is_dp:
    is_dp = True
    is_distributedDataParallel = False

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if opt.is_dp or opt.is_distributedDataParallel:
    # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"
    args.ngpus = len(args.gpu_id.split(','))
    # gpu_id_x = ["cuda:0", "cuda:1"]

if opt.is_distributedDataParallel:
    is_dp = False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(args.ngpus)
    dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    current_rank = dist.get_rank()
    opt.rank = current_rank

seed=42
def seed_torch(seed=seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    if opt.is_distributedDataParallel:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

seed_torch(seed)

if opt.is_distributedDataParallel:
    batch = opt.batch_size

resume = False
if args.resume:
    resume = True

def save_mat(query_img, query_txt, retrieval_img, retrieval_txt, query_labels, retrieval_labels, save__dir, output_dim=64, map=0.0, flag='mir', mode_name="i2t"):
    save_dir = os.path.join(save__dir, "PR_cruve")
    os.makedirs(save_dir, exist_ok=True)

    query_img = query_img.cpu().detach().numpy()
    query_txt = query_txt.cpu().detach().numpy()
    retrieval_img = retrieval_img.cpu().detach().numpy()
    retrieval_txt = retrieval_txt.cpu().detach().numpy()
    query_labels = query_labels.cpu().detach().numpy()
    retrieval_labels = retrieval_labels.cpu().detach().numpy()

    result_dict = {
        'q_img': query_img,
        'q_txt': query_txt,
        'r_img': retrieval_img,
        'r_txt': retrieval_txt,
        'q_l': query_labels,
        'r_l': retrieval_labels,
        'map': map
    }
    scio.savemat(os.path.join(save_dir, str(output_dim) + "-ours-" + flag + "-" + mode_name + ".mat"), result_dict)
    print(f">>>>>> save best {mode_name} data!")


def adjust_learning_rate(org_lr, lr_update, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = org_lr * (0.1 ** (epoch // lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def my_loss(y_true, y_pred):
    loss2 = opt.alpha * (y_pred.abs() - 1).pow(3).abs().mean()
    bceloss = nn.BCEWithLogitsLoss()
    loss3 = bceloss(y_true, torch.sigmoid(y_pred))
    return loss2 + loss3

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def train(**kwargs):
    opt.parse(kwargs)

    opt.vis_env = '{}_{}_full_{}'.format(opt.flag, opt.bit, 'mymodel')
    if opt.vis_env:
        if current_rank == 0:
            vis = Visualizer(opt.vis_env, port=opt.vis_port, server="http://10.109.118.58")

    images, tags, labels, _, inp_file = load_data(opt.data_path, opt.adj, opt.inp, type=opt.dataset)
    if opt.is_dual_card:
        inp_file = torch.from_numpy(inp_file).to("cuda:0", non_blocking=True)
    else:
        inp_file = torch.from_numpy(inp_file).cuda(non_blocking=True)

    train_data = Dataset(opt, images, tags, labels, is_pt=is_pt)
    if opt.is_distributedDataParallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=local_rank)
        train_dataloader = DataLoader(train_data, batch_size=batch, sampler=train_sampler, drop_last=False, pin_memory=True, shuffle=False, num_workers=16, prefetch_factor=2)
    else:
        train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, drop_last=False,
                                    pin_memory=True, shuffle=True, num_workers=16, prefetch_factor=2)


    L = train_data.get_labels()
    temp_adj = (L.t() @ L).numpy()
    adj = cosine_similarity(temp_adj, temp_adj)
    num = (torch.sum(L, dim=0)).numpy()
    adj_file = {'adj': adj, 'num': num}
    if opt.is_dual_card:
        L = L.to("cuda:0", non_blocking=True)
    else:
        L = L.cuda(non_blocking=True)

    # test
    i_query_data = Dataset(opt, images, tags, labels, test='image.query', is_pt=is_pt)
    i_db_data = Dataset(opt, images, tags, labels, test='image.db', is_pt=is_pt)
    t_query_data = Dataset(opt, images, tags, labels, test='text.query', is_pt=is_pt)
    t_db_data = Dataset(opt, images, tags, labels, test='text.db', is_pt=is_pt)

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, drop_last=False, pin_memory=True,
                                    shuffle=False, num_workers=16, prefetch_factor=2)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, drop_last=False, pin_memory=True, shuffle=False, num_workers=16, prefetch_factor=2)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, drop_last=False, pin_memory=True,
                                    shuffle=False, num_workers=16, prefetch_factor=2)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, drop_last=False, pin_memory=True, shuffle=False, num_workers=16, prefetch_factor=2)

    query_labels, db_labels = i_query_data.get_labels()
    if opt.is_dual_card:
        image_model = IMG(opt.hidden_dim, opt.bit, opt.dropout, opt.num_label, adj_file).to("cuda:0", non_blocking=True)
        image_model.features = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True, num_classes=opt.hidden_dim).to("cuda:1", non_blocking=True)
        text_model = TXT(opt.text_dim, opt.hidden_dim, opt.bit, opt.dropout, opt.num_label, adj_file).to("cuda:0", non_blocking=True)

        label_model = LABEL(opt.num_label, opt.hidden_dim, opt.bit).to("cuda:0", non_blocking=True)

        gat_model = GAT_MODEL(inp=inp_file, flag=opt.flag, num_class=opt.num_label, adj_file=adj_file, hidden_dim=opt.hidden_dim * 2, hash_dim=opt.bit).to("cuda:0", non_blocking=True)
    else:
        image_model = IMG(opt.hidden_dim, opt.bit, opt.dropout, opt.num_label, adj_file).cuda()

        text_model = TXT(opt.text_dim, opt.hidden_dim, opt.bit, opt.dropout, opt.num_label, adj_file).cuda()

        label_model = LABEL(opt.num_label, opt.hidden_dim, opt.bit).cuda()

        gat_model = GAT_MODEL(inp=inp_file, flag=opt.flag, num_class=opt.num_label, adj_file=adj_file, hidden_dim=opt.hidden_dim * 2, hash_dim=opt.bit).cuda()

    if opt.is_dp:
        image_model = DP(image_model).cuda()
        text_model = DP(text_model).cuda()
        gat_model = DP(gat_model).cuda()
        label_model = DP(label_model).cuda()

    if opt.is_distributedDataParallel:
        image_model = DDP(image_model, device_ids=[local_rank], output_device=local_rank)
        text_model = DDP(text_model, device_ids=[local_rank], output_device=local_rank)
        gat_model = DDP(gat_model, device_ids=[local_rank], output_device=local_rank)  # find_unused_parameters=True
        label_model = DDP(label_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if opt.is_distributedDataParallel:
        if opt.optim == 'Adam':
            optimizer_img_adam = Adam(image_model.parameters(), lr=opt.img_lr * opt.accumulation_steps, weight_decay=opt.img_weight_decay * opt.accumulation_steps)
            optimizer_txt_adam = Adam(text_model.parameters(), lr=opt.txt_lr * opt.accumulation_steps, weight_decay=opt.txt_weight_decay * opt.accumulation_steps)
            optimizer_gat_adam = Adam(gat_model.parameters(), lr=opt.gat_lr * opt.accumulation_steps, weight_decay=opt.gat_weight_decay * opt.accumulation_steps)
            optimizer_label_adam = Adam(label_model.parameters(), lr=opt.label_lr * opt.accumulation_steps, weight_decay=opt.label_weight_decay * opt.accumulation_steps)
        elif opt.optim == 'AdamW':
            optimizer_img_adam = AdamW(image_model.parameters(), lr=opt.img_lr * opt.accumulation_steps, weight_decay=opt.img_weight_decay * opt.accumulation_steps)
            optimizer_txt_adam = Adam(text_model.parameters(), lr=opt.txt_lr * opt.accumulation_steps, weight_decay=opt.txt_weight_decay * opt.accumulation_steps)
            optimizer_gat_adam = AdamW(gat_model.parameters(), lr=opt.gat_lr * opt.accumulation_steps, weight_decay=opt.gat_weight_decay * opt.accumulation_steps)
            optimizer_label_adam = AdamW(label_model.parameters(), lr=opt.label_lr * opt.accumulation_steps, weight_decay=opt.label_weight_decay * opt.accumulation_steps)
        else:
            optimizer_img_adam = NAdam(image_model.parameters(), lr=opt.img_lr * opt.accumulation_steps, weight_decay=opt.img_weight_decay * opt.accumulation_steps)
            optimizer_txt_adam = NAdam(text_model.parameters(), lr=opt.txt_lr * opt.accumulation_steps, weight_decay=opt.txt_weight_decay * opt.accumulation_steps)
            optimizer_gat_adam = NAdam(gat_model.parameters(), lr=opt.gat_lr * opt.accumulation_steps, weight_decay=opt.gat_weight_decay * opt.accumulation_steps)
            optimizer_label_adam = Adam(label_model.parameters(), lr=opt.label_lr * opt.accumulation_steps, weight_decay=opt.label_weight_decay * opt.accumulation_steps)

    else:
        if opt.optim == 'Adam':
            optimizer_img_adam = Adam(image_model.parameters(), lr=opt.img_lr * opt.accumulation_steps, weight_decay=opt.img_weight_decay * opt.accumulation_steps)
            optimizer_txt_adam = Adam(text_model.parameters(), lr=opt.txt_lr * opt.accumulation_steps, weight_decay=opt.txt_weight_decay * opt.accumulation_steps)
            optimizer_gat_adam = Adam(gat_model.parameters(), lr=opt.gat_lr * opt.accumulation_steps, weight_decay=opt.gat_weight_decay * opt.accumulation_steps)
            optimizer_label_adam = Adam(label_model.parameters(), lr=opt.label_lr * opt.accumulation_steps, weight_decay=opt.label_weight_decay * opt.accumulation_steps)
        elif opt.optim == 'AdamW':
            optimizer_img_adam = AdamW(image_model.parameters(), lr=opt.img_lr * opt.accumulation_steps, weight_decay=opt.img_weight_decay * opt.accumulation_steps)
            optimizer_txt_adam = Adam(text_model.parameters(), lr=opt.txt_lr * opt.accumulation_steps, weight_decay=opt.txt_weight_decay * opt.accumulation_steps)
            optimizer_gat_adam = AdamW(gat_model.parameters(), lr=opt.gat_lr * opt.accumulation_steps, weight_decay=opt.gat_weight_decay * opt.accumulation_steps)
            optimizer_label_adam = AdamW(label_model.parameters(), lr=opt.label_lr * opt.accumulation_steps, weight_decay=opt.label_weight_decay * opt.accumulation_steps)
        else:
            optimizer_img_adam = NAdam(image_model.parameters(), lr=opt.img_lr * opt.accumulation_steps, weight_decay=opt.img_weight_decay * opt.accumulation_steps)
            optimizer_txt_adam = NAdam(text_model.parameters(), lr=opt.txt_lr * opt.accumulation_steps, weight_decay=opt.txt_weight_decay * opt.accumulation_steps)
            optimizer_gat_adam = NAdam(gat_model.parameters(), lr=opt.gat_lr * opt.accumulation_steps, weight_decay=opt.gat_weight_decay * opt.accumulation_steps)
            optimizer_label_adam = Adam(label_model.parameters(), lr=opt.label_lr * opt.accumulation_steps, weight_decay=opt.label_weight_decay * opt.accumulation_steps)

    optimizer_img = optimizer_img_adam
    optimizer_txt = optimizer_txt_adam
    optimizer_gat = optimizer_gat_adam
    optimizer_label = optimizer_label_adam

    if opt.is_dual_card:
        tri_loss = TripletLoss(reduction='sum').to("cuda:0", non_blocking=True)
        criterion = nn.BCEWithLogitsLoss().to("cuda:0", non_blocking=True)
        w_loss_func = Wasserstein().to("cuda:0", non_blocking=True)
    else:
        tri_loss = TripletLoss(reduction='sum')
        criterion = nn.BCEWithLogitsLoss()
        w_loss_func = Wasserstein()

    loss = []
    loss_quan = []
    loss_triplet = []
    loss_clas = []
    loss_contra = []
    rec_loss = []
    loss1_list = []
    loss2_list = []
    loss3_list = []
    loss4_list = []
    loss5_list = []
    loss6_list = []
    loss7_list = []
    loss31_list = []
    loss32_list = []
    loss8_list = []
    loss9_list = []
    loss10_list = []
    loss11_list = []
    loss12_list = []
    loss_triplet2 = []
    loss_triplet3 = []
    loss_triplet4 = []
    loss_triplet5 = []
    s_loss = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    train_times = []
    K = 1.5
    ETA = 0.1
    ALPHA = 0.9

    if amp:
        with autocast(dtype=torch.float16):
            if opt.is_dual_card:
                F_buffer = torch.randn(opt.training_size, opt.bit, dtype=torch.float16).to("cuda:0", non_blocking=True)
                G_buffer = torch.randn(opt.training_size, opt.bit, dtype=torch.float16).to("cuda:0", non_blocking=True)
                L_buffer = torch.randn(opt.training_size, opt.bit, dtype=torch.float16).to("cuda:0", non_blocking=True)
            else:
                F_buffer = torch.randn(opt.training_size, opt.bit, dtype=torch.float16).cuda(non_blocking=True)
                G_buffer = torch.randn(opt.training_size, opt.bit, dtype=torch.float16).cuda(non_blocking=True)
                L_buffer = torch.randn(opt.training_size, opt.bit, dtype=torch.float16).cuda(non_blocking=True)
    else:
        if opt.is_dual_card:
                F_buffer = torch.randn(opt.training_size, opt.bit).to("cuda:0", non_blocking=True)
                G_buffer = torch.randn(opt.training_size, opt.bit).to("cuda:0", non_blocking=True)
                L_buffer = torch.randn(opt.training_size, opt.bit).to("cuda:0", non_blocking=True)
        else:
            F_buffer = torch.randn(opt.training_size, opt.bit).cuda(non_blocking=True)
            G_buffer = torch.randn(opt.training_size, opt.bit).cuda(non_blocking=True)
            L_buffer = torch.randn(opt.training_size, opt.bit).cuda(non_blocking=True)

    B = torch.sign(F_buffer + G_buffer + L_buffer)

    scaler = GradScaler()

    start_epoch = 0
    if resume:
        checkpoint_path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
        best_model_path = os.path.join(checkpoint_path, 'best_checkpoint.pth.tar')
        assert os.path.isfile(best_model_path)
        checkpoint_model = torch.load(best_model_path)
        checkpoint = {
                    'mapi2t': checkpoint_model['mapi2t'],
                    'mapt2i': checkpoint_model['mapt2i'],
                    'epoch': checkpoint_model['epoch'] + 1,
                    'img_model': checkpoint_model["img_model"],
                    'img_optimizer': checkpoint_model["img_optimizer"],
                    'txt_model': checkpoint_model["txt_model"],
                    'txt_optimizer': checkpoint_model["txt_optimizer"],
                    'gat_model': checkpoint_model["gat_model"],
                    'gat_optimizer': checkpoint_model["gat_optimizer"],
                    'label_model': checkpoint_model["label_model"],
                    'label_optimizer': checkpoint_model["label_optimizer"]
                }
        best_mapi2t = checkpoint['mapi2t']
        best_mapt2i = checkpoint['mapt2i']
        start_epoch = checkpoint['epoch']
        image_model.load_state_dict(checkpoint['img_model'])
        optimizer_img.load_state_dict(checkpoint['img_optimizer'])
        text_model.load_state_dict(checkpoint['txt_model'])
        optimizer_txt.load_state_dict(checkpoint['txt_optimizer'])
        gat_model.load_state_dict(checkpoint['gat_model'])
        optimizer_gat.load_state_dict(checkpoint['gat_optimizer'])
        label_model.load_state_dict(checkpoint['label_model'])
        optimizer_label.load_state_dict(checkpoint['label_optimizer'])
        print('Load checkpoint at epoch {}.'.format(start_epoch))
        print('Best accuracy so far mapi2t:{}    mapt2i:{}.'.format(best_mapi2t, best_mapt2i))
    for epoch in range(start_epoch, opt.max_epoch):
        if current_rank == 0 and opt.is_distributedDataParallel:
            print("\n")
            print(optimizer_img)
        t1 = time.time()
        loss_qi = 0
        loss_ti = 0
        loss_ql = 0
        loss_ti2 = 0
        loss_ti3 = 0
        loss_ti4 = 0
        loss_ti5 = 0
        loss_qt = 0
        loss_tt = 0
        loss_tt2 = 0
        loss_tt3 = 0
        loss_tt4 = 0
        loss_tt5 = 0
        loss_c = 0
        total = 0
        if opt.is_distributedDataParallel:
            train_sampler.set_epoch(epoch)
        for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):

            if opt.is_dual_card:
                img = img.to("cuda:0", non_blocking=True)
                txt = txt.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            else:
                img = img.cuda(non_blocking=True)
                txt = txt.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            if amp:
                with autocast(dtype=torch.float16):
                    f_l, hid_l, h_l = label_model(label)

                    f_t, hid_t, h_t = text_model(txt)

                    if opt.is_dual_card:
                        f_i, hid_i, h_i = image_model(img.to("cuda:1", non_blocking=True))
                        f_i = f_i.to("cuda:0", non_blocking=True)
                        f_t = f_t.to("cuda:0", non_blocking=True)
                        h_i = h_i.to("cuda:0", non_blocking=True)
                    else:
                        f_i, hid_i, h_i = image_model(img)


                    L_buffer[ind, :] = h_l.data

                    pred = gat_model(f_i, f_t)

                    F = torch.as_tensor(F_buffer)
                    G = torch.as_tensor(G_buffer)
                    L = torch.as_tensor(L_buffer)
                    KLloss_ll = 1.9 * multilabelsimilarityloss_KL(h_l, L, h_l, L)
                    KLloss_lx = multilabelsimilarityloss_KL(h_l, L, h_l, F)
                    KLloss_ly = 0.3 * multilabelsimilarityloss_KL(h_l, L, h_l, G)
                    # print(h_i.shape)
                    F_buffer[ind, :] = h_i.data

                    F = torch.as_tensor(F_buffer)
                    L = torch.as_tensor(L_buffer)
                    KLloss_xx = 1.9 * multilabelsimilarityloss_KL(h_l, L, h_i, F)
                    KLloss_xl = multilabelsimilarityloss_KL(h_l, L, h_i, L)

                    G_buffer[ind, :] = h_t.data
                    L = torch.as_tensor(L_buffer)
                    G = torch.as_tensor(G_buffer)
                    KLloss_yy = 1.9 * multilabelsimilarityloss_KL(h_l, L, h_t, G)
                    KLloss_yl = multilabelsimilarityloss_KL(h_l, L, h_t, L)

                    loss2 = KLloss_ll + KLloss_lx + KLloss_ly + KLloss_xx + KLloss_xl + KLloss_yy + KLloss_yl

                    i_tri = tri_loss(opt, h_i, label, target=h_t, margin=opt.margin)  + \
                            tri_loss(opt, h_i, label, target=h_i, margin=opt.margin)

                    t_tri = tri_loss(opt, h_t, label, target=h_i, margin=opt.margin)  + \
                            tri_loss(opt, h_t, label, target=h_t, margin=opt.margin)

                    i_ql = torch.sum((1. - Func.cosine_similarity(h_i, h_i.detach().sign(), dim=1))) + torch.sum(torch.pow(h_i - h_i.detach().sign(), 2)) + quantizationLoss(h_i, B[ind, :])
                    t_ql = torch.sum((1. - Func.cosine_similarity(h_t, h_t.detach().sign(), dim=1))) + torch.sum(torch.pow(h_t - h_t.detach().sign(), 2)) + quantizationLoss(h_t, B[ind, :])
                    l_ql = torch.sum((1. - Func.cosine_similarity(h_l, h_l.detach().sign(), dim=1))) + torch.sum(torch.pow(h_l - h_l.detach().sign(), 2)) + quantizationLoss(h_l, B[ind, :])


                    loss_quant = i_ql + t_ql + l_ql
                    loss_class = criterion(pred, label)
                    loss1, _, _ = w_loss, _, _ = w_loss_func(h_i, h_t)

                    err = opt.alpha * (i_tri + t_tri) + loss_class + loss_quant + w_loss + loss2
            else:
                f_l, hid_l, h_l = label_model(label)

                f_t, hid_t, h_t = text_model(txt)

                if opt.is_dual_card:
                    f_i, hid_i, h_i = image_model(img.to("cuda:1", non_blocking=True))
                    f_i = f_i.to("cuda:0", non_blocking=True)
                    f_t = f_t.to("cuda:0", non_blocking=True)
                    h_i = h_i.to("cuda:0", non_blocking=True)
                else:
                    f_i, hid_i, h_i = image_model(img)


                L_buffer[ind, :] = h_l.data

                pred = gat_model(f_i, f_t)

                F = torch.as_tensor(F_buffer)
                G = torch.as_tensor(G_buffer)
                L = torch.as_tensor(L_buffer)
                KLloss_ll = 1.9 * multilabelsimilarityloss_KL(h_l, L, h_l, L)
                KLloss_lx = multilabelsimilarityloss_KL(h_l, L, h_l, F)
                KLloss_ly = 0.3 * multilabelsimilarityloss_KL(h_l, L, h_l, G)
                # print(h_i.shape)
                F_buffer[ind, :] = h_i.data

                F = torch.as_tensor(F_buffer)
                L = torch.as_tensor(L_buffer)
                KLloss_xx = 1.9 * multilabelsimilarityloss_KL(h_l, L, h_i, F)
                KLloss_xl = multilabelsimilarityloss_KL(h_l, L, h_i, L)

                G_buffer[ind, :] = h_t.data
                L = torch.as_tensor(L_buffer)
                G = torch.as_tensor(G_buffer)
                KLloss_yy = 1.9 * multilabelsimilarityloss_KL(h_l, L, h_t, G)
                KLloss_yl = multilabelsimilarityloss_KL(h_l, L, h_t, L)

                loss2 = KLloss_ll + KLloss_lx + KLloss_ly + KLloss_xx + KLloss_xl + KLloss_yy + KLloss_yl

                i_tri = tri_loss(opt, h_i, label, target=h_t, margin=opt.margin)  + \
                        tri_loss(opt, h_i, label, target=h_i, margin=opt.margin)

                t_tri = tri_loss(opt, h_t, label, target=h_i, margin=opt.margin)  + \
                        tri_loss(opt, h_t, label, target=h_t, margin=opt.margin)

                i_ql = torch.sum((1. - Func.cosine_similarity(h_i, h_i.detach().sign(), dim=1))) + torch.sum(torch.pow(h_i - h_i.detach().sign(), 2)) + quantizationLoss(h_i, B[ind, :])
                t_ql = torch.sum((1. - Func.cosine_similarity(h_t, h_t.detach().sign(), dim=1))) + torch.sum(torch.pow(h_t - h_t.detach().sign(), 2)) + quantizationLoss(h_t, B[ind, :])
                l_ql = torch.sum((1. - Func.cosine_similarity(h_l, h_l.detach().sign(), dim=1))) + torch.sum(torch.pow(h_l - h_l.detach().sign(), 2)) + quantizationLoss(h_l, B[ind, :])


                loss_quant = i_ql + t_ql + l_ql
                loss_class = criterion(pred, label)
                loss1, _, _ = w_loss, _, _ = w_loss_func(h_i, h_t)

                err = opt.alpha * (i_tri + t_tri) + loss_class + loss_quant + w_loss + loss2

            if opt.accumulation_steps != 1:
                if amp:
                    scaler.scale(err).backward()
                else:
                    err.backward()
                err = err / opt.accumulation_steps
                if (i + 1) % opt.accumulation_steps == 0:
                    if amp:
                        scaler.step(optimizer_txt)
                        scaler.step(optimizer_img)
                        scaler.step(optimizer_gat)
                        scaler.step(optimizer_label)
                        scaler.update()
                    else:
                        optimizer_txt.step()
                        optimizer_img.step()
                        optimizer_gat.step()
                        optimizer_label.step()
                        optimizer_txt.zero_grad(set_to_none=True)
                        optimizer_img.zero_grad(set_to_none=True)
                        optimizer_gat.zero_grad(set_to_none=True)
                        optimizer_label.zero_grad(set_to_none=True)


            else:
                if amp:
                    scaler.scale(err).backward()
                    scaler.step(optimizer_txt)
                    scaler.step(optimizer_img)
                    scaler.step(optimizer_gat)
                    scaler.step(optimizer_label)
                    scaler.update()
                else:
                    optimizer_txt.zero_grad(set_to_none=True)
                    optimizer_img.zero_grad(set_to_none=True)
                    optimizer_gat.zero_grad(set_to_none=True)
                    optimizer_label.zero_grad(set_to_none=True)
                    err.backward()
                    optimizer_txt.step()
                    optimizer_img.step()
                    optimizer_gat.step()
                    optimizer_label.step()

            loss_c = loss_class + loss_c
            loss_qi = i_ql + loss_qi
            loss_ti = i_tri + loss_ti
            loss_ql = l_ql + loss_ql
            total = err + total
            loss_qt = t_ql + loss_qt
            loss_tt = t_tri + loss_tt

        for param in optimizer_img.param_groups:
            param['lr'] = opt.img_lr * 0.5 * (1 + np.cos((i + 1) / (epoch + 1) * np.pi))

        for param in optimizer_txt.param_groups:
            param['lr'] = opt.txt_lr * 0.5 * (1 + np.cos((i + 1) / (epoch + 1) * np.pi))

        for param in optimizer_gat.param_groups:
            param['lr'] = opt.gat_lr * 0.5 * (1 + np.cos((i + 1) / (epoch + 1) * np.pi))

        for param in optimizer_label.param_groups:
            param['lr'] = opt.label_lr * 0.5 * (1 + np.cos((i + 1) / (epoch + 1) * np.pi))

        B = torch.sign(F_buffer + G_buffer + L_buffer)

        if current_rank == 0:
            loss_quan.append([loss_qi.item(), loss_qt.item(), loss_ql.item()])
            loss_triplet.append([loss_ti.item(), loss_tt.item()])
            loss_clas.append(loss_c.item())
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            loss.append(total.item())
            print('...epoch: %3d, img_net_loss: %3.3f' % (epoch + 1, loss_quan[-1][0] + loss_triplet[-1][0]))
            print('...epoch: %3d, txt_net_loss: %3.3f' % (epoch + 1, loss_quan[-1][1] + loss_triplet[-1][1]))
            print('...epoch: %3d, label_net_loss: %3.3f' % (epoch + 1, loss_quan[-1][2]))
            print('...epoch: %3d, total_loss: %3.3f' % (epoch + 1, loss[-1] + loss_quan[-1][0] + loss_quan[-1][1]))

        delta_t = time.time() - t1

        if opt.vis_env:
            if current_rank == 0:
                vis.plot('img_loss_quan', loss_quan[-1][0])
                vis.plot('img_loss_triplet', loss_triplet[-1][0])
                vis.plot('txt_loss_quan', loss_quan[-1][1])
                vis.plot('label_loss_quan', loss_quan[-1][2])
                vis.plot('txt_loss_triplet', loss_triplet[-1][1])
                vis.plot('loss_class', loss_clas[-1])
                vis.plot('loss1', loss1_list[-1])
                vis.plot('loss2', loss2_list[-1])
                vis.plot('total_loss', loss[-1])

        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            with torch.no_grad():
                mapi2t, mapt2i, qBX, qBY, rBX, rBY = valid(image_model, text_model, inp_file, i_query_dataloader, i_db_dataloader,
                                        t_query_dataloader, t_db_dataloader,
                                        query_labels, db_labels)
            if current_rank == 0:
                print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
                mapi2t_list.append(mapi2t)
                mapt2i_list.append(mapt2i)
                train_times.append(delta_t)

            if opt.vis_env:
                d = {
                        'mapi2t': mapi2t,
                        'mapt2i': mapt2i
                }
                if current_rank == 0:
                    vis.plot_many(d)
            if current_rank == 0:
                if 0.5 * (mapi2t + mapt2i) > max_average:
                    max_mapi2t = mapi2t
                    max_mapt2i = mapt2i
                    current_acc = 0.5 * (mapi2t + mapt2i)
                    max_average = max(current_acc, max_average)

                    checkpoint = {
                        'mapi2t': mapi2t,
                        'mapt2i': mapt2i,
                        'epoch': epoch + 1,
                        'img_model': image_model.state_dict(),
                        'img_optimizer': optimizer_img.state_dict(),
                        'txt_model': text_model.state_dict(),
                        'txt_optimizer': optimizer_txt.state_dict(),
                        'gat_model': gat_model.state_dict(),
                        'gat_optimizer': optimizer_gat.state_dict(),
                        'label_model': label_model.state_dict(),
                        'label_optimizer': optimizer_label.state_dict()
                        }
                    checkpoint_path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
                    save_mat(query_img=qBX, query_txt=qBY, retrieval_img=rBX, retrieval_txt=rBY,
                             query_labels=query_labels, retrieval_labels=db_labels, save__dir=checkpoint_path,
                             output_dim=opt.bit, map=max_mapi2t, flag=opt.flag, mode_name="i2t")
                    save_mat(query_img=qBX, query_txt=qBY, retrieval_img=rBX, retrieval_txt=rBY,
                             query_labels=query_labels, retrieval_labels=db_labels, save__dir=checkpoint_path,
                             output_dim=opt.bit, map=max_mapt2i, flag=opt.flag, mode_name="t2i")
                    best_model_path = os.path.join(checkpoint_path, 'best_checkpoint.pth.tar')
                    torch.save(checkpoint, best_model_path)
                    

    print('...training procedure finish')
    if opt.valid:
        if current_rank == 0:
            print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i, qBX, qBY, rBX, rBY = valid(image_model, text_model, inp_file, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader, query_labels, db_labels)
        if current_rank == 0:
            print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        

    if current_rank == 0:
        path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
        with open(os.path.join(path, 'result.pkl'), 'wb') as f:
            pickle.dump([train_times, mapi2t_list, mapt2i_list], f)
        
        return max_average, err.item()


def valid(image_model, text_model, inp, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader, query_labels, db_labels):
    image_model.eval()
    text_model.eval()

    qBX = generate_img_code(image_model, inp, x_query_dataloader, opt.query_size, opt.bit)
    qBY = generate_txt_code(text_model, inp, y_query_dataloader, opt.query_size, opt.bit)
    rBX = generate_img_code(image_model, inp, x_db_dataloader, opt.db_size, opt.bit)
    rBY = generate_txt_code(text_model, inp, y_db_dataloader, opt.db_size, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

    image_model.train()
    text_model.train()
    return mapi2t.item(), mapt2i.item(), qBX, qBY, rBX, rBY


def test(**kwargs):
    opt.parse(kwargs)

    images, tags, labels, adj_file, inp_file = load_data(opt.data_path, opt.adj, opt.inp, type=opt.dataset)
    if opt.is_dual_card:
        inp_file = torch.from_numpy(inp_file).to("cuda:0", non_blocking=True)
    else:
        inp_file = torch.from_numpy(inp_file).cuda(non_blocking=True)

    bits = [opt.bit]
    with torch.no_grad():
        for bit in bits:
            image_model = IMG(opt.hidden_dim, bit, opt.dropout, opt.num_label, adj_file).cuda()
            text_model = TXT(opt.text_dim, opt.hidden_dim, bit, opt.dropout, opt.num_label, adj_file).cuda()
            path = 'checkpoints/' + opt.dataset + '_' + str(bit)
            load_model(image_model, path)
            load_model(text_model, path)
            image_model.eval()
            text_model.eval()

            i_query_data = Dataset(opt, images, tags, labels, test='image.query')
            i_db_data = Dataset(opt, images, tags, labels, test='image.db')
            t_query_data = Dataset(opt, images, tags, labels, test='text.query')
            t_db_data = Dataset(opt, images, tags, labels, test='text.db')

            i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False, num_workers=16, pin_memory=True)
            i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False, num_workers=16, pin_memory=True)
            t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False, num_workers=16, pin_memory=True)
            t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False, num_workers=16, pin_memory=True)

            query_labels, db_labels = i_query_data.get_labels()
            if opt.is_dual_card:
                query_labels = query_labels.to("cuda:0", non_blocking=True)
                db_labels = db_labels.to("cuda:0", non_blocking=True)
            else:
                query_labels = query_labels.cuda(non_blocking=True)
                db_labels = db_labels.cuda(non_blocking=True)

            qBX = generate_img_code(image_model, inp_file, i_query_dataloader, opt.query_size, bit)
            qBY = generate_txt_code(text_model, inp_file, t_query_dataloader, opt.query_size, bit)
            rBX = generate_img_code(image_model, inp_file, i_db_dataloader, opt.db_size, bit)
            rBY = generate_txt_code(text_model, inp_file, t_db_dataloader, opt.db_size, bit)

            mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
            mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)
            if current_rank == 0:
                print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
                scio.savemat('./checkpoints/{}-{}-hash.mat'.format(opt.dataset, bit), {
                    'Qi': qBX.cpu().numpy(), 'Qt': qBY.cpu().numpy(), 'Di': rBX.cpu().numpy(), 'Dt': rBY.cpu().numpy(),
                    'retrieval_L': db_labels.cpu().numpy(), 'query_L': query_labels.cpu().numpy()
                })


def generate_img_code(model, inp, test_dataloader, num, bit):
    if opt.is_dual_card:
        B = torch.zeros(num, bit).to("cuda:0", non_blocking=True)
    else:
        B = torch.zeros(num, bit).cuda(non_blocking=True)
    for i, (input_data) in tqdm(enumerate(test_dataloader)):
        if opt.is_dual_card:
            input_data = input_data.to("cuda:1", non_blocking=True)
        else:
            input_data = input_data.cuda(non_blocking=True)

        if opt.is_distributedDataParallel or is_dp:
            b = model.module.generate_img_code(input_data)
        else:
            b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, inp, test_dataloader, num, bit):
    if opt.is_dual_card:
        B = torch.zeros(num, bit).to("cuda:0", non_blocking=True)
    else:
        B = torch.zeros(num, bit).cuda(non_blocking=True)
    for i, (input_data) in tqdm(enumerate(test_dataloader)):
        if opt.is_dual_card:
            input_data = input_data.to("cuda:0", non_blocking=True)
        else:
            input_data = input_data.cuda(non_blocking=True)

        if opt.is_distributedDataParallel or is_dp:
            b = model.module.generate_txt_code(input_data)
        else:
            b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B

def calculate_distance(x, y):
    d = torch.sqrt(torch.sum((x - y) ** 2))
    return d


def calculate_distance_matrix(x, y):
    d = torch.cdist(x, y)
    return d


def cal_B(D):
    (n1, n2) = D.shape
    DD = torch.square(D)                
    Di = torch.sum(DD, dim=1) / n1       
    Dj = torch.sum(DD, dim=0) / n1         
    Dij = torch.sum(DD) / (n1 ** 2)    
    B = torch.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)   # 计算b(ij)
    return B

def calc_loss(loss):
    l = 0.
    for v in loss.values():
        l += v[-1]
    return l

def avoid_inf(x):
    return torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.max(torch.zeros_like(x), x)


def load_model(model, path):
    if path is not None:
        if opt.is_distributedDataParallel:
            model.module.load(os.path.join(path, model.module.module_name + '.pth'))
        else:
            model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    if current_rank == 0:
        if opt.is_distributedDataParallel:
            model.module.save(model.module.module_name + '.pth', path)
        else:
            model.save(model.module_name + '.pth', path)

if __name__ == '__main__':
    train(flag = args.flag, bit = args.bit, batch_size = args.batch, max_epoch=300)