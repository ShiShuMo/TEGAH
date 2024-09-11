import torch
from scipy.io import loadmat
from tqdm import tqdm


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

bits = [64, 32, 16]
datasets = ['mir', 'nus', 'coco']
for dataset in datasets:
    for bit in bits:
        mat_file = "hashcodes/{}-ours-{}-i2t.mat".format(bit, dataset)

        result = loadmat(mat_file)
        qBX = torch.from_numpy(result['q_img'][:]).cuda()  # image query
        qBY = torch.from_numpy(result['q_txt'][:]).cuda()  # text query
        rBX = torch.from_numpy(result['r_img'][:]).cuda()  # image retrieval
        rBY = torch.from_numpy(result['r_txt'][:]).cuda()  # text retrieval
        query_L = torch.from_numpy(result['q_l'][:]).cuda()  # query label
        retrieval_L = torch.from_numpy(result['r_l'][:]).cuda()  # retrieval label

        map_i2t = calc_map_k(qB=qBX, rB=rBY, query_label=query_L, retrieval_label=retrieval_L)
        map_t2i = calc_map_k(qB=qBY, rB=rBX, query_label=query_L, retrieval_label=retrieval_L)

        print("dataset: {}, bits: {}, map_i2t: {:.4f}, map_t2i: {:.4f}".format(dataset, bit, round(map_i2t.item(), 4), round(map_t2i.item(), 4)))
