import pickle
import torch
from torch import Tensor
import time
from typing import *

def load_index(dataset):
    # load_index1
    index_1 = {}
    with open('../data/DBP15K/DBP_1/{}/index_1'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_index = line.split('\t')
            index_1[ent_id] = int(ent_index)
    # load_index2
    index_2 = {}
    with open('../data/DBP15K/DBP_1/{}/index_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_index = line.split('\t')
            index_2[ent_id] = int(ent_index)
    return index_1, index_2

def load_reverse_index(dataset):
    # load_index1
    r_index_1 = {}
    with open('../data/DBP15K/DBP_1/{}/index_1'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_index = line.split('\t')
            r_index_1[ent_index] = int(ent_id)
    # load_index2
    r_index_2 = {}
    with open('../data/DBP15K/DBP_1/{}/index_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_index = line.split('\t')
            r_index_2[ent_index] = int(ent_id)
    return r_index_1, r_index_2


def pair_load(dataset):
    pair = []
    with open('../data/DBP15K/DBP_1/{}/ill_ent_ids'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id1, ent_id2 = line.split('\t')
            pair.append((int(ent_id1), int(ent_id2)))

    print('pair length:{}'.format(len(pair)))
    return pair, len(pair)


def pkl_load(dataset):
    with open('../data/DBP15K/pkl/{}_GA_id_img_feature_dict.pkl'.format(dataset), 'rb') as f:
        image_emb_dic = pickle.load(f)

    return image_emb_dic


def ent_load(dataset):
    ent_1 = {}
    with open('../data/DBP15K/DBP_1/{}/ent_ids_1'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent_1[int(ent_id)] = ent_name
    ent_2 = {}
    with open('../data/DBP15K/DBP_1/{}/ent_ids_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent_2[int(ent_id)] = ent_name

    return ent_1, ent_2


def ent_id_load(dataset):
    ent_id_1 = {}
    with open('../data/DBP15K/DBP_1/{}/ent_ids_1'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent_id_1[ent_name] =  int(ent_id)
    ent_id_2 = {}
    with open('../data/DBP15K/DBP_1/{}/ent_ids_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent_id_2[ent_name] = int(ent_id)

    return ent_id_1, ent_id_2


def dev_pair_load(dataset):
    dev_pair = []
    with open('../data/DBP15K/DBP_1/{}/dev_ent_ids'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id1, ent_id2 = line.split('\t')
            dev_pair.append((int(ent_id1), int(ent_id2)))

    print('dev pair length:{}'.format(len(dev_pair)))
    return dev_pair


def sinkhorn_test(scores, len_pair,device):
    scores = torch.Tensor(scores).to(device)

    sim_mat_r = 1 - scores

    # matrix_sinkhorn
    if sim_mat_r.dim == 3:
        M = sim_mat_r
    else:
        M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
    m, n = sim_mat_r.shape
    a = torch.ones([1, m], requires_grad=False, device=device)
    b = torch.ones([1, n], requires_grad=False, device=device)
    P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
    P = view2(P)

    # evaluate_sim
    result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_pair),
                                                   torch.arange(len_pair)], dim=0),
                                 sim_x2y=P,
                                 no_csls=True)
    return result


@torch.no_grad()
def evaluate_sim_matrix(link, sim_x2y, sim_y2x=None, ignore=(None, None), start="\t", no_csls=True, mrr=False):
    start_outer = start
    start = start + start
    device = link.device
    sim_x2y = sim_x2y.to(device)
    if sim_x2y.is_sparse:
        sim_x2y = sim_x2y.to_dense()
    MRR = 'MRR'
    match_sim0, match_id0, sim_matrix0 = get_topk_sim(sim_x2y)
    result = get_hit_k(match_id0, link, 0, ignore=ignore[0], start=start)
    result["MR"] = get_mr(link, sim_matrix0, 0, start=start)
    result["MRR"] = get_mrr(link, sim_matrix0, 0, start=start)
    # mr
    if sim_y2x is not None:
        sim_y2x = sim_y2x.to(device)
        if sim_y2x.is_sparse:
            sim_y2x = sim_y2x.to_dense()
        match_sim1, match_id1, sim_matrix1 = get_topk_sim(sim_y2x)

        result_rev = get_hit_k(match_id1, link, 1, ignore=ignore[1], start=start)
        result_rev[MRR] = get_mrr(link, sim_matrix1, 1, start=start)
        if no_csls:
            return result, result_rev
        print(start_outer + '------csls')
        match_sim0, match_id0, sim_matrix0 = get_csls_sim(sim_matrix0, match_sim0, match_sim1)
        match_sim1, match_id1, sim_matrix1 = get_csls_sim(sim_matrix1, match_sim1, match_sim0)

        result_csls_0 = get_hit_k(match_id0, link, 0, ignore=ignore[0], start=start)
        result_csls_0[MRR] = get_mrr(link, sim_matrix0, 0, start=start)

        result_csls_1 = get_hit_k(match_id1, link, 1, ignore=ignore[1], start=start)
        result_csls_1[MRR] = get_mrr(link, sim_matrix1, 1, start=start)
        return result, result_rev, result_csls_0, result_csls_1
    else:
        return result


def get_csls_sim(sim_matrix: Tensor, dist0: Tensor, dist1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    k = dist1.size(1)
    sim_matrix = csls_impl(sim_matrix, dist0, dist1)
    r, rid = torch.topk(sim_matrix, k, dim=1)
    return r, rid, sim_matrix


def csls_impl(sim_matrix, dist0, dist1) -> Tensor:
    dist0 = dist0.mean(dim=1).view(-1, 1).expand_as(sim_matrix)
    dist1 = dist1.mean(dim=1).view(1, -1).expand_as(sim_matrix)
    sim_matrix = sim_matrix * 2 - dist0 - dist1
    return sim_matrix




def get_hit_k(match_id: Tensor, link: Tensor, src=0, k_list=(1, 5, 10), ignore=None, start=""):
    trg = 1 - src
    total = link.size(1)
    if ignore is not None:
        match_id[ignore] = torch.ones_like(match_id[ignore], device=match_id.device, dtype=torch.long) * -1
        ignore_sum = ignore.clone()
        ignore_sum[link[src]] = False
        print(start + "total ignore:", ignore.sum(), ", valid ignore", ignore_sum.sum())
        total = total - ignore_sum.sum()
    print(start + "total is ", total)
    match_id = match_id[link[src]]
    link: Tensor = link[trg]
    hitk_result = {}
    for k in k_list:
        if k > match_id.size(1):
            break
        match_k = match_id[:, :k]
        link_k = link.view(-1, 1).expand(-1, k)
        hit_k = (match_k == link_k).sum().item()
        hitk_result['hits@{}'.format(k)] = hit_k / total
        print("{2}hits@{0} is {1}".format(k, hit_k / total, start))
    return hitk_result





@torch.no_grad()
def get_mrr(link: Tensor, sim_matrix: Tensor, which=0, batch_size=4096, start="\t"):
    all = link.size(1)
    curr = 0
    mrr = torch.tensor(0.).to(link.device)
    while curr < all:
        begin, end = curr, min(curr + batch_size, all)
        curr = end
        src, trg = link[which, begin:end], link[1 - which, begin:end]
        sim = sim_matrix[src]
        sim = torch.argsort(sim, dim=1, descending=True)
        sim = torch.argmax((sim == trg.view(-1, 1)).to(torch.long), dim=1, keepdim=False)
        mrr += (1.0 / (sim + 1).to(float)).sum()
    mrr /= all
    print("{0}MRR is {1}".format(start, mrr))
    return mrr.item()

@torch.no_grad()
def get_mr(link: Tensor, sim_matrix: Tensor, which=0, batch_size=4096, start="\t"):
    all = link.size(1)
    curr = 0
    mr = torch.tensor(0.).to(link.device)
    while curr < all:
        begin, end = curr, min(curr + batch_size, all)
        curr = end
        src, trg = link[which, begin:end], link[1 - which, begin:end]
        sim = sim_matrix[src]
        sim = torch.argsort(sim, dim=1, descending=True)
        sim = torch.argmax((sim == trg.view(-1, 1)).to(torch.long), dim=1, keepdim=False)
        # +1 for the index
        mr += (sim + 1).sum()

    mr /= all
    print("{0}MR is {1}".format(start, mr))
    return mr.item()



def get_topk_sim(sim: Tensor, k_ent=10) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.topk(sim, k=k_ent) + (sim,)
# torch.topk() : 用来获取张量或者数组中最大或者最小的元素以及索引位置


def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def sinkhorn(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor, eps: float,
             max_iters: int = 100, stop_thresh: float = 1e-3):
    """
    Compute the Sinkhorn divergence between two sum of dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    :param a: A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = [m, n]
    :param b: A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = [m, n]
    :param M: A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V.
             i.e. shape = [m, n, n] and each i.e. M[k, i, j] = ||u[k,_i] - v[k, j]||
    :param eps: The reciprocal of the sinkhorn regularization parameter
    :param max_iters: The maximum number of Sinkhorn iterations
    :param stop_thresh: Stop if the change in iterates is below this value
    :return:
    """
    # a and b are tensors of size [m, n]
    # M is a tensor of size [m, n, n]

    nb = M.shape[0]
    m = M.shape[1]
    n = M.shape[2]

    if a.dtype != b.dtype or a.dtype != M.dtype:
        raise ValueError("Tensors a, b, and M must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(M) = %s"
                         % (str(a.dtype), str(b.dtype), str(M.dtype)))
    if a.device != b.device or a.device != M.device:
        raise ValueError("Tensors a, b, and M must be on the same device got: "
                         "device(a) = %s, device(b) = %s, device(M) = %s"
                         % (a.device, b.device, M.device))
    if len(M.shape) != 3:
        raise ValueError("Got unexpected shape for M (%s), should be [nb, m, n] where nb is batch size, and "
                         "m and n are the number of samples in the two input measures." % str(M.shape))
    if torch.Size(a.shape) != torch.Size([nb, m]):
        raise ValueError("Got unexpected shape for tensor a (%s). Expected [nb, m] where M has shape [nb, m, n]." %
                         str(a.shape))

    if torch.Size(b.shape) != torch.Size([nb, n]):
        raise ValueError("Got unexpected shape for tensor b (%s). Expected [nb, n] where M has shape [nb, m, n]." %
                         str(b.shape))

    # Initialize the iteration with the change of variable
    u = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
    v = eps * torch.log(b)

    M_t = torch.transpose(M, 1, 2)

    def stabilized_log_sum_exp(x):
        # max_x = torch.max(x, dim=2)[0]
        # x = x - max_x.unsqueeze(2)
        # ret = torch.log(torch.sum(torch.exp(x), dim=2)) + max_x
        # return ret
        return torch.logsumexp(x, -1)

    for current_iter in range(max_iters):
        u_prev = u
        v_prev = v

        summand_u = (-M + v.unsqueeze(1)) / eps
        u = eps * (torch.log(a) - stabilized_log_sum_exp(summand_u))

        summand_v = (-M_t + u.unsqueeze(1)) / eps
        v = eps * (torch.log(b) - stabilized_log_sum_exp(summand_v))

        err_u = torch.max(torch.sum(torch.abs(u_prev - u), dim=1))
        err_v = torch.max(torch.sum(torch.abs(v_prev - v), dim=1))

        if err_u < stop_thresh and err_v < stop_thresh:
            break

    log_P = (-M + u.unsqueeze(2) + v.unsqueeze(1)) / eps

    P = torch.exp(log_P)

    return P
