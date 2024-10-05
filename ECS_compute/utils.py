from torch import Tensor
import time
import torch
from typing import *

def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)


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
