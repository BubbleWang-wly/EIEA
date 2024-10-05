import argparse
import numpy as np
from tqdm import tqdm
import torch
import pickle
# from utils_db_0_8 import *
# from utils import *
from utils import *

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


# dataset_path = 'DB15K-FB15K'
# dataset_path = 'YAGO15K-FB15K'
dataset_path_list = ['DB15K-FB15K', 'YAGO15K-FB15K']
for dataset_path in dataset_path_list:
    print(dataset_path)
    source_dataset, target_dataset = dataset_path.split('-')
    dataset = target_dataset + '-' + source_dataset
    source_dataset = source_dataset.lower()
    target_dataset = target_dataset.lower()
    use_img_num = 6


    if source_dataset == 'db15k':
        from utils_db_0_8 import *
    else:
        from utils_yb_0_8 import *


    # load source image embedding
    with open('../data/image_embed_1/{}.npy'.format(source_dataset), 'rb') as f:
        source_embedding = pickle.load(f)
        # source_embedding = np.load(f)

    # load source entity id to image id mapping
    with open('../data/image_embed_1/{}'.format(source_dataset), 'rb') as f:
        source_id2img = pickle.load(f)


    print('source embedding shape: {}'.format(source_embedding.shape))
    print('source id2img length: {}'.format(len(source_id2img)))


    # load target image embedding
    with open('../data/image_embed_1/{}.npy'.format(target_dataset), 'rb') as f:
        # target_embedding = np.load(f)
        target_embedding = pickle.load(f)

    # load target entity id to image id mapping
    with open('../data/image_embed_1/{}'.format(target_dataset), 'rb') as f:
        target_id2img = pickle.load(f)


    print('target embedding shape: {}'.format(target_embedding.shape))
    print('target id2img length: {}'.format(len(target_id2img)))
    print('source target entity num:', len(source2id), len(target2id))
    # score init as -float('inf')


    with open(pjoin('../data/MMEA-data/seed0.8', dataset, 'dev_ent_ids'), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id1, ent_id2 = line.split()
            dev_pair.append((int(ent_id1), int(ent_id2)))


    image_scores = np.zeros((len(source2id), len(target2id)))
    image_scores = -float('inf') * np.ones((len(source2id), len(target2id)))


    for i in tqdm(range(len(source2id))):
        for j in range(len(target2id)):
            for ii in range(min(use_img_num, len(source_id2img[i]))):
                for jj in range(min(use_img_num, len(target_id2img[j]))):
                    image_scores[i, j] = max(image_scores[i, j],
                                             np.dot(source_embedding[source_id2img[i][ii]],
                                                    target_embedding[target_id2img[j][jj]]))

    dev_pair = np.array(dev_pair)
    scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    for i, l in enumerate(dev_pair[:, 0]):
        for j, r in enumerate(dev_pair[:, 1]):
            scores[i][j] = image_scores[l][r - len(source2id)] if image_scores[l][r - len(source2id)] != -float('inf') else 0

    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))


    # save the scores as .npy file
    # np.save(f'/data_extend/wluyao/code/CM_msp/{args.dataset}/Vis.npy', scores)
    print(scores.shape)
    np.save(f'../data/ECS_results/seed0.8/{dataset}/Vis.npy', scores)

    # evaluate side_ass result for Vis

    # scores = torch.Tensor(scores)
    # scores = matrix_sinkhorn(1 - scores)
    #
    # sparse_eval.evaluate_sim_matrix(link=torch.stack([torch.arange(len(dev_pair)),
    #                                         torch.arange(len(dev_pair))], dim=0),
    #                                         sim_x2y=scores,
    #                                         no_csls=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    sinkhorn_test(scores, scores.shape[0], device=device)


"""
    DB15K-FB15K
    0
    10277
    source attr num: 225
    target attr num: 104
    source attr value num: 25796
    target attr value num: 17134
    source embedding shape: (66989, 1536)
    source id2img length: 11824
    target embedding shape: (88402, 1536)
    target id2img length: 14834
    source target entity num: 12842 14951
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12842/12842 [3:14:25<00:00,  1.10it/s]
    (2569, 2569)
                    total is  2569
                    hits@1 is 0.20007785130400935
                    hits@5 is 0.2962242117555469
                    hits@10 is 0.34955235500194626
                    MR is 287.25885009765625
                    MRR is 0.2517918348312378

    YAGO15K-FB15K
    source entity num: 15404
    target entity num: 14951
    all entity num: 30355
    min source id: 0
    max source id: 15403
    min target id: 0
    max target id: 14950
    min all id: 0
    max all id: 30354
    seeds num: 11199
    id2attrs num: 30355
    source attr num: 7
    target attr num: 104
    source attr value num: 12645
    target attr value num: 14824
    source embedding shape: (59144, 1536)
    source id2img length: 10406
    target embedding shape: (88402, 1536)
    target id2img length: 14834
    source target entity num: 15404 14951
    100%|????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 15404/15404 [2:59:06<00:00,  1.43it/s]
    (2239, 2239)
                    total is  2239
                    hits@1 is 0.21170165252344797
                    hits@5 is 0.30951317552478785
                    hits@10 is 0.3617686467172845
                    MR is 254.8329620361328
                    MRR is 0.26370739936828613


"""