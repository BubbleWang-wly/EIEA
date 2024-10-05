import argparse
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
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


parser = argparse.ArgumentParser(description='side_ass process (Attr) for PathFusion')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])

args = parser.parse_args()

source_dataset, target_dataset = args.dataset.split('-')


if source_dataset == 'DB15K':
    from utils_db import *
else:
    from utils_yb import *

source_keyValue_sents = source_attr_value_list
target_keyValue_sents = target_attr_value_list


# model = SentenceTransformer('/home/qb/wluyao/code/pretrain_model/roberta-base-nli-stsb-mean-tokens/').to(device)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# model = SentenceTransformer('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt').to(device)
# model = SentenceTransformer('/home/qb/wluyao/code/pretrain_model/roberta-base-nli-stsb-mean-tokens/').to(device)
model = SentenceTransformer('/data_extend/wluyao/pretrain_model/Roberta_finetuning_semantic_similarity_stsb_multi_mt').to(device)

# print(device)

source_key_embeddings = []
target_key_embeddings = []
source_value = []
target_value = []

batch_size = 128
for i in trange(0, len(source_keyValue_sents), batch_size):
    key_sents = source_keyValue_sents[i:i + batch_size]

    for j in range(len(key_sents)):
        try:
            # source_value: 存储value值
            # 但是不是所有value都是数目，也就是把数字的value存储，无数字的用0代替
            source_value.append(float(key_sents[j].split(' ')[1]))
        except:
            source_value.append(0)
        key_sents[j] = key_sents[j].split(' ')[0]
        # key_sents：只保留了[0],也就是说只保留属性名

    source_key_embeddings.append(model.encode(key_sents))
source_key_embeddings = np.concatenate(source_key_embeddings, axis=0)

for i in tqdm(range(0, len(target_keyValue_sents), batch_size)):
    target_key_sents = target_keyValue_sents[i:i + batch_size]
    for j in range(len(target_key_sents)):
        try:
            target_value.append(float(target_key_sents[j].split(' ')[1]))
        except:
            target_value.append(0)
        target_key_sents[j] = target_key_sents[j].split(' ')[0]
    target_key_embeddings.append(model.encode(target_key_sents))
target_key_embeddings = np.concatenate(target_key_embeddings, axis=0)


source_value = np.array(source_value)[:, np.newaxis]  # np.newaxis 插入新维度 source_value.shape (25796, 1)
target_value = np.array(target_value)[np.newaxis, :]  # target_value.shape (1, 17134)
scores_key = np.matmul(source_key_embeddings, target_key_embeddings.T)  # scores_key.shape (25796, 17134)
scores_value = 1 / (np.abs(source_value - target_value) + 1e-3)

attr2attr = scores_key * scores_value

source2target = source2attr @ attr2attr @ target2attr.T

# scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
# for i in range(len(dev_pair)):
#     for j in range(len(dev_pair)):
#         scores[i][j] = source2target[dev_pair[i][0]][dev_pair[j][1] - len(source2id)]

scores = (source2target - source2target.min()) / (source2target.max() - source2target.min())  # 归一化
# scores = (scores - scores.min()) / (scores.max() - scores.min())  # 归一化

# save the scores as .npy file
np.save(f'/data_extend/wluyao/code/CM_msp/{args.dataset}/Attr1.npy', scores)


# evaluate side_ass result for Attr

sinkhorn_test(scores, scores.shape[0], device=device)