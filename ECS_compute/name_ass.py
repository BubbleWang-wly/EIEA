import argparse
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
from utils import *
import pickle
from os.path import join as pjoin


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

# path = '/home/wluyao/code/EA-agent/CM/data/MSP_results/FB15K-DB15K/Attr1.npy'
# target_embedding = np.load(path)
# print(target_embedding.shape)
# (10277, 10277)


parser = argparse.ArgumentParser(description='Explicit Alignment (Name) for PathFusion')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])

args = parser.parse_args()
print(args.dataset)


source_dataset, target_dataset = args.dataset.split('-')
source_dataset = source_dataset.lower()
target_dataset = target_dataset.lower()


if source_dataset == 'db15k':
    source_name_path = '/data_extend/wluyao/code/MMEA-dataset-process/entity/entity_name/DB_name.txt'
    target_name_path = '/data_extend/wluyao/code/MMEA-dataset-process/entity/entity_name/FB_DB_name.txt'
    dataset = 'FB15K-DB15K'
else:
    source_name_path = '/data_extend/wluyao/code/MMEA-dataset-process/entity/entity_name/YAGO_name.txt'
    target_name_path = '/data_extend/wluyao/code/MMEA-dataset-process/entity/entity_name/FB_YAGO_name.txt'
    dataset = 'FB15K-YAGO15K'
sourceid_name ={}
targetid_name ={}

with open(source_name_path, 'r') as f:
    for line in f:
        line = line.strip()
        id, name = line.split(' ')
        sourceid_name[int(id)] = name

with open(target_name_path, 'r') as f:
    for line in f:
        line = line.strip()
        id, name = line.split(' ')
        targetid_name[int(id)] = name

print('min source id:', min(sourceid_name.keys()))
print('max source id:', max(sourceid_name.keys()))
print('source entity num:', len(sourceid_name))

print('min target id:', min(targetid_name.keys()))
print('max target id:', max(targetid_name.keys()))
print('target entity num:', len(targetid_name))

dev_pair = []
with open(pjoin('/home/qb/wluyao/code/EA-agent/PathFusion-main/data', args.dataset, 'dev_ent_ids'), 'r') as f:
    for line in f:
        line = line.strip()
        ent_id1, ent_id2 = line.split('\t')
        dev_pair.append((int(ent_id1), int(ent_id2)))


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = SentenceTransformer('/data_extend/wluyao/pretrain_model/Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
model.eval()

batch_size = 128
source_name_embedding = []
target_name_embedding = []

sourceid_name_lt = list(sourceid_name.values())
targetid_name_lt = list(targetid_name.values())

for i in trange(0, len(sourceid_name_lt), batch_size):
    key_sents = sourceid_name_lt[i:i+batch_size]
    source_name_embedding.append(model.encode(key_sents))
source_name_embedding = np.concatenate(source_name_embedding, axis=0)

for i in trange(0, len(targetid_name_lt), batch_size):
    key_sents = targetid_name_lt[i:i+batch_size]
    target_name_embedding.append(model.encode(key_sents))
target_name_embedding = np.concatenate(target_name_embedding, axis=0)

scores_all = np.matmul(source_name_embedding, target_name_embedding.T)
print(scores_all.shape)

scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
for i in range(len(dev_pair)):
    for j in range(len(dev_pair)):
        scores[i][j] = scores_all[dev_pair[i][0]][dev_pair[j][1] - len(sourceid_name)]

print(scores.shape)
# save the scores as .npy file
np.save(f'/home/qb/wluyao/code/EA-agent/CM/data/MSP_results/{dataset}/name.npy', scores)
sinkhorn_test(scores, scores.shape[0],  device=device)