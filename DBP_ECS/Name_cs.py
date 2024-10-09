import json
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import trange
import numpy as np
from torch.nn import functional as F
from pkl_read import ent_load, load_index, dev_pair_load, sinkhorn_test

def load_name_json(dataset):
    json_path = r'../data/DBP15K/translated_ent_name/dbp_{}.json'.format(dataset)
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    name_dir = {}
    for i in range(len(data)):
        name_dir[data[i][0]] = ' '.join(data[i][1])
    return name_dir


dataset = ['ja_en', 'fr_en', 'zh_en']
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = SentenceTransformer(
    '../Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
model.eval()
for ds in dataset:
    print(ds)
    name_dir = load_name_json(ds)
    ent_1, ent_2 = ent_load(ds)
    dev_pair = dev_pair_load(ds)
    ent_num = len(name_dir)
    index_1, index_2 = load_index(ds)

    new_name_dir = {}
    for k in name_dir.keys():
        if str(k) in index_1.keys():
            new_k = index_1[str(k)]
        elif str(k) in index_2:
            new_k = index_2[str(k)]
        else:
            print('error')
        new_name_dir[int(new_k)] = name_dir[k]
    sorted_dict = dict(sorted(new_name_dir.items()))

    name_lt = list(sorted_dict.values())
    batch_size = 128
    name_embedding = []
    for i in trange(0, ent_num, batch_size):
        key_sents = name_lt[i:i + batch_size]
        name_embedding.append(model.encode(key_sents))
    name_embedding = np.concatenate(name_embedding, axis=0)

    ent1_name_embedding = name_embedding[:len(ent_1)]
    ent2_name_embedding = name_embedding[len(ent_1):]
    scores_all = np.matmul(ent1_name_embedding, ent2_name_embedding.T)
    print(scores_all.shape)

    scores_all = (scores_all - np.min(scores_all)) / (np.max(scores_all) - np.min(scores_all))
    print()
    print(scores_all[0][0])

    scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    # for i in range(len(dev_pair)):
    #     for j in range(len(dev_pair)):
    #         scores[i][j] = scores_all[dev_pair[i][0]]
    dev_pair = np.array(dev_pair)
    for i, l in enumerate(dev_pair[:, 0]):
        for j, r in enumerate(dev_pair[:, 1]):
            scores[i][j] = scores_all[l][r-len(ent_1)]

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # save the scores as .npy file
    np.save(f'../data/DBP15K/DBP_ECS/Name_{ds}.npy', scores)
    sinkhorn_test(scores, scores.shape[0], device=device)




"""

ja_en
dev pair length:10500
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 310/310 [00:39<00:00,  7.93it/s]
(19814, 19780)

0.73146623
(10500, 10500)
                total is  10500
                hits@1 is 0.7599047619047619
                hits@5 is 0.8288571428571428
                hits@10 is 0.8501904761904762
                MR is 151.67971801757812
                MRR is 0.7918210625648499
fr_en
dev pair length:10500
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 310/310 [00:38<00:00,  7.98it/s]
(19661, 19993)

0.60832983
(10500, 10500)
                total is  10500
                hits@1 is 0.8992380952380953
                hits@5 is 0.9356190476190476
                hits@10 is 0.9455238095238095
                MR is 29.358762741088867
                MRR is 0.9162187576293945
zh_en
dev pair length:10500
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 305/305 [00:38<00:00,  7.87it/s]
(19388, 19572)

0.73918676
(10500, 10500)
                total is  10500
                hits@1 is 0.6523809523809524
                hits@5 is 0.7213333333333334
                hits@10 is 0.7458095238095238
                MR is 371.5570373535156
                MRR is 0.6857342720031738

"""



