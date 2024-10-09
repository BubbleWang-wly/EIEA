import numpy as np
from pkl_read import ent_id_load, dev_pair_load, sinkhorn_test
import torch

def load_attr(fns, e, ent2id, topA=1000, kg_min=0):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]-kg_min][attr2id[th[i]]] = 1.0
    return attr


dataset = ['ja_en', 'fr_en', 'zh_en']
for d in dataset:
    print(d)
    # att_1, att_2 = load_att(d)
    dev_pair = dev_pair_load(d)
    ent_id_1, ent_id_2 = ent_id_load(d)
    fns_1 = ['../data/DBP15K/{}/training_attrs_1'.format(d)]
    att_m_1 = load_attr(fns_1, len(ent_id_1), ent_id_1, kg_min=0)
    fns_2 = ['../data/DBP15K/{}/training_attrs_2'.format(d)]
    att_m_2 = load_attr(fns_2, len(ent_id_2), ent_id_2, kg_min=len(ent_id_1))

    print(att_m_1.shape)
    print(att_m_2.shape)

    scores_all = np.matmul(att_m_1, att_m_2.T)
    print(scores_all.shape)

    scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    dev_pair = np.array(dev_pair)
    for i, l in enumerate(dev_pair[:, 0]):
        for j, r in enumerate(dev_pair[:, 1]):
            scores[i][j] = scores_all[l][r-len(ent_id_1)]

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # save the scores as .npy file
    np.save(f'../data/DBP15K/DBP_ECS/Att_{d}_2.npy', scores)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sinkhorn_test(scores, scores.shape[0], device=device)


"""
(EIEA-env) [root@QC-AI DBP_ECS]# python att_CS_2.py                                                                                           
ja_en
dev pair length:10500
(19814, 1000)
(19780, 1000)
(19814, 19780)
(10500, 10500)
                total is  10500
                hits@1 is 0.0006666666666666666
                hits@5 is 0.0021904761904761906
                hits@10 is 0.0044761904761904765
                MR is 3188.80224609375
                MRR is 0.0033399604726582766
fr_en
dev pair length:10500
(19661, 1000)
(19993, 1000)
(19661, 19993)
(10500, 10500)
                total is  10500
                hits@1 is 0.00038095238095238096
                hits@5 is 0.0017142857142857142
                hits@10 is 0.0026666666666666666
                MR is 3228.09912109375
                MRR is 0.002442853758111596
zh_en
dev pair length:10500
(19388, 1000)
(19572, 1000)
(19388, 19572)
(10500, 10500)
                total is  10500
                hits@1 is 0.0005714285714285715
                hits@5 is 0.0026666666666666666
                hits@10 is 0.004857142857142857
                MR is 3229.731689453125
                MRR is 0.002890021540224552

"""



