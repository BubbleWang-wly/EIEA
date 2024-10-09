import torch
from sentence_transformers import SentenceTransformer
from pkl_read import ent_load, dev_pair_load, sinkhorn_test
import numpy as np
from tqdm import trange

def load_att(dataset):
    d_l = dataset.split('_')[0]
    ent1_att_path = '../data/DBP15K/{}/training_attrs_1'.format(dataset)
    ent2_att_path = '../data/DBP15K/{}/training_attrs_2'.format(dataset)
    att_1 ={}
    att_2 ={}

    with open(ent1_att_path, 'r') as f:
        for line in f:
            line = line.strip()
            l = line.split('\t')
            l_p = 'http://' + str(d_l) +'.dbpedia.org/property/'
            new_l = [item.replace("http://dbpedia.org/property/", "").replace(l_p, "") for item in l[1:]]
            att_1[l[0]] = '||'.join(new_l)
    with open(ent2_att_path, 'r') as f:
        for line in f:
            line = line.strip()
            l = line.split('\t')
            # new_l = [item.replace("http://ja.dbpedia.org/property/", "") for item in l[1:]]
            l_p = 'http://' + str(d_l) +'.dbpedia.org/property/'
            new_l = [item.replace("http://dbpedia.org/property/", "").replace(l_p, "") for item in l[1:]]
            att_2[l[0]] = '||'.join(new_l)
    return att_1, att_2


def id_att(ent_1, ent_2, att_1, att_2):
    id_att_1 = {}
    id_att_2 = {}
    for id in ent_1.keys():
        if ent_1[id] in att_1.keys():
            att = att_1[ent_1[id]]
            id_att_1[id] = att
        else:
            id_att_1[id] = 'No_property'
    for id in ent_2.keys():
        if ent_2[id] in att_2.keys():
            att = att_2[ent_2[id]]
            id_att_2[id] = att
        else:
            id_att_2[id] = 'No_property'
    return id_att_1, id_att_2


dataset = ['ja_en', 'fr_en', 'zh_en']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = SentenceTransformer(
    # '../Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
model = SentenceTransformer(
    '../LaBSE/').to(device)
model.eval()
for d in dataset:
    print(d)
    att_1, att_2 = load_att(d)
    ent_1, ent_2 = ent_load(d)
    dev_pair = dev_pair_load(d)
    ent_1_num = len(ent_1)
    ent_2_num = len(ent_2)
    id_att_1, id_att_2 = id_att(ent_1, ent_2, att_1, att_2)

    att_lt_1 = list(id_att_1.values())
    att_lt_2 = list(id_att_2.values())
    batch_size = 512
    att_embedding_1 = []
    att_embedding_2 = []

    for i in trange(0, ent_1_num, batch_size):
        key_sents = att_lt_1[i:i + batch_size]
        att_embedding_1.append(model.encode(key_sents))
        # print(att_embedding_1[-1].shape)
    att_embedding_1 = np.concatenate(att_embedding_1, axis=0)

    for i in trange(0, ent_2_num, batch_size):
        key_sents = att_lt_2[i:i + batch_size]
        att_embedding_2.append(model.encode(key_sents))
    att_embedding_2 = np.concatenate(att_embedding_2, axis=0)

    print(att_embedding_1.shape)
    print(att_embedding_2.shape)

    scores_all = np.matmul(att_embedding_1, att_embedding_2.T)
    print(scores_all.shape)
    scores_all = (scores_all - np.min(scores_all)) / (np.max(scores_all) - np.min(scores_all))
    print(scores_all[0][0])

    scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    dev_pair = np.array(dev_pair)
    for i, l in enumerate(dev_pair[:, 0]):
        for j, r in enumerate(dev_pair[:, 1]):
            scores[i][j] = scores_all[l][r-len(ent_1)]

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # save the scores as .npy file
    # np.save(f'../data/DBP15K/DBP_ECS/Att_{d}_1.npy', scores) # Robert
    np.save(f'../data/DBP15K/DBP_ECS/Att_{d}_3.npy', scores) # LaBSE
    sinkhorn_test(scores, scores.shape[0], device=device)



"""
1.
    ja_en
    dev pair length:10500
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [06:52<00:00, 10.57s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:59<00:00,  9.22s/it]
    (19814, 1024)
    (19780, 1024)
    (19814, 19780)
    (10500, 10500)
                    total is  10500
                    hits@1 is 0.019714285714285715
                    hits@5 is 0.05104761904761905
                    hits@10 is 0.076
                    MR is 2110.478759765625
                    MRR is 0.040488358587026596
    fr_en
    dev pair length:10500
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:45<00:00,  8.86s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [06:35<00:00,  9.89s/it]
    (19661, 1024)
    (19993, 1024)
    (19661, 19993)
    (10500, 10500)
                    total is  10500
                    hits@1 is 0.0015238095238095239
                    hits@5 is 0.007523809523809524
                    hits@10 is 0.014095238095238095
                    MR is 2877.2216796875
                    MRR is 0.007567377761006355
    zh_en
    dev pair length:10500
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [07:14<00:00, 11.44s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [06:10<00:00,  9.49s/it]
    (19388, 1024)
    (19572, 1024)
    (19388, 19572)
    (10500, 10500)
                    total is  10500
                    hits@1 is 0.06495238095238096
                    hits@5 is 0.14466666666666667
                    hits@10 is 0.19247619047619047
                    MR is 1268.18505859375
                    MRR is 0.10811395198106766


2. LaBSE
    ja_en
        dev pair length:10500
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:41<00:00,  1.07s/it]
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:39<00:00,  1.00s/it]
        (19814, 768)
        (19780, 768)
        (19814, 19780)
        0.5599729
        (10500, 10500)
                        total is  10500
                        hits@1 is 0.04780952380952381
                        hits@5 is 0.11552380952380953
                        hits@10 is 0.16352380952380952
                        MR is 547.0552368164062
                        MRR is 0.0889933854341507
    fr_en
        dev pair length:10500
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:33<00:00,  1.16it/s]
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:53<00:00,  1.35s/it]
        (19661, 768)
        (19993, 768)
        (19661, 19993)
        0.79370373
        (10500, 10500)
                        total is  10500
                        hits@1 is 0.011333333333333334
                        hits@5 is 0.04057142857142857
                        hits@10 is 0.06676190476190476
                        MR is 1092.2547607421875
                        MRR is 0.03248130530118942
    zh_en
        dev pair length:10500
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:41<00:00,  1.10s/it]
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:38<00:00,  1.02it/s]
        (19388, 768)
        (19572, 768)
        (19388, 19572)
        0.738148
        (10500, 10500)
                        total is  10500
                        hits@1 is 0.1180952380952381
                        hits@5 is 0.2321904761904762
                        hits@10 is 0.2896190476190476
                        MR is 330.5966796875
                        MRR is 0.1775858998298645

"""