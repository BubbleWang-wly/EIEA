from pkl_read import *
import numpy as np
from tqdm import tqdm
import torch

dataset = ['ja_en', 'fr_en', 'zh_en']
for ds in dataset:
    pair, len_pair = pair_load(ds)
    pair = np.array(pair)
    ent_1, ent_2 = ent_load(ds)
    dev_pair = dev_pair_load(ds)
    image_source_dir = pkl_load(ds)
    # print(len_pair) pair length:15000

    ent1_id_list = list(ent_1.keys())
    ent2_id_list = list(ent_2.keys())

    print('ent_1_min:', min(ent1_id_list))
    print('ent_1_max:', max(ent1_id_list))
    print('ent_2_min:', min(ent2_id_list))
    print('ent_2_max:', max(ent2_id_list))

    r_index_1, r_index_2 = load_reverse_index(ds)
    len_ent_1 = len(ent_1)
    len_ent_2 = len(ent_2)

    image_scores = -float('inf') * np.ones((len_pair, len_pair))

    for i in tqdm(range(len(ent_1))):
        for j in range(len(ent_2)):
            try:
                image_scores[i, j] = max(image_scores[i, j],
                                         np.dot(image_source_dir[r_index_1[str(ent1_id_list[i])]],
                                                image_source_dir[r_index_2[str(ent2_id_list[j])]]))
                # print(image_scores[i, j])
            except:
                continue
                # print([i, j])

    dev_pair = np.array(dev_pair)
    scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    print(scores.shape)


    for i, l in enumerate(dev_pair[:, 0]):
        for j, r in enumerate(dev_pair[:, 1]):
            scores[i][j] = image_scores[l][r-len_ent_1] if image_scores[l][r-len_ent_1] != -float('inf') else 0
    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    file_name = '../data/DBP15K/DBP_ECS/Vis_{}.npy'.format(ds)
    np.save(file_name, scores)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    sinkhorn_test(scores, scores.shape[0], device=device)



"""
pair length:15000
dev pair length:10500
ent_1_min: 0
ent_1_max: 19813
ent_2_min: 19814
ent_2_max: 39593
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 19814/19814 [11:56<00:00, 27.66it/s]
(10500, 10500)
(10500, 10500)

                total is  10500
                hits@1 is 0.3201904761904762
                hits@5 is 0.3441904761904762
                hits@10 is 0.3554285714285714
                MR is 1889.7198486328125
                MRR is 0.3336591422557831
pair length:15000
dev pair length:10500
ent_1_min: 0
ent_1_max: 19660
ent_2_min: 19661
ent_2_max: 39653
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 19661/19661 [12:35<00:00, 26.03it/s]
(10500, 10500)
(10500, 10500)
                total is  10500
                hits@1 is 0.34723809523809523
                hits@5 is 0.3730476190476191
                hits@10 is 0.3832380952380952
                MR is 1761.74462890625
                MRR is 0.36045461893081665
pair length:15000
dev pair length:10500
ent_1_min: 0
ent_1_max: 19387
ent_2_min: 19388
ent_2_max: 38959
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 19388/19388 [13:35<00:00, 23.78it/s]
(10500, 10500)
(10500, 10500)
                total is  10500
                hits@1 is 0.3464761904761905
                hits@5 is 0.37285714285714283
                hits@10 is 0.38571428571428573
                MR is 1959.4832763671875
                MRR is 0.3604724705219269

    
"""