# from pkl_read import *
import random


def pair_load(dataset):
    pair = []
    with open('../data/DBP15K/DBP_1/{}/ill_ent_ids'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id1, ent_id2 = line.split('\t')
            pair.append((int(ent_id1), int(ent_id2)))

    print('pair length:{}'.format(len(pair)))
    return pair, len(pair)


dataset = ['ja_en', 'fr_en', 'zh_en']
for ds in dataset:
    pair, len_pair = pair_load(ds)
    random.shuffle(pair)
    split_index = int(len(pair) * 0.3)
    train_data = pair[:split_index]
    test_data = pair[split_index:]

    with open('../data/DBP15K/DBP_1/{}/sup_ent_ids'.format(ds), 'w') as train_file:
        for ent_id1, ent_id2 in train_data:
            train_file.write(f"{ent_id1}\t{ent_id2}\n")

    with open('../data/DBP15K/DBP_1/{}/dev_ent_ids'.format(ds), 'w') as test_file:
        for ent_id1, ent_id2 in test_data:
            test_file.write(f"{ent_id1}\t{ent_id2}\n")
