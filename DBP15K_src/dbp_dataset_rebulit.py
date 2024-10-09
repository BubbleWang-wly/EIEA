

def ent_load(dataset):
    index_1 = {}
    index_2 = {}
    ent_1 = {}
    with open('../data/DBP15K/{}/ent_ids_1'.format(dataset), 'r') as f:
        i = 0
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            index_1[ent_id] = i
            ent_1[i] = ent_name
            i += 1

    print('i', i)

    ent_2 = {}
    with open('../data/DBP15K/{}/ent_ids_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            index_2[ent_id] = i
            ent_2[i] = ent_name
            i += 1
    # save ent_1
    with open('../data/DBP15K/DBP_1/{}/ent_ids_1'.format(dataset), 'w') as f:
        for k, v in ent_1.items():
            f.write('{}\t{}\n'.format(k, v))
    # save ent_2
    with open('../data/DBP15K/DBP_1/{}/ent_ids_2'.format(dataset), 'w') as f:
        for k, v in ent_2.items():
            f.write('{}\t{}\n'.format(k, v))
    # save index_1
    with open('../data/DBP15K/DBP_1/{}/index_1'.format(dataset), 'w') as f:
        for k, v in index_1.items():
            f.write('{}\t{}\n'.format(k, v))
    # save index_2
    with open('../data/DBP15K/DBP_1/{}/index_2'.format(dataset), 'w') as f:
        for k, v in index_2.items():
            f.write('{}\t{}\n'.format(k, v))


def triple_load_save_new_triple(dataset):
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

    # load_triple_1
    triple_1 = []
    with open('../data/DBP15K/{}/triples_1'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            h, r, t = line.split('\t')
            triple_1.append([index_1[h], r, index_1[t]])
    # load_triple_2
    triple_2 = []
    with open('../data/DBP15K/{}/triples_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            h, r, t = line.split('\t')
            triple_2.append([index_2[h], r, index_2[t]])

    # save new_triple
    with open('../data/DBP15K/DBP_1/{}/triples_1'.format(dataset), 'w') as f:
        for i in range(len(triple_1)):
            f.write('{}\t{}\t{}\n'.format(triple_1[i][0], triple_1[i][1], triple_1[i][2]))

    with open('../data/DBP15K/DBP_1/{}/triples_2'.format(dataset), 'w') as f:
        for i in range(len(triple_2)):
            f.write('{}\t{}\t{}\n'.format(triple_2[i][0], triple_2[i][1], triple_2[i][2]))

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


def load_ill_pair_save_new(dataset):
    index_1, index_2 = load_index(dataset)

    ill_pairs = []
    with open('../data/DBP15K/{}/ill_ent_ids'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            h, t = line.split('\t')
            ill_pairs.append([index_1[h], index_2[t]])
    with open('../data/DBP15K/DBP_1/{}/ill_ent_ids'.format(dataset), 'w') as f:
        for i in range(len(ill_pairs)):
            f.write('{}\t{}\n'.format(ill_pairs[i][0], ill_pairs[i][1]))


dataset = ['ja_en', 'fr_en', 'zh_en']
for d in dataset:
    print(d)
    load_ill_pair_save_new(d)

    # triple_load_save_new_triple(d)
#     ent_load(d)
