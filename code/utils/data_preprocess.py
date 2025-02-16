import os
import json
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

metadata_corpus = []
tags_corpus = []
node_id_map = {}
ntcir_id_map = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer("../../../datasets/model/stella_en_400M_v5", trust_remote_code=True).cuda()


def stella_embedding(texts):
    text_embeds = model.encode(texts)
    return text_embeds


def generate_node_dat(ori_datasets_file, ori_tags_file, out_node_file, out_corpus_file):
    with open(ori_datasets_file, 'r') as f:
        datasets_df = pd.read_csv(f, na_filter=False)
    with open(ori_tags_file, 'r') as f:
        tags_df = pd.read_csv(f, na_filter=False)
    for r in datasets_df.itertuples():
        metadata_corpus.append("\n".join([r.title, r.description]))
    for r in tags_df.itertuples():
        tags_corpus.append(r.tag)
    dataset_embeds = stella_embedding(metadata_corpus)
    print(dataset_embeds.shape)
    tags_embeds = stella_embedding(tags_corpus)
    print(tags_embeds.shape)
    node_id = 0
    all_nodes = []
    corpus = {}
    for r in datasets_df.itertuples():
        record = [node_id, r[1], 0, dataset_embeds[r[0]]]  # 0: Dataset, 1: Tag
        all_nodes.append(record)
        node_id_map[r[1]] = node_id
        ntcir_id_map[r[2]] = r[1]
        corpus[node_id] = r.title + '\n' + r.description
        node_id += 1
    for r in tags_df.itertuples():
        record = [node_id, r[1], 1, tags_embeds[0]]
        all_nodes.append(record)
        node_id_map[r[1]] = node_id
        corpus[node_id] = r.tag
        node_id += 1
    with open(out_node_file, 'w') as f:
        for node in all_nodes:
            f.write(str(node[0]))
            f.write('\t')
            f.write(node[1])
            f.write('\t')
            f.write(str(node[2]))
            f.write('\t')
            for i, attr in enumerate(node[3]):
                f.write(str(attr))
                if i != len(node[3]) - 1:
                    f.write(',')
            f.write('\n')
        # f.writelines(all_nodes)
    with open(out_corpus_file, 'w') as f:
        json.dump(corpus, f)


def generate_link_dat(dataset_dataset_rel_file, dataset_tags_rel_file, outfile):
    link_types = {
        "0": {"start": "0", "end": "0", "meaning": "dataset is Replica to dataset"},
        "1": {"start": "0", "end": "0", "meaning": "dataset is Variant to dataset"},
        "2": {"start": "0", "end": "0", "meaning": "dataset is Version to dataset"},
        "3": {"start": "0", "end": "0", "meaning": "dataset is Subset to dataset"},
        "4": {"start": "0", "end": "0", "meaning": "dataset is SuperSet to dataset"},
        "5": {"start": "0", "end": "0", "meaning": "dataset is Derived from dataset"},
        "6": {"start": "0", "end": "1", "meaning": "dataset Has tag"},
        "7": {"start": "1", "end": "0", "meaning": "tag is Attached to dataset"}
    }
    dd_rel_df = pd.read_csv(dataset_dataset_rel_file, na_filter=False)
    dt_rel_df = pd.read_csv(dataset_tags_rel_file, na_filter=False)
    ori_edge_type = ['Replica', 'Variant', 'Subset', 'Derived', 'Version']
    links = []
    for dd in dd_rel_df.itertuples():
        ori_id1 = dd[1]
        ori_id2 = dd[2]
        ori_rel = dd[3]
        new_id1 = node_id_map[ori_id1]
        new_id2 = node_id_map[ori_id2]
        if ori_rel == 'Replica':
            links.append([new_id1, new_id2, 0, 1.0])
            links.append([new_id2, new_id1, 0, 1.0])
        elif ori_rel == 'Variant':
            links.append([new_id1, new_id2, 1, 1.0])
            links.append([new_id2, new_id1, 1, 1.0])
        elif ori_rel == 'Version':
            links.append([new_id1, new_id2, 2, 1.0])
            links.append([new_id2, new_id1, 2, 1.0])
        elif ori_rel == 'Subset':
            links.append([new_id1, new_id2, 3, 1.0])
            links.append([new_id2, new_id1, 4, 1.0])
        elif ori_rel == 'Derived':
            links.append([new_id1, new_id2, 5, 1.0])
        else:
            assert False
    for dt in dt_rel_df.itertuples():
        ori_id1 = dt[1]
        ori_id2 = dt[2]
        new_id1 = node_id_map[ori_id1]
        new_id2 = node_id_map[ori_id2]
        links.append([new_id1, new_id2, 6, 1.0])
        links.append([new_id2, new_id1, 7, 1.0])
    links.sort(key=lambda x: x[2], reverse=False)
    with open(outfile, 'w') as f:
        for link in links:
            f.write(str(link[0]))
            f.write('\t')
            f.write(str(link[1]))
            f.write('\t')
            f.write(str(link[2]))
            f.write('\t')
            f.write(str(link[3]))
            f.write('\n')


def generate_query_dat(query_file, outfile):
    query_df = pd.read_csv(query_file, na_filter=False)
    with open(outfile, 'w') as f:
        for query in query_df.itertuples():
            f.write(str(query[1]))
            f.write('\t')
            f.write(query[2])
            f.write('\n')


def generate_FAERY_data_dat(ori_train_file, ori_val_file, ori_test_file, pairs_dat):
    with open(ori_train_file, 'r') as f:
        ori_train_list = json.load(f)
    with open(ori_val_file, 'r') as f:
        ori_val_list = json.load(f)
    with open(ori_test_file, 'r') as f:
        ori_test_list = json.load(f)
    # print(ntcir_id_map)
    train_data, val_data, test_data, pairs = {}, {}, {}, {}
    for i, datalist in enumerate([ori_train_list, ori_val_list, ori_test_list]):
        for record in datalist:
            pair_id = record['qdpair_id']
            query_id = record['query_id']
            context_nodes = [node_id_map[ntcir_id_map[record['target_dataset_id']]]]
            dataset_node = node_id_map[ntcir_id_map[record['candidate_dataset_id']]]
            score = record['drel'] * record['qrel']
            if score > 0:
                label = 1
            else:
                label = 0
            pairs[pair_id] = {'query': query_id,
                              'contexts': context_nodes}
            if i == 0:
                if pair_id not in train_data.keys():
                    train_data[pair_id] = {}
                train_data[pair_id][dataset_node] = label
                # train_data.append([query_id, ','.join(context_nodes), dataset_node, label])
            elif i == 1:
                if pair_id not in val_data.keys():
                    val_data[pair_id] = {}
                val_data[pair_id][dataset_node] = label
            elif i == 2:
                if pair_id not in test_data.keys():
                    test_data[pair_id] = {}
                test_data[pair_id][dataset_node] = label
                # test_data.append([query_id, ','.join(context_nodes), dataset_node, label])

    raw_data = {'train': {}, 'val': {}, 'test': {}}
    for pair_id, candidate_lists in train_data.items():
        raw_data['train'][pair_id] = candidate_lists
    for pair_id, candidate_lists in val_data.items():
        raw_data['val'][pair_id] = candidate_lists
    for pair_id, candidate_lists in test_data.items():
        raw_data['test'][pair_id] = candidate_lists
    with open(os.path.join('../../data/' + 'DSEBench', 'train.json'), 'w') as f:
        json.dump(raw_data['train'], f)
    with open(os.path.join('../../data/' + 'DSEBench', 'val.json'), 'w') as f:
        json.dump(raw_data['val'], f)
    with open(os.path.join('../../data/' + 'DSEBench', 'test.json'), 'w') as f:
        json.dump(raw_data['test'], f)
    with open(pairs_dat, 'w') as f:
        json.dump(pairs, f)


def generate_data_dat(ori_train_file, ori_test_file, train_outfile, test_outfile, pairs_dat, prefix):
    ori_train_df = pd.read_csv(ori_train_file, na_filter=False)
    ori_test_df = pd.read_csv(ori_test_file, na_filter=False)
    train_data, test_data, pairs = {}, {}, {}
    for i, df in enumerate([ori_train_df, ori_test_df]):
        for record in df.itertuples():
            pair_id = record[4]

            query_id = record[2]
            context_nodes = []
            for cid in record[3].split(','):
                context_nodes.append(node_id_map[cid])
            dataset_node = node_id_map[record[5]]
            if record[6] > 0:
                label = 1
            else:
                label = 0
            pairs[pair_id] = {'query': query_id,
                              'contexts': context_nodes}
            if i == 0:
                if pair_id not in train_data.keys():
                    train_data[pair_id] = {}
                train_data[pair_id][dataset_node] = label
                # train_data.append([query_id, ','.join(context_nodes), dataset_node, label])
            else:
                if pair_id not in test_data.keys():
                    test_data[pair_id] = {}
                test_data[pair_id][dataset_node] = label
                # test_data.append([query_id, ','.join(context_nodes), dataset_node, label])
    random.seed(2025)
    train_keys = list(train_data.keys())
    val_ratio = 0.2
    random.shuffle(train_keys)
    val_keys = train_keys[:int(val_ratio * len(train_keys))]
    raw_data = {'train': {}, 'val': {}, 'test': {}}
    for pair_id, candidate_lists in train_data.items():

        if pair_id in val_keys:
            raw_data['val'][pair_id] = candidate_lists
        else:
            raw_data['train'][pair_id] = candidate_lists
    for pair_id, candidate_lists in test_data.items():
        raw_data['test'][pair_id] = candidate_lists
    with open(os.path.join('../../data/' + prefix, 'train.json'), 'w') as f:
        json.dump(raw_data['train'], f)
    with open(os.path.join('../../data/' + prefix, 'val.json'), 'w') as f:
        json.dump(raw_data['val'], f)
    with open(os.path.join('../../data/' + prefix, 'test.json'), 'w') as f:
        json.dump(raw_data['test'], f)
    with open(train_outfile, 'w') as f1, open(test_outfile, 'w') as f2:
        files = [f1, f2]
        for i, data_list in enumerate([train_data, test_data]):
            json.dump(data_list, files[i])
    with open(pairs_dat, 'w') as f:
        json.dump(pairs, f)


def get_bm25_test(bm25_file, outfile):
    with open(bm25_file, 'r') as f:
        bm25_data = json.load(f)
    bm25_test = {}
    for pair_id, candidate_lists in bm25_data.items():
        node_candidate = {}
        for c in candidate_lists:
            if c == 'DATASET_00000000':
                node_candidate["0"] = candidate_lists[c]
            else:
                node_candidate[str(eval(c.lstrip('DATASET_0')))] = candidate_lists[c]

        # for candidate in candidate_lists:
        bm25_test[pair_id] = node_candidate
    with open(outfile, 'w') as f:
        json.dump(bm25_test, f)


def load_metadata(ori_datasets_file):
    id_map = {}
    with open(ori_datasets_file, 'r') as f:
        metadata = pd.read_csv(f)
    for record in metadata.itertuples():
        id_map[record[2]] = record[1]
    return id_map


def process_faery_bm25_result(bm25_result_file, metadata_file, outfile):
    with open(bm25_result_file, 'r') as f:
        bm25_data = json.load(f)
    new_bm25_data, BM25_node = {}, {}
    id_map = load_metadata(metadata_file)
    for pair_id, candidate_lists in bm25_data.items():
        new_candidate_lists = []
        node_candidate = {}
        for c in candidate_lists:
            nc = id_map[c]
            if nc == 'DATASET_00000000':
                new_candidate_lists.append("0")
                node_candidate["0"] = candidate_lists[c]
            else:
                new_candidate_lists.append(str(eval(nc.lstrip('DATASET_0'))))
                node_candidate[str(eval(nc.lstrip('DATASET_0')))] = candidate_lists[c]
        # for candidate in candidate_lists:
        # new_bm25_data[pair_id] = new_candidate_lists
        new_bm25_data[pair_id] = node_candidate
    with open(outfile, 'w') as f:
        json.dump(new_bm25_data, f)


if __name__ == '__main__':
    generate_node_dat('../../data/DSEBench/graph/faery_dataset_metadata.csv',
                      '../../data/DSEBench/graph/faery_tags.csv',
                      '../../data/DSEBench/node.dat',
                      '../../data/DSEBench/corpus.json')
    generate_link_dat('../../data/DSEBench/graph/faery_dataset_dataset_rel.csv',
                      '../../data/DSEBench/graph/faery_dataset_tag_rel.csv',
                      '../../data/DSEBench/link.dat')

