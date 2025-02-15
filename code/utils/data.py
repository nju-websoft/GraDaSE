import json
import os.path
import pickle
import sys
import random
import torch
import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp


def load_data(prefix='FAERY', args=None):
    from data_loader import data_loader
    dl = data_loader('../data/' + prefix, args)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    train_data = []
    test_data = []
    val_data = []
    random.seed(2025)
    train_keys = list(dl.labels_train['data'].keys())
    random.shuffle(train_keys)
    for pair_id in train_keys:
        candidate_lists = dl.labels_train['data'][pair_id]
        candidate_keys = list(candidate_lists.keys())
        random.shuffle(candidate_keys)
        for k in candidate_keys:
            v = candidate_lists[k]
            label = v['rel']
            input_ids = v['input_ids']
            train_data.append([pair_id,
                               dl.labels_train['pair_info'][pair_id]['query_node_ids'],
                               dl.labels_train['pair_info'][pair_id]['targets'],
                               input_ids,
                               eval(k),
                               dl.dataset_input_ids[eval(k)],
                               label])
    for pair_id, candidate_lists in dl.labels_val['data'].items():
        for k, v in candidate_lists.items():
            label = v['rel']
            input_ids = v['input_ids']
            val_data.append([pair_id,
                             dl.labels_val['pair_info'][pair_id]['query_node_ids'],
                             dl.labels_val['pair_info'][pair_id]['targets'],
                             input_ids,
                             eval(k),
                             dl.dataset_input_ids[eval(k)],
                             label])
    for pair_id, candidate_lists in dl.labels_test['data'].items():
        for k, v in candidate_lists.items():

            label = v['rel']
            input_ids = v['input_ids']
            test_data.append([pair_id,
                              dl.labels_test['pair_info'][pair_id]['query_node_ids'],
                              dl.labels_test['pair_info'][pair_id]['targets'],
                              input_ids,
                              eval(k),
                              dl.dataset_input_ids[eval(k)],
                              label])
    train_val_test = {'train': train_data, 'val': val_data, 'test': test_data}
    return features, \
        adjM, \
        train_val_test, \
        dl


def batch_data(data, batch_size):
    batches = []
    batch = {'pair_id': [],
             'query': [], 'targets': [], 'dataset': [],
             'qt_seq': [], 'ct_seqs': [],
             'query_input_ids': [], 'dataset_input_ids': [], 'mask_pos': [],
             'labels': []}

    for i in range(len(data)):
        batch['pair_id'].append(data[i][0])
        batch['query'].append(data[i][1])
        batch['targets'].append(data[i][2])
        batch['query_input_ids'].append(data[i][3])
        batch['qt_seq'].append(data[i][4])
        batch['dataset'].append(data[i][5])
        batch['ct_seqs'].append(data[i][6])
        batch['dataset_input_ids'].append(data[i][7])
        batch['mask_pos'].append(data[i][8])
        batch['labels'].append(data[i][9])
        if (i + 1) % batch_size == 0:
            batch['query_input_ids'] = torch.cat(batch['query_input_ids'], 0)
            batch['dataset_input_ids'] = torch.cat(batch['dataset_input_ids'], 0)
            batch['qt_seq'] = torch.stack(batch['qt_seq'])
            batch['ct_seqs'] = torch.stack(batch['ct_seqs'])
            batches.append(batch)
            batch = {'pair_id': [],
                     'query': [], 'targets': [], 'dataset': [],
                     'qt_seq': [], 'ct_seqs': [],
                     'query_input_ids': [], 'dataset_input_ids': [], 'mask_pos': [],
                     'labels': []}

    if len(batches) * batch_size < len(data):
        batch['query_input_ids'] = torch.cat(batch['query_input_ids'], 0)
        batch['dataset_input_ids'] = torch.cat(batch['dataset_input_ids'], 0)
        batch['qt_seq'] = torch.stack(batch['qt_seq'])
        batch['ct_seqs'] = torch.stack(batch['ct_seqs'])
        batches.append(batch)
    return batches
