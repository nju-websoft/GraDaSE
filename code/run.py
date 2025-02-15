import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
import scipy.sparse as sp
import dgl
import numpy as np
import torch
import pandas as pd
import networkx as nx
import torch.nn.functional as F
from tqdm import tqdm

from model_new import GraDaSE, myGAT
from utils.data import load_data, batch_data
from utils.pytorchtools import EarlyStopping
from utils.graphtools import k_shortest_paths

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NX_CURGAPH_AUTOCONFIG'] = 'True'
node_successors = {}
sys.path.append('utils/')
MASK_MAX = 1


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def list_to_sp_mat(node_num, li):
    data = [x[2] for x in li]
    i = [x[0] for x in li]
    j = [x[1] for x in li]
    return sp.coo_matrix((data, (i, j)), shape=(node_num, node_num)).tocsr()


def generate_seq(graph, empty_seq, start_ids, seq_len, start_scnt, skip_idx):
    cnt, scnt = 0, 0
    # print('Start: ', start_ids)
    for c in start_ids:
        empty_seq[cnt] = c
        cnt += 1
        if cnt >= seq_len:
            break


def nx_graph(loader):
    nodes = loader.nodes
    all_nodes = range(nodes['total'])
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    all_links = []
    for r in loader.links['edge_type']:
        all_links.extend(loader.links['edge_type'][r])
    G.add_edges_from(all_links)
    return G

metadata = {}
metadata_df = pd.read_csv('../data/Datafinder/graph/datafinder_dataset_metadata.csv', na_filter=False)
for d in metadata_df.itertuples():
    metadata[d.id] = d.title

tags = {}
tags_df = pd.read_csv('../data/Datafinder/graph/datafinder_tags.csv', na_filter=False)
for d in tags_df.itertuples():
    tags[d.id] = d.tag

# pairs = {}
with open('../data/Datafinder/pairs.json', 'r') as f:
    pairs = json.load(f)


def run_model_FAERY(args):
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    feats_type = args.feats_type
    features_list, adjM, train_val_test, dl = load_data(args.dataset, args)
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    print(device)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    train_data = train_val_test['train']
    print(train_data[0])
    print("****************************")
    val_data = train_val_test['val']
    test_data = train_val_test['test']
    # g = dgl.DGLGraph(adjM + adjM.T)
    g = dgl.DGLGraph(adjM)
    # g = dgl.add_self_loop(g)
    g = dgl.remove_self_loop(g)
    print(g.edges()[0].shape)
    nxg = nx_graph(dl)
    d = dict(nx.degree(nxg))
    print("Average Degree为：", sum(d.values()) / len(nxg.nodes))

    seqs = [[], [], []]
    if not os.path.exists(os.path.join(f'../data/{args.dataset}', f'seqs_v1_25.pickle')):
        for i, data_list in enumerate([train_data, val_data, test_data]):
            avg_len_cq, avg_len_cd = 0, 0
            num_cq, num_cd = 0, 0
            with tqdm(total=len(data_list)) as pbar:
                for d in data_list:
                    query_ids = d[1]
                    targets_ids = d[2]
                    dataset_id_pos = d[4]
                    label = d[-1]
                    cq_seqs = []
                    cd_seqs_pos = []
                    qd_seqs = []

                    if dataset_id_pos not in targets_ids:
                        for c in targets_ids:
                            paths = k_shortest_paths(nxg, dataset_id_pos, c, args.num_seqs, )
                            # print(paths)
                            cd_seqs_pos.extend(paths)
                            for path in paths:
                                avg_len_cd += len(path) + 1
                                num_cd += 1
                    cd_seqs_pos.sort(key=len)

                    seqs[i].append([d[0], d[1], d[2], d[3], cq_seqs, [cd_seqs_pos, qd_seqs], label])
                    pbar.update(1)
            print(avg_len_cd/num_cd)
        with open(os.path.join(f'../data/{args.dataset}', f'seqs_v1_25.pickle'), 'wb') as f:
            pickle.dump(seqs, f)
    else:
        seqs = pickle.load(open(os.path.join(f'../data/{args.dataset}', f'seqs_v1_25.pickle'), 'rb'))

    print('Generating Node Sequences...')
    data_seqs = {'train': [], 'val': [], 'test': []}
    for i, data_list in enumerate([train_data, val_data, test_data]):
        print(list(data_seqs.keys())[i])
        all_ct_seqs = torch.zeros((len(data_list), args.num_seqs, args.len_seq)).long()
        all_qt_seqs = torch.zeros((len(data_list), args.len_seq)).long()
        with tqdm(total=len(data_list)) as pbar:
            for data_id, d in enumerate(data_list):
                query_ids = d[1]
                context_ids = d[2]
                dataset_id_pos = d[4]
                ct_seqs = all_ct_seqs[data_id]
                qt_seqs = all_qt_seqs[data_id]
                cd_paths = seqs[i][data_id][5][0]

                start_nodes = [qid for qid in query_ids]
                scnt = len(start_nodes) - 1
                start_nodes.extend(context_ids)
                generate_seq(g, qt_seqs, start_nodes, args.len_seq, 0, range(scnt + 1, len(start_nodes)))

                j = 0
                mask = 0
                if dataset_id_pos in context_ids:
                    mask = -1
                dc_paths = []
                for path in cd_paths:
                    assert path[0] == dataset_id_pos
                    dc_paths.append(path)
                for path in dc_paths:
                    if j >= args.num_seqs:
                        break
                    start_nodes = []
                    start_nodes.extend(path)
                    scnt = len(start_nodes) - 1
                    skip_idx = range(scnt + 1, len(start_nodes))
                    generate_seq(g, ct_seqs[j], start_nodes, args.len_seq, scnt, skip_idx)
                    j += 1
                data_seqs[list(data_seqs.keys())[i]].append(
                    [d[0], d[1], d[2], d[3], qt_seqs, dataset_id_pos, ct_seqs, d[5], mask, d[-1]])
                pbar.update(1)
    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]
    g = g.to(device)

    train_batches = batch_data(data_seqs['train'], args.batch_size, )
    val_batches = batch_data(data_seqs['val'], args.batch_size, )
    test_batches = batch_data(data_seqs['test'], args.batch_size, )
    num_classes = 2
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)
    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k + 1 + len(dl.links['count'])
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    test_results = {}
    for m in metrics:
        test_results[m] = torch.zeros(args.repeat)
    for i in range(args.repeat):
        ranker = GraDaSE(g, 64, len(dl.links['count']) * 2 + 1, num_classes, in_dims, args.hidden_dim,
                            args.num_layers, args.num_gnns,
                            args.num_heads, args.dropout, temper=args.temperature, num_type=len(node_cnt),
                            beta=args.beta,
                            top_k=args.top_k, num_seqs=args.num_seqs)

        ranker.to(device)
        optimizer = torch.optim.Adam(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience // 2,
                                                               verbose=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        criterion.to(device)
        # training loop

        ranker.train()
        early_stopping = EarlyStopping(
            patience=args.patience, verbose=True,
            save_path='checkpoint/GraDaSE_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.dataset,
                                                                                  args.num_layers,
                                                                                  args.num_gnns,
                                                                                  args.len_seq,
                                                                                  args.lr,
                                                                                  args.batch_size,
                                                                                  args.num_seqs, args.top_k,
                                                                                  i))
        early_stop = False
        steps = 0
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            ranker.train()
            for batch in train_batches:
                steps += 1
                scores = ranker(features_list,
                                e_feat,
                                batch['qt_seq'], type_emb, node_type,
                                [batch['ct_seqs'], batch['mask_pos'], batch['dataset_input_ids'].to(device), ],
                                batch['targets']
                                )
                batch['labels'] = torch.tensor(batch['labels'], dtype=torch.float).to(device)
                train_loss = criterion(scores, batch['labels'])
                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                t_end = time.time()
                print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                    epoch, train_loss.item(), t_end - t_start))

                t_start = time.time()
            # validation
            ranker.eval()
            print("*****************Eval*****************")
            qrels, run = {}, {}
            with open(f'../data/{args.dataset}/val.json', 'r') as f:
                qrels = json.load(f)
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_batches:
                    scores = ranker(features_list,
                                    e_feat,
                                    val_batch['qt_seq'], type_emb, node_type,
                                    [val_batch['ct_seqs'], val_batch['mask_pos'],
                                     val_batch['dataset_input_ids'].to(device), ],
                                    val_batch['targets']
                                    )
                    val_batch['labels'] = torch.tensor(val_batch['labels'], dtype=torch.float).to(device)
                    val_loss += criterion(scores, val_batch['labels']).item()
                    for id_ in range(len(val_batch['qt_seq'])):
                        if val_batch['pair_id'][id_] not in run.keys():
                            run[val_batch['pair_id'][id_]] = {}
                        dataset_id_pos = val_batch['dataset'][id_]
                        pred_score = scores[id_].cpu().numpy().tolist()
                        run[val_batch['pair_id'][id_]][str(dataset_id_pos)] = pred_score

                eval_result = dl.evaluate_valid(qrels, run, metrics)
                print(eval_result)

            scheduler.step(val_loss)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss, t_end - t_start))
            # early stopping
            early_stopping(val_loss, eval_result, ranker)
            early_stop = early_stopping.early_stop
            if early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        ranker.load_state_dict(torch.load(
            'checkpoint/GraDaSE_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.num_gnns,
                                                                     args.len_seq, args.lr, args.batch_size,
                                                                     args.num_seqs, args.top_k, i)))
        ranker.eval()
        with torch.no_grad():
            qrels, run, bm25_result = {}, {}, {}
            with open(f'../data/{args.dataset}/test.json', 'r') as f:
                qrels = json.load(f)
            with open(f'../data/{args.dataset}/bm25_test.json', 'r') as f:
                bm25_result = json.load(f)
            for batch in test_batches:
                scores = ranker(features_list,
                                e_feat,
                                batch['qt_seq'], type_emb, node_type,
                                [batch['ct_seqs'], batch['mask_pos'], batch['dataset_input_ids'].to(device), ],
                                batch['targets']
                                )

                for id_ in range(len(batch['qt_seq'])):
                    if batch['pair_id'][id_] not in run.keys():
                        run[batch['pair_id'][id_]] = {}
                    dataset_id = batch['dataset'][id_]
                    pred_score = scores[id_].cpu().numpy().tolist()
                    run[batch['pair_id'][id_]][str(dataset_id)] = pred_score
            result = dl.evaluate_valid(qrels, run, metrics)

            print(f"Repeat: {i}")
            for metric in metrics:
                test_results[metric][i] = result[metric]
                print(f'{metric}: {result[metric]:.4f}', end='\t')
            print('\n')

            with open(
                    'result/result_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.dataset, args.num_layers, args.num_gnns,
                                                                        args.len_seq, args.lr, args.batch_size,
                                                                        args.num_seqs, args.top_k, i), 'w') as f:
                json.dump(result, f)
            with open('result/run_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.dataset, args.num_layers, args.num_gnns,
                                                                       args.len_seq, args.lr, args.batch_size,
                                                                       args.num_seqs, args.top_k, i), 'w') as f:
                json.dump(run, f)
    for metric in metrics:
        print(f"{metric}: {test_results[metric].mean().item():.4f}, std: {test_results[metric].std().item():.4f}", )


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='GraDaSE')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2' +
                         '4 - only term features (id vec for others);' +
                         '5 - only term features (zero vec for others).')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--mode', type=str, default="qc")
    ap.add_argument('--hidden-dim', type=int, default=256,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--dataset', type=str, default='DBLP', help='DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 2.')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=20, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
    ap.add_argument('--num-gnns', type=int, default=4,
                    help='The number of layers of both structural and heterogeneous encoder')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--scope', type=str, default="origin")
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--retrieve-num', type=int, default=100)
    ap.add_argument('--eval-steps', type=int, default=500)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=20, help='The length of node sequence.')
    ap.add_argument('--num-seqs', type=int, default=15, help='The length of node sequence.')
    ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
    ap.add_argument('--test-only', type=bool, default=False, help='only test')
    ap.add_argument('--temperature', type=float, default=1.0, help='Temperature of attention score')
    ap.add_argument('--beta', type=float, default=1.0, help='Weight of heterogeneity-level attention score')

    args = ap.parse_args()
    print(args)
    run_model_FAERY(args)
