import json
import os
import random
import time
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from tqdm import tqdm
import pytrec_eval
import json
from transformers import AutoTokenizer
from FlagEmbedding import FlagReranker


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class data_loader:
    def __init__(self, path, args):
        self.path = path
        self.bm25 = None
        self.bm25_num = args.retrieve_num
        self.top_k = args.top_k
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.dataset_input_ids = {}
        self.corpus = self.load_corpus('corpus.json')
        self.queries = self.load_keywords('queries.tsv')
        self.pairs = self.load_pairs('pairs.json')
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('train.json')
        self.labels_val = self.load_labels('val.json')
        self.labels_test = self.load_labels('bm25_test.json')

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        new_links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        new_labels_train = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        new_labels_test = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg + cnt))

                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test

                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x: old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][-meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now + [col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0] >= 0 else self.links['meta'][-meta[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0] >= 0 else self.links['meta'][-meta2[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type],
                               self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, test_idx, label, file_name, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label = []
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j] == 1]
                multi_label.append(','.join(label_list))
            label = multi_label
        elif mode == 'bi':
            label = np.array(label)
        else:
            return
        with open(file_name, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def evaluate(self, pred):
        print(
            f"{bcolors.WARNING}Warning: If you want to obtain test score, please submit online on biendata.{bcolors.ENDC}")
        y_true = self.labels_test['data'][self.labels_test['mask']]
        micro = f1_score(y_true, pred, average='micro')
        macro = f1_score(y_true, pred, average='macro')
        result = {
            'micro-f1': micro,
            'macro-f1': macro
        }
        return result

    def evaluate_valid(self, qrels, preds, metrics):
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, metrics)
        eval_results = evaluator.evaluate(preds)
        results = {}
        for metric in metrics:
            results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
        return results

    def load_keywords(self, name):
        queries = {'map': {}, 'word_ids': {}}
        with open(os.path.join(self.path, name), 'r') as f:
            for line in f:
                th = line.split('\t')
                if "Datafinder" in self.path:
                    th[0] = eval(th[0])
                queries['map'][th[0]] = th[1].strip()
                queries['word_ids'][th[0]] = self.process_text(th[1].strip(), 256)
        return queries

    def load_pairs(self, name):
        with open(os.path.join(self.path, name), "r") as f:
            json_data = json.load(f)
        return json_data

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'total': 0, 'data': {}, 'pair_info': {}}
        with open(os.path.join(self.path, name), "r") as f:
            json_data = json.load(f)
        bm25_results, bge_input, query_nodes = [], [], {}
        temp_num = 0
        datatype = name.replace('.json', "")
        if os.path.exists(os.path.join(self.path, f'query_nodes_{self.bm25_num}_{self.top_k}_{datatype}.json')):
            query_nodes = json.load(open(os.path.join(self.path, f'query_nodes_{self.bm25_num}_{self.top_k}_{datatype}.json')))
        else:
            with tqdm(total=len(json_data)) as pbar:
                for k, v in json_data.items():
                    temp_num += 1
                    pair_id = k
                    query_id = self.pairs[pair_id]['query']
                    query_keywords = self.queries['map'][str(query_id)]
                    tokenized_query = query_keywords.lower().split()
                    entity_scores = self.bm25.get_scores(tokenized_query)
                    combined_list = list(zip(entity_scores, [i for i in range(len(self.corpus))]))
                    sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)[:self.bm25_num]
                    new_corpus = []
                    for s, id_ in sorted_combined_list:
                        new_corpus.append(self.corpus[id_])
                    zipped_retrieval_result = list(zip([query_keywords] * len(sorted_combined_list), new_corpus))
                    bm25_results.append(sorted_combined_list)
                    bge_input.extend(zipped_retrieval_result)
                    pbar.update(1)
            scores = self.reranker.compute_score(bge_input, normalize=True)
            temp_num = 0
            for i, k in enumerate(list(json_data.keys())):
                temp_num += 1
                rerank_result = list(zip([j for j in range(len(bm25_results[i]))],
                                         scores[i*len(bm25_results[i]):i*len(bm25_results[i]) + len(bm25_results[i])]))
                sorted_rerank_result = sorted(rerank_result, key=lambda x: x[1], reverse=True)[:self.top_k]
                query_node_ids = []
                for r in sorted_rerank_result:
                    query_node_ids.append(bm25_results[i][r[0]][1])
                query_nodes[k] = query_node_ids
            with open(os.path.join(self.path, f'query_nodes_{self.bm25_num}_{self.top_k}_{datatype}.json'), "w") as f:
                json.dump(query_nodes, f)

        temp_num = 0
        with tqdm(total=len(json_data)) as pbar:
            for k, v in json_data.items():
                temp_num += 1
                pair_id = k
                query_id = self.pairs[pair_id]['query']
                context_ids = self.pairs[pair_id]['targets']
                query_node_ids = query_nodes[k]
                labels['pair_info'][pair_id] = {'query_node_ids': sorted(query_node_ids),
                                                'targets': sorted(context_ids),
                                                'query_ori_id': query_id,
                                                'qc_input_ids': self.process_text(
                                                    self.queries['map'][str(query_id)] + "\n" +
                                                    "\n".join([self.corpus[cid] for cid in context_ids]), 256)}

                if 'test' in name:
                    for dataset_node, rel in v.items():
                        if pair_id not in labels['data'].keys():
                            labels['data'][pair_id] = {}
                        input_ids = self.process_text(self.queries['map'][str(query_id)] + "\n" +
                                                                     "\n".join(
                                                                         [self.corpus[cid] for cid in context_ids]) +
                                                                     "[SEP]" + self.corpus[eval(dataset_node)],
                                                                     512)
                        labels['data'][pair_id][dataset_node] = {"rel": rel, "input_ids": input_ids}

                else:
                    for dataset_node, rel in v.items():
                        if pair_id not in labels['data'].keys():
                            labels['data'][pair_id] = {}
                        if rel > 0:
                            rel = 1
                        else:
                            rel = 0
                        input_ids = self.process_text(self.queries['map'][str(query_id)] + "\n" +
                                                      "\n".join(
                                                          [self.corpus[cid] for cid in context_ids]) +
                                                      "[SEP]" + self.corpus[eval(dataset_node)],
                                                      512)
                        labels['data'][pair_id][dataset_node] = {"rel": rel, "input_ids": input_ids}

                labels['total'] += len(json_data)
                pbar.update(1)
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        node_degree = {}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if h_id not in node_degree.keys():
                    node_degree[h_id] = 0
                if t_id not in node_degree.keys():
                    node_degree[t_id] = 0
                node_degree[h_id] += 1
                node_degree[t_id] += 1
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list), 'edge_type': defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                var = random.randint(0, 9)
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['edge_type'][r_id].append((h_id, t_id))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_corpus(self, name):
        with open(os.path.join(self.path, name), "r") as f:
            json_data = json.load(f)
        corpus, processed_corpus = [], {}
        for k, v in json_data.items():
            corpus.append(v)
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return corpus

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}, 'type': {}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['type'][node_id] = node_type
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                    if node_type == 0:
                        self.dataset_input_ids[node_id] = self.process_text(self.corpus[node_id], 256)
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes

    def process_text(self, text, max_length):
        tokens_dict = self.tokenizer(text, max_length=max_length, padding='max_length', truncation=True,
                                     return_tensors='pt')
        return tokens_dict['input_ids']


