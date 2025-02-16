# GraDaSE: Graph-Based Dataset Search with Examples

GraDaSE is agraph-based method for Dataset Search with Examples task.

[//]: # (We provide the implementaion of GraDaSE based on the official PyTorch implementation of HGB&#40;https://github.com/THUDM/HGB&#41;)

## Descriptions
The repository is organised as follows:

data/: the original data of two test collections: DataFinder-E and DSEBench.

code/: the code files of the implementation of GraDaSE.

- code/run.py: run reranking of GraDaSE.

- code/model.py: implementation of GraDaSE.

- code/utils/: contains tool functions.

## Data

"data/" directory contains the information of dataset graph and data for two test collections: DataFinder-E and DSEBench.

DataFinder-E and [DSEBench](https://github.com/nju-websoft/DSEBench) are test collections for Dataset Search with Examples, which is the task of reranking a list of candidate datasets based on a keyword query and target datasets. The DataFinder-E is adapted from [Datafinder](https://github.com/viswavi/datafinder). 

### Graph

The "./data/{test collection}/graph/" directory provides the ID and metadata of each dataset, the relationship between datasets and the relationships between datasets and tags in CSV format.

### Queries

The "./data/{test collection}/queries.tsv" provides the keywords queries. The first column is the id of query, and the second column is the text of query.

### Pairs of (q, T)

The "./data/{test collection}/pairs.json" provides the pair ID, keyword query and target datasets in JSON format. 

```
{
  pairID: {"query": query_id, "targets": [dataset_id, ...]}, ...
}
```

### Train, Val and Test
Take the "./data/{test collection}/train.json" file for example. The train.json file contains pair id and candidate datasets list in JSON format.

```
{pair_id: {dataset_id: rel_score, ...}
```

The retrieval results of BM25 are in the "./data/{test collection}/bm25_test.json" file.

## Code

"code/" directory contains the implementation of GraDaSE.

### Requirements
Python==3.10.15

Pytorch==2.4.0

Networkx==3.2.1

numpy==2.0.2

dgl==2.4.0

scipy==1.14.1

### Running experiments
We train our model using NVIDIA GeForce RTX 4090 with CUDA 12.2.

For reranking on DataFinder-E:

```
bash DataFinder.sh
```

For reranking on DSEBench:

```
bash DSEBench.sh
```

