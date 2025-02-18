# Test Collections

The data of DataFinder-E and DSEBench can be downloaded from [Zenodo](https://zenodo.org/records/14876878).

```
wget https://zenodo.org/records/14876878/files/DataFinder-E.zip
wget https://zenodo.org/records/14876878/files/DSEBench.zip
```

The structure of the "data/"directory is as follows:
```
data
├── DataFinder
│   ├── graph
│   │   ├── datafinder_dataset_dataset_rel.csv
│   │   ├── datafinder_dataset_metadata.csv
│   │   ├── datafinder_dataset_tag_rel.csv
│   │   ├── datafinder_keywords.csv
│   │   └── datafinder_tags.csv
│   ├── BM25_results.json
│   ├── bm25_test.json
│   ├── corpus.json
│   ├── link.dat
│   ├── node.dat
│   ├── pairs.json
│   ├── queries.tsv
│   ├── test.json
│   ├── train.json
│   └── val.json
└── DSEBench
│   ├──...

```

## Graph
{test collection}_dataset_metadata.csv contains the metadata of datasets.

{test collection}_dataset_dataset_rel.csv contains the relationships between datasets.

```
"id1","id2","relationship"
...
"DATASET_00001115","DATASET_00002228","Replica"
...
```

{test collection}_dataset_tag_rel.csv contains the relationships between datasets and tags.

```
"id1","id2","relationship"
...
"DATASET_00000000","TAG_00000000","HAS"
...
```

{test collection}_keywords.csv and {test collection}_tags.csv contain the queries and tags of the test collection.

link.dat and node.dat files are the basis of dataset graph construction by dgl. 
For every line in node.dat, node_id, dataset_id, node_type and node_attribute are split by \t. 
For every line in link.dat, node_id1, node_id2, edge_type, weight are split by \t. 

## Corpus
corpus.json is the corpus for BM25 in Target-Biased Query Representation.
```
{
  node_id: node_info,
  ...
}
```

## Queries

The "{test collection}/queries.tsv" provides the keywords queries. The first column is the id of query, and the second column is the text of query.

## Pairs of (q, T)

The "{test collection}/pairs.json" provides the pair ID, keyword query and target datasets in JSON format. 

```
{
  pairID: {"query": query_id, "targets": [dataset_id, ...]}, ...
}
```

## Train, Val and Test
Take the "{test collection}/train.json" file for example. The train.json file contains pair id and candidate datasets list in JSON format.

```
{pair_id: {dataset_id: rel_score, ...}
```

The retrieval results of BM25 are in the "{test collection}/bm25_test.json" file.
