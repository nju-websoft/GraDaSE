# GraDaSE: Graph-Based Dataset Search with Examples

[//]: # (We provide the implementaion of GraDaSE based on the official PyTorch implementation of HGB&#40;https://github.com/THUDM/HGB&#41;)

## 1. Descriptions
The repository is organised as follows:

collections/: the original data of two test collections: DataFinder-E and DSEBench.

code/: the code files of the implementation of GraDaSE.

- code/run.py: run reranking of GraDaSE.

- code/model.py: implementation of GraDaSE.

- code/utils/: contains tool functions.

## 2. Requirements
Python==3.10.15

Pytorch==2.4.0

Networkx==3.2.1

numpy==2.0.2

dgl==2.4.0

scipy==1.14.1

## 3. Running experiments
We train our model using NVIDIA GeForce RTX 4090 with CUDA 12.2.

For reranking on DataFinder-E:

python run.py --dataset Datafinder --len-seq 10 --num-seqs 10 --dropout 0.1 --beta 0.1 --temperature 0.2 --lr 1e-4 --epoch 50 --feats-type 0 --num-gnns 3 --batch-size 128 --patience 5 --num-layers 3 --eval-steps 200 --num-heads 4 --top-k 5

For reranking on DSEBench:

python run.py --dataset FAERY --len-seq 10 --num-seqs 10  --dropout 0.1 --beta 0.1 --temperature 0.5 --lr 5e-5 --epoch 10 --feats-type 0 --mode qc --num-gnns 1 --batch-size 128 --patience 1 --num-layers 3 --eval-steps 500 --num-heads 4 --top-k 5

