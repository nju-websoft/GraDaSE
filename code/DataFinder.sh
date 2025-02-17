#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset DataFinder --len-seq 10 --num-seqs 10  --dropout 0.1 --beta 0.1 --temperature 0.2 --lr 1e-4 --epoch 50 --feats-type 0 --num-gnns 3 --batch-size 128 --patience 5 --num-layers 3 --weight-decay 1e-4 --eval-steps 200 --num-heads 4 --repeat 5 --top-k 5
