#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset DSEBench --len-seq 10 --num-seqs 10 --dropout 0.1 --beta 0.1 --temperature 0.5 --lr 5e-5 --epoch 10 --feats-type 0 --num-gnns 1 --batch-size 128 --patience 1 --num-layers 3 --eval-steps 500 --num-heads 4 --top-k 5
