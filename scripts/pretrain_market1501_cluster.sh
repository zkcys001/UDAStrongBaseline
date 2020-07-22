#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/anaconda3/bin/python source_pretrain.py -ds market1501 -dt dukemtmc -a resnet50_sbs --seed 0 --margin 0.0 \
	--num-instances 4 -b 64 -j 16 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 \
	--eval-step 20 --logs-dir logs/market1501TOdukemtmc/resnet50_sbs-pretrain-0_norm_4gpu \
	--resume '' --data-dir '/zhengk/Datasets/reid/'

