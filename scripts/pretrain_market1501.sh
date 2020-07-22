#!/bin/sh


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 source_pretrain.py -ds market1501 -dt dukemtmc -a resnet50 --seed 0 --margin 0.0 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 \
	--eval-step 20 --logs-dir logs/market1501TOdukemtmc/resnet50-pretrain-0
