#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 sbs_traindbscan_unc.py -tt dukemtmc -st market1501 -a resnet50_multi_sbs\
	--num-instances 4 --lr 0.00035 --iters 200 -b 64 --epochs 200 \
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/resnet50_multi_sbs-pretrain-0_norm/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/resnet50_multi_sbs_unc
