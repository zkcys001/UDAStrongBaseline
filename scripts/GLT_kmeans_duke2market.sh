#!/bin/sh


for i in $(seq 0 10)
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 sbs_trainkmeans.py -tt market1501 -st dukemtmc -a resnet50_sbs\
        --num-instances 4 --lr 0.00035 --iters 200 -b 64 --epochs 40 --dropout 0.2 --n-jobs 16\
        --init-1 logs/dukemtmcTOmarket1501/resnet50_sbs-pretrain-0_norm/model_best.pth.tar \
        --logs-dir logs/dukemtmcTOmarket1501/resnet50_sbs_GLT \
        --cluster-iter ${i} --choice_c 1 --ncs 700,700,700
done

