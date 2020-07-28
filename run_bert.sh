#!/usr/bin/env bash
python ./source/main.py train \
    --model-path-base ./models/bert_model \
    --epochs 500 \
    --use-bert \
    --const-lada 0.8 \
    --dataset ptb \
    --embedding-path ./data/glove/glove.gz \
    --model-name bert_model \
    --checks-per-epoch 4 \
    --num-layers 2 \
    --learning-rate 0.00005 \
    --batch-size 50 \
    --eval-batch-size 20 \
    --subbatch-max-tokens 500 \
    --train-ptb-path ./data/ptb/02-21.10way.clean \
    --dev-ptb-path ./data/ptb/22.auto.clean
