#!bin/bash

python3 ./source/main.py train \
    --model-path-base ./models/single_model \
    --epochs 500 \
    --use-chars-lstm \
    --use-words \
    --use-tags \
    --use-cat \
    --num-layers 12 \
    --dataset ptb \
    --embedding-path ./data/glove/glove.gz \
    --model-name joint_single \
    --embedding-type glove \
    --checks-per-epoch 4 \
    --train-ptb-path ./data/ptb/02-21.10way.clean \
    --dev-ptb-path ./data/ptb/22.auto.clean
