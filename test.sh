#!/bin/bash

python3 ./source/main.py test \
    --dataset ptb \
    --test-ptb-path data/ptb/23.auto.clean \
    --embedding-path data/glove/glove.gz \
    --model-path-base models/single_model_best_dev=91.46.pt
