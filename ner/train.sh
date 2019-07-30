#!/usr/bin/env bash

python train.py \
--input_path=./data/train_and_dev.txt \
--data_path=./data \
--model_path=./result \
--train_ratio=0.8 \
--dev_ratio=0.2 \
--epoch=5 \
--optimizer=adam \
--dropout=0.5
