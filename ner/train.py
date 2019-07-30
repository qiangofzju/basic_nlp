# -*- coding: utf-8 -*-
# created hy HaiYan Yu on 2019/03/05
import argparse

from model.config import Config
from utils import process, build, train, save

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--epoch', type=float, default=15)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    process(args=args)
    config = Config(load=False, args=args)
    build(config)
    config.load()
    train(config)
    save(args=args)
