# -*- coding: utf-8 -*-
import os
import codecs
import random

import tensorflow as tf

from model.data_utils import get_processing_word, CoNLLDataset, get_vocabs, UNK, NUM, NUU, FLT, FLU, write_vocab, \
    get_char_vocab
from model.ner_model import NERModel


def process(args):
    # file = 'origin_data.txt'
    # for file in os.listdir(args.input_path):
    # filename = os.path.join(args.input_path, file)
    filename = args.input_path
    output = args.data_path

    train_fp = codecs.open(output + "/train.txt", mode="w", encoding="utf-8")
    valid_fp = codecs.open(output + "/valid.txt", mode="w", encoding="utf-8")

    train_fp.write("\n")
    valid_fp.write("\n")

    dup_total_lines = codecs.open(filename, encoding="utf-8").readlines()
    line_set = set()
    total_lines = []
    for line in dup_total_lines:
        if line in line_set:
            continue
        total_lines.append(line)
        line_set.add(line)

    print('数据集总数：' + str(len(dup_total_lines)))
    print('去重后的数据条数：' + str(len(total_lines)))

    total_lines_num = len(total_lines)
    train_num = int(total_lines_num * args.train_ratio)
    valid_num = int(total_lines_num * args.dev_ratio)
    fp = train_fp

    print('数据条数：' + str(total_lines_num))
    print('训练集数据条数：' + str(train_num))
    print('验证集数据条数：' + str(valid_num))

    random.shuffle(total_lines)

    for idx, line in enumerate(total_lines):
        line = line.strip()
        split_line = line.split("__label__")
        assert len(split_line) == 2, "每行必须要能按__label__划分成两份"
        sentence = split_line[0].strip().split(" ")
        tags = split_line[1].strip().split(" ")
        if len(sentence) != len(tags):
            continue
        # assert len(sentence) == len(tags), "句子与标注必须一致"
        if idx == train_num:
            fp = valid_fp
        # elif idx == (train_num + valid_num):
        #     fp = test_fp

        for word, tag in zip(sentence, tags):
            if len(word.strip()) == 0:
                continue
            fp.write(word.strip() + "  " + tag.strip() + "\n")
        fp.write("\n")

    train_fp.close()
    valid_fp.close()


def build(config):
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev ) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    # config = Config(load=False, args=args)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    train = CoNLLDataset(config.filename_train, processing_word)

    vocab, _ = get_vocabs([train], config.min_count)
    vocab.insert(0, UNK)

    special_flag = [NUM, NUU, FLT, FLU]
    for index, flag in enumerate(special_flag, 1):
        if flag in vocab:
            vocab.remove(flag)
        vocab.insert(index, flag)

    # Generators
    dev = CoNLLDataset(config.filename_dev, processing_word)
    # test = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    _, vocab_tags = get_vocabs([train, dev])

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    vocab_chars.insert(0, UNK)

    write_vocab(vocab_chars, config.filename_chars)


def train(config):
    # build model
    model = NERModel(config)
    model.build()

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)


def save(args):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=args.model_path+'/model.weights/')  # 读取保存的模型

        builder = tf.saved_model.builder.SavedModelBuilder(args.model_path+"/model_pb")
        builder.add_meta_graph_and_variables(
            session,
            [tf.saved_model.tag_constants.SERVING]
        )
        builder.save()
