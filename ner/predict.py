# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
import codecs
import os

from model.config_predict import Config
from model.data_utils import align_data, get_chunks
from model.ner_model import NERModel


def get_word_tags(sentence, tags):
    output = ""
    for tag, start, end in tags:
        output += "".join(sentence[start:end]) + ":" + tag + " -- "
    return output


def get_map(map_file):
    f = codecs.open(map_file, 'r', 'utf-8')
    res_map = {}
    for line in f.readlines():
        items = line.strip().split('\t')
        if len(items) == 2:
            res_map[items[0]] = items[1]
    f.close()
    return res_map


def get_standard(entity_dict):
    standard = {'std_obj': set(), 'std_pos': set(), 'std_pro': set()}
    path = os.path.dirname(__file__)
    obj_map = get_map(path + '/dict/object_map.txt')
    pos_map = get_map(path + '/dict/position_map.txt')
    pro_map = get_map(path + '/dict/problem_map.txt')
    obj_pos_map = get_map(path + '/dict/object_position_map.txt')

    for item in entity_dict['OBJ']:
        if item in obj_map:
            standard['std_obj'].add(obj_map[item])
        else:
            standard['std_obj'].add('其他')
        if item in obj_pos_map:
            standard['std_pos'].add(obj_pos_map[item])

    for item in entity_dict['POS']:
        if item in pos_map:
            standard['std_pos'].add(pos_map[item])
        else:
            standard['std_pos'].add('其他')

    for item in entity_dict['PRO']:
        if item in pro_map:
            standard['std_pro'].add(pro_map[item])
        else:
            standard['std_pro'].add('其他')

    return standard


def get_entity(sentence, tags):
    res = {'OBJ': [], 'POS': [], 'PRO': []}
    p = ''
    s = ''
    for i, tag in enumerate(tags):
        if tag == 'O':
            if p and s:
                if p in res:
                    res[p].append(s)
                else:
                    res[p] = [s]
                p = ''
                s = ''
        elif tag.endswith('_B'):
            if p and s:
                if p in res:
                    res[p].append(s)
                else:
                    res[p] = [s]
            p = tag.split('_')[0]
            s = sentence[i]
        else:
            s += sentence[i]
    if p and s:
        if p in res:
            res[p].append(s)
        else:
            res[p] = [s]

    return res


def predict(model, sentences, test_filename):
    fp = codecs.open(test_filename, mode="w", encoding="utf-8")
    fp.write('input\tpred_res\tpred_position\tpred_object\tpred_problem\tpred_standard_position\tpred_standard_object'
             '\tpred_standard_problem\ttag\ttag_position\ttag_object\ttag_problem\n')

    for i, sentence in enumerate(sentences):
        line = sentence.strip()
        if not line:
            continue

        split_line = line.split("__label__")
        if len(split_line) == 2:
            sentence = split_line[0].strip().split(" ")
            tags = split_line[1].strip().split(" ")
        elif len(split_line) == 1:
            sentence = split_line[0].strip().split(" ")
            tags = []

        preds = model.predict(sentence)
        pred_entity = get_entity(sentence, preds)
        pred_standard = get_standard(pred_entity)
        word_lab_pred = align_data({"input": sentence, "output": preds})

        line = '\t'.join([
            word_lab_pred['input'],
            word_lab_pred['output'],
            ','.join(pred_entity['POS']),
            ','.join(pred_entity['OBJ']),
            ','.join(pred_entity['PRO']),
            ','.join(pred_standard['std_pos']),
            ','.join(pred_standard['std_obj']),
            ','.join(pred_standard['std_pro']),
        ])

        if tags and len(sentence) == len(tags):
            word_lab = align_data({"input": sentence, "output": tags})
            tag_entity = get_entity(sentence, tags)
            line += '\t'.join(['',
                               word_lab['output'],
                               ','.join(tag_entity['POS']),
                               ','.join(tag_entity['OBJ']),
                               ','.join(tag_entity['PRO'])
                               ]) + '\n'
        else:
            line += '\t\t\t\t\n'

        fp.write(line)

        # lab_chunks = list(get_chunks(list(map(lambda x: model.config.vocab_tags[x], tags)), model.config.vocab_tags))
        # lab_pred_chunks = list(get_chunks(list(map(lambda x: model.config.vocab_tags[x], preds)),
        #                                  model.config.vocab_tags))

        # fp.write("sent:" + split_line[0].strip() + "\n")
        # fp.write("true: " + get_word_tags(sentence,lab_chunks) + "\n")
        # fp.write("pred: " + get_word_tags(sentence,lab_pred_chunks) + "\n")

        if i % 1000 == 0:
            print(str(i))
        fp.flush()
    fp.close()


def main(args):
    # create instance of config
    config = Config(args=args)

    # build model
    model = NERModel(config)
    model.build(True)
    model.restore_session(config.dir_model)
    filename_test = args.input_path
    total_sentences = codecs.open(filename_test, encoding="utf-8").readlines()

    test_filename = config.dir_output + "test_pred.txt"

    predict(model, total_sentences, test_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    main(args)
