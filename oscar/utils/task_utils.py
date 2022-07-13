# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import _pickle as cPickle
import csv
import json
import logging
import os
import sys
from io import open

import torch

logger = logging.getLogger(__name__)


class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, score=None, img_key=None, q_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.score = score
        self.img_key = img_key
        self.q_id = q_id


class InputFeat(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, score, img_feat):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.score = score
        self.img_feat = img_feat


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class VQATextProcessor(DataProcessor):
    """ Processor for the VQA Text data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):  # datasets/vqa train2014_qla_mrcnn.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/vqa/train2014_qla_mrcnn.json
        # print(lines)  # 647480
        '''
        [
            {
                "q": "How many cookies can be seen?", 
                "o": "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple", 
                "an": [1504], 
                "s": [1.0], 
                "img_id": 9
            }, 
            ...
        ]
        '''
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):  # datasets/vqa val2014_qla_mrcnn.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/vqa/val2014_qla_mrcnn.json
        # print(lines)  # 10631
        '''
        [
            {
                "q": "What is he sitting on?", 
                "o": "person person bottle cup person cup remote couch handbag couch frisbee couch person potted plant person", 
                "an": [487, 2969, 2898], 
                "s": [0.9, 0.6, 1.0], 
                "img_id": 241
            },
            ...
        ]
        '''
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):  # datasets/vqa test2015_qla_mrcnn.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/vqa/test2015_qla_mrcnn.json
        # print(lines)  # 447793
        '''
        [
            {
                "q_id": "1000", 
                "q": "What is the fence made of?", 
                "o": "parking meter truck car car person tv car truck car car", 
                "an": null, 
                "s": null, 
                "img_id": 1
            },
            ...
        ]
        '''
        return self._create_examples(lines, "test")

    def get_labels(self, label_file):  # datasets/vqa/trainval_ans2label.pkl
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))  # datasets/vqa/trainval_ans2label.pkl
        # print(ans2label)  # 长度为3129的dict
        '''
        {
            '': 0, 
            'boats': 835, 
            'not at all': 2421, 
            'name': 1, 
            'harley davidson': 78, 
            ..., 
            'stopping': 3128
        }
        '''
        label_list = list(ans2label.values())  # 长度为3129的list
        # print(label_list)
        '''
        [0, 835, 2421, 1, 78, ..., 3128]
        '''
        return label_list

    def _create_examples(self, lines, set_type):  # datasets/vqa/train2014_qla_mrcnn.json文件内容 'train'
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            # print(line)
            '''
            {
                "q": "How many cookies can be seen?", 
                "o": "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple", 
                "an": [1504], 
                "s": [1.0], 
                "img_id": 9
            }
            '''
            if set_type != 'test' and len(line['an']) == 0:  # set_type为'train'或'dev'且line['an']=[]时跳过该行(即跳过训练集和验证集中无answer的数据)
                continue

            guid = "%s-%s" % (set_type, str(i))  # "train-0" 或 "dev-0" 或 "test-0"
            text_a = line['q']  # "How many cookies can be seen?"
            text_b = line['o'].replace(';', ' ').strip()  # "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple"
            label = None if set_type.startswith('test') else line['an']  # [1504] 或 [487, 2969, 2898] 或 None
            score = None if set_type.startswith('test') else line['s']  # [1.0] 或 [0.9, 0.6, 1.0] 或 None
            img_key = line['img_id']  # 9 或 241 或 1
            q_id = int(line['q_id']) if set_type.startswith('test') else 0  # 0 或 0 或 1000
            examples.append(
                InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id)
            )
        # print(examples)
        '''
        [
            InputInstance(
                guid="train-0", 
                text_a="How many cookies can be seen?", 
                text_b="bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple",
                label=[1504]
                score=[1.0], 
                img_key=9, 
                q_id=0
            ),
            ...
        ]
        '''
        return examples


class VQATextAProcessor(DataProcessor):
    """ Processor for the VQA Text data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2014_qla.tsv")), "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "val2014_qla.tsv")), "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        return list(ans2label.values())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type!='test' and len(line['an']) == 0: continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = None  # line['o'] # or None
            label = None if set_type.startswith('test') else line['an']
            score = None if set_type.startswith('test') else line['s']
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples


class GQAProcessor(DataProcessor):
    """ Processor for the GQA data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):  # datasets/gqa gqa_all_qla_train.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/gqa/gqa_all_qla_train.json
        # print(lines)  # 16317209
        '''
        [
            {
                'q': 'What is on the white wall?', 
                'o': 'Animal Window Tree Horse Door Furniture Building Animal House Horse Bench', 
                'an': [628], 
                'img_id': '2375429', 
                'q_id': '07333408'
            }, 
            ...
        ]
        '''
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):  # datasets/gqa gqa_all_qla_val.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/gqa/gqa_all_qla_val.json
        # print(lines)  # 16317209
        '''
        [
            {
                "q": "Do the shorts have dark color?", 
                "o": "Person Footwear Tree Man Tree Tree Tree Tree Tree Tree", 
                "an": [1], 
                "img_id": "n288870", 
                "q_id": "20968379"
            }, 
            ...
        ]
        '''
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):  # datasets/gqa gqa_all_qla_submission.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/gqa/gqa_all_qla_submission.json
        # print(lines)  # 16317209
        '''
        [
            {
                "q": "Do you see a bench to the right of her?", 
                "o": "Person;Human face;Furniture;Woman;Furniture;Personal care;Glasses;Bench;Bench;Trousers;Training bench", 
                "an": null, 
                "img_id": "2365049", 
                "q_id": "11183447"
            }, 
            ...
        ]
        '''
        return self._create_examples(lines, "test")

    def get_labels(self, label_file='trainval_testdev_all_ans2label.pkl'):  # datasets/gqa/trainval_testdev_all_ans2label.pkl
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))  # datasets/gqa/trainval_testdev_all_ans2label.pkl
        # print(ans2label)  # 长度为1853的dict
        '''
        {'no': 0, 'yes': 1, 'towel': 2, 'man': 3, 'sidewalk': 4, ..., 'bottle cap': 1852}
        '''
        label_list = list(ans2label.values())  # 长度为1853的list
        # print(label_list)
        '''
        [0, 1, 2, 3, 4, ..., 1852]
        '''
        return label_list

    def _create_examples(self, lines, set_type):  # datasets/gqa/gqa_all_qla_train.json文件内容 'train'
        """Creates examples for the training and dev sets."""
        # print(lines)  # 长度为16317209的list
        '''
        [
            {
                'q': 'What is on the white wall?', 
                'o': 'Animal Window Tree Horse Door Furniture Building Animal House Horse Bench', 
                'an': [628], 
                'img_id': '2375429', 
                'q_id': '07333408'
            }, 
            ...
        ]
        '''

        examples = []
        for (i, line) in enumerate(lines):
            # print(line)
            '''
            {
                'q': 'What is on the white wall?', 
                'o': 'Animal Window Tree Horse Door Furniture Building Animal House Horse Bench', 
                'an': [628], 
                'img_id': '2375429', 
                'q_id': '07333408'
            }
            '''
            if set_type != 'test' and len(line['an']) == 0:  # set_type为'train'或'dev'且line['an']=[]时跳过该行(即跳过训练集和验证集中无answer的数据)
                continue

            guid = "%s-%s" % (set_type, str(i))  # 'train-0' 或 'dev-0' 或 'test-0'
            text_a = line['q']  # 'What is on the white wall?'
            # text_b = line['o'] # or None
            text_b = line['o'].replace(';', ' ').strip()  # 'Animal Window Tree Horse Door Furniture Building Animal House Horse Bench'
            label = None if set_type.startswith('test') else line['an']  # [628] 或 [1] 或 None
            score = None  # 无score
            img_key = line['img_id']  # '2375429' 或 'n288870' 或 '2365049'
            q_id = int(line['q_id']) if set_type.startswith('test') else 0  # 0 或 0 或 11183447
            examples.append(
                InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id)
            )
        # print(examples)
        '''
        [
            InputInstance(
                guid='train-0', 
                text_a='What is on the white wall?', 
                text_b='Animal Window Tree Horse Door Furniture Building Animal House Horse Bench', 
                label=[628], 
                score=None, 
                img_key='2375429', 
                q_id=0
            ),
            ...
        ]
        '''
        return examples


class NLVRProcessor(DataProcessor):
    """ Processor for the NLVR data set. """

    def get_train_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_train.json'):  # datasets/nlvr2 True nlvr2_train.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/nlvr2/nlvr2_train.json
        # print(lines)  # 86373
        '''
        [
            {
                "img_id": {
                    "left": "nlvr2_train-10171-0-img0", 
                    "right": "nlvr2_train-10171-0-img1"
                }, 
                "q": "An image shows one leather pencil case, displayed open with writing implements tucked inside.", 
                "o": {
                    "left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", 
                    "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"
                }, 
                "label": 0
            }, 
            ...
        ]
        '''
        return self._create_examples(lines, "train", use_label_seq)  # datasets/nlvr2/nlvr2_train.json文件内容 'train' True

    def get_dev_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_dev.json'):  # datasets/nlvr2 True nlvr2_dev.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/nlvr2/nlvr2_dev.json
        # print(lines)  # 6982
        '''
        [
            {
                "img_id": {
                    "left": "nlvr2_dev-850-0-img0", 
                    "right": "nlvr2_dev-850-0-img1"
                }, 
                "q": "The right image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage.", 
                "o": {
                    "left": "garden ground rocks plants mulch rock ground block barrel leaves wall plant barrel wall plants ground dirt rock plant", 
                    "right": "bottles bottles bottle grass bottle bottles bottle bottle bottle bottle bottle bottle label vase bottle label bottle wine bottle label bottles label bottle bottle bottle bottle bottle bucket meter bottle bottle bottles bag bottle wine bottle bottles bottle bottle bottle label bottle bottle label bottles"
                }, 
                "label": 0
            },
            ...
        ]
        '''
        return self._create_examples(lines, "dev", use_label_seq)  # datasets/nlvr2/nlvr2_dev.json文件内容 'dev' True

    def get_test_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_test1.json'):  # datasets/nlvr2 True nlvr2_test1.json
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))  # datasets/nlvr2/nlvr2_test1.json
        # print(lines)  # 6967
        '''
        [
            {
                "img_id": {
                    "left": "nlvr2_test1-0-1-img0", 
                    "right": "nlvr2_test1-0-1-img1"
                }, 
                "q": "There is an empty glass.", 
                "o": {
                    "left": "wall bottle table label glass wall glass sticker wood pole circle lid label shadow skateboard water wood neck neck", 
                    "right": "drink wall picture sign glass wine picture glass man table sign window bottle window table reflection glass letter bowl wall face liquid letter sign letter sign hair"
                }, 
                "label": 0
            },
            ...
        ]
        '''
        return self._create_examples(lines, "test", use_label_seq)  # datasets/nlvr2/nlvr2_test1.json文件内容 'test' True

    def get_labels(self, label_file=None):
        """ See base class."""

        # ans2label = cPickle.load(open(label_file, 'rb'))
        # label_list = list(ans2label.values())
        # return label_list
        return [0, 1]

    def _create_examples(self, lines, set_type, use_label_seq=True):  # datasets/nlvr2/nlvr2_train.json文件内容 'train' True
        """ Creates examples for the training and dev sets. """

        # print(lines)
        '''
        [
            {
                "img_id": {
                    "left": "nlvr2_train-10171-0-img0", 
                    "right": "nlvr2_train-10171-0-img1"
                }, 
                "q": "An image shows one leather pencil case, displayed open with writing implements tucked inside.", 
                "o": {
                    "left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", 
                    "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"
                }, 
                "label": 0
            }, 
            ...
        ]
        '''

        examples = []
        for (i, line) in enumerate(lines):
            # print(line)
            '''
            {
                "img_id": {
                    "left": "nlvr2_train-10171-0-img0", 
                    "right": "nlvr2_train-10171-0-img1"
                }, 
                "q": "An image shows one leather pencil case, displayed open with writing implements tucked inside.", 
                "o": {
                    "left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", 
                    "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"
                }, 
                "label": 0
            }
            '''

            guid = "%s-%s" % (set_type, str(i))  # 'train-0'或'dev-0'或'test-0'
            text_a = line['q']  # "An image shows one leather pencil case, displayed open with writing implements tucked inside."
            text_b = line['o'] if use_label_seq else None  # {"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}
            # label = None if set_type.startswith('test') else line['label']
            label = line['label']  # 0 （test1的label为0）
            score = None  # 无score
            img_key = line['img_id']  # {"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}
            # q_id = int(line['q_id']) if set_type.startswith('test') else 0
            q_id = 0  # 无q_id
            examples.append(
                InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id)
            )
        # print(examples)
        '''
        [
            InputInstance(
                guid="train-0", 
                text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", 
                text_b={"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}, 
                label=0, 
                score=None, 
                img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, 
                q_id=0
            ),
            ...
        ]
        '''
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):  # max_length = args.max_seq_length - 3
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "vqa_text": VQATextProcessor,
    "vqa_text_a": VQATextAProcessor,
    "gqa": GQAProcessor,
    "nlvr": NLVRProcessor
}

output_modes = {
    "vqa_text": "classification",
    "vqa_text_a": "classification",
    "gqa": "classification",
    "nlvr": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "vqa_text": 3129,
    "vqa_text_a": 3129,
    "gqa": 1853,
    "nlvr": 2
}
