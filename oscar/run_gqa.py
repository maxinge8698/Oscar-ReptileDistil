# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time

import _pickle as cPickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')  # 新添加的目录会优先于其他目录被import检查

from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers.pytorch_transformers import WEIGHTS_NAME

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from oscar.utils.misc import set_seed
from oscar.utils.task_utils import _truncate_seq_pair, output_modes, processors

#
import warnings

warnings.filterwarnings('ignore')
#

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification, BertTokenizer),
}

log_json = []

debug_size = 500


class GQADataset(Dataset):
    """ GQA Dataset """

    def __init__(self, args, name, tokenizer, img_features):  # args 'train'或'val'或'test' tokenizer img_features
        super(GQADataset, self).__init__()

        assert name in ['train', 'val', 'test', 'test-dev', 'train+val']

        self.args = args
        self.name = name  # 'train'或'val'或'test'
        self.tokenizer = tokenizer

        self.output_mode = output_modes[args.task_name]  # "classification"

        # load image features
        self.img_features = img_features
        # print(self.img_features)  # 148854
        '''
        {
            ...,
            '2352065': tensor([[ 0.8760,  4.3490,  0.4122,  ...,  0.9983,  0.9449,  0.6555],
                              [ 0.5602,  0.0730,  2.6269,  ...,  0.9983,  0.4950,  0.7504],
                              [ 0.6406,  1.8985,  0.8664,  ...,  0.9552,  0.8789,  0.7888],
                              ...,
                              [ 0.0746, 10.0312,  1.2866,  ...,  0.9983,  0.3342,  0.4202],
                              [ 0.0000,  0.0000,  2.6000,  ...,  0.5594,  0.0615,  0.0673],
                              [ 0.0000,  0.0000,  4.0430,  ...,  0.5451,  0.0337,  0.0465]])
        }
        '''

        self.examples, self.labels = _load_dataset(args, name)  # args 'train'或'val'或'test'
        # print(self.examples)  # 16317209 或 172174 或 4237524
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
        # print(self.labels)  # 1853
        '''
        # [0, 1, 2, 3, 4, ..., 1852]
        '''

        self.label_map = {label: i for i, label in enumerate(self.labels)}
        # print(self.label_map)
        '''
        {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            ...,
            1852, 1852
        }
        '''

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))  # 'train'或'val'或'test' 16317209或172174或4237524
        '''
        03/19/2022 16:43:21 - INFO - __main__ - train Data Examples: 16317209
        或
        03/19/2022 16:41:33 - INFO - __main__ - val Data Examples: 172174
        或
        03/19/2022 17:42:50 - INFO - __main__ - test Data Examples: 4237524
        '''

    def tensorize_example(self,
                          example,  # InputInstance(guid='train-0', text_a='What is on the white wall?', text_b='Animal Window Tree Horse Door Furniture Building Animal House Horse Bench', label=[628], score=None, img_key='2375429', q_id=0)
                          cls_token_at_end=False,  # False
                          pad_on_left=False,  # False
                          cls_token='[CLS]',  # '[CLS]'
                          sep_token='[SEP]',  # '[SEP]'
                          cls_token_segment_id=1,  # 0
                          pad_token_segment_id=0,  # 0
                          pad_token=0,  # 0
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1,
                          mask_padding_with_zero=True):

        # print(example)
        """
        InputInstance(
            guid='train-0',
            text_a='What is on the white wall?',
            text_b='Animal Window Tree Horse Door Furniture Building Animal House Horse Bench',
            label=[628],
            score=None,
            img_key='2375429',
            q_id=0
        )
        """

        tokens_a = self.tokenizer.tokenize(example.text_a)  # ['what', 'is', 'on', 'the', 'white', 'wall', '?']

        tokens_b = None
        if example.text_b:  # 'Animal Window Tree Horse Door Furniture Building Animal House Horse Bench',
            tokens_b = self.tokenizer.tokenize(example.text_b)  # ['animal', 'window', 'tree', 'horse', 'door', 'furniture', 'building', 'animal', 'house', 'horse', 'bench']
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
        else:  # Account for [CLS] and [SEP] with "- 2"
            """ PASS """
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]  # Account for [CLS] and [SEP] with "- 2"
            """ PASS """

        tokens = tokens_a + [sep_token]  # ['what', 'is', 'on', 'the', 'white', 'wall', '?', '[SEP]']
        segment_ids = [sequence_a_segment_id] * len(tokens)  # [0, 0, 0, 0, 0, 0, 0, 0]

        if tokens_b:  # ['animal', 'window', 'tree', 'horse', 'door', 'furniture', 'building', 'animal', 'house', 'horse', 'bench']
            tokens += tokens_b + [sep_token]  # ['what', 'is', 'on', 'the', 'white', 'wall', '?', '[SEP]', 'animal', 'window', 'tree', 'horse', 'door', 'furniture', 'building', 'animal', 'house', 'horse', 'bench', '[SEP]']
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if cls_token_at_end:
            """ PASS """
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            """ PASS """
        else:
            tokens = [cls_token] + tokens  # ['[CLS]', 'what', 'is', 'on', 'the', 'white', 'wall', '?', '[SEP]', 'animal', 'window', 'tree', 'horse', 'door', 'furniture', 'building', 'animal', 'house', 'horse', 'bench', '[SEP]']
            segment_ids = [cls_token_segment_id] + segment_ids  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # [101, 2054, 2003, 2006, 1996, 2317, 2813, 1029, 102, 4111, 3332, 3392, 3586, 2341, 7390, 2311, 4111, 2160, 3586, 6847, 102]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)  # 128 - 21 = 107
        if pad_on_left:
            """ PASS """
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            """ PASS """
        else:
            input_ids = input_ids + ([pad_token] * padding_length)  # [101, 2054, 2003, 2006, 1996, 2317, 2813, 1029, 102, 4111, 3332, 3392, 3586, 2341, 7390, 2311, 4111, 2160, 3586, 6847, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features
        if self.args.img_feature_type.startswith('dis_code'):
            """ PASS """
            img_feat = self.img_features[example.img_key]
            if self.args.img_feature_type == 'dis_code_ln':  # for discrete code image representation
                img_feat = img_feat.reshape(-1, img_feat.shape[0])
            if self.args.img_feature_type == 'dis_code_t':  # transposed
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * 64
            else:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
            """ PASS """
        else:  # faster_r-cnn
            # print(self.img_features)  # 148854
            '''
            {
                ...,
                '2352065': tensor([[ 0.8760,  4.3490,  0.4122,  ...,  0.9983,  0.9449,  0.6555],
                                  [ 0.5602,  0.0730,  2.6269,  ...,  0.9983,  0.4950,  0.7504],
                                  [ 0.6406,  1.8985,  0.8664,  ...,  0.9552,  0.8789,  0.7888],
                                  ...,
                                  [ 0.0746, 10.0312,  1.2866,  ...,  0.9983,  0.3342,  0.4202],
                                  [ 0.0000,  0.0000,  2.6000,  ...,  0.5594,  0.0615,  0.0673],
                                  [ 0.0000,  0.0000,  4.0430,  ...,  0.5451,  0.0337,  0.0465]])
            }
            '''
            img_feat = self.img_features[example.img_key]  # '2375429'
            # print(img_feat)  # (22, 2054)
            '''
            tensor([[0.0246, 0.0482, 0.6033,  ..., 0.7129, 0.6491, 0.7129],
                    [0.0000, 0.5073, 0.3817,  ..., 0.9244, 0.7513, 0.8781],
                    [0.0422, 0.0039, 0.1156,  ..., 0.5032, 0.5584, 0.5032],
                    ...,
                    [0.0000, 0.0000, 0.0000,  ..., 0.6762, 0.0927, 0.2839],
                    [0.0000, 0.1310, 0.0000,  ..., 0.9649, 0.2411, 0.1644],
                    [0.0000, 0.0000, 0.0000,  ..., 0.8226, 0.1893, 0.4761]])
            '''
            if img_feat.shape[0] > self.args.max_img_seq_length:  # 22 > 50
                img_feat = img_feat[0:self.args.max_img_seq_length, ]  # (50, 2054)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+50
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:  # 22 < 50
                if self.args.max_img_seq_length > 0:  # 50 > 0
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+22: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))  # (50-22, 2054)
                img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((22, 2054), (28, 2054), dim=0) -> (50, 2054)
                if self.args.max_img_seq_length > 0:  # 50 > 0
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])  # 150+28: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]
        # print(len(input_ids), len(input_mask), len(segment_ids), img_feat.shape)  # 128 178 128 (50, 2054)

        if self.args.output_mode == "classification":  # "classification"
            if example.label is None:  # 测试集example.label=None, example.score=None
                label_id = [0]  # 测试集的label_id=[0]
                score = [0]  # 测试集的score=[0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:  # 训练集和验证集example.label=[628], example.score=None
                label_id = [self.label_map[l] for l in example.label]  # [628] -> [628]
                score = [0]  # 测试集和验证集的score=[0]
        elif self.args.output_mode == "regression":
            """ PASS """
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
            """ PASS """
        else:
            raise KeyError(self.args.output_mode)

        """ PASS """
        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            # img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)
        """ PASS """

        return (torch.tensor(input_ids, dtype=torch.long),  # (128,):
                torch.tensor(input_mask, dtype=torch.long),  # (178,):
                torch.tensor(segment_ids, dtype=torch.long),  # (128,):
                torch.tensor([label_id[0]], dtype=torch.long),  # (1,): tensor([628])
                img_feat,  # (50, 2054)
                torch.tensor([example.q_id], dtype=torch.long))  # (1,): tensor([0])

    def __getitem__(self, index):
        # print(self.examples)
        """
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
        """
        entry = self.examples[index]
        # print(entry)
        '''
        InputInstance(
            guid='train-0',
            text_a='What is on the white wall?',
            text_b='Animal Window Tree Horse Door Furniture Building Animal House Horse Bench',
            label=[628],
            score=None,
            img_key='2375429',
            q_id=0
        )
        '''
        example = self.tensorize_example(entry,  # InputInstance(guid='train-0', text_a='What is on the white wall?', text_b='Animal Window Tree Horse Door Furniture Building Animal House Horse Bench', label=[628], score=None, img_key='2375429', q_id=0)
                                         cls_token_at_end=bool(self.args.model_type in ['xlnet']),  # False
                                         pad_on_left=bool(self.args.model_type in ['xlnet']),  # False
                                         cls_token=self.tokenizer.cls_token,  # '[CLS]'
                                         sep_token=self.tokenizer.sep_token,  # '[SEP]'
                                         cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,  # 0
                                         pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)  # 0
        # print(example)
        '''
        (
            input_ids: (128,),
            input_mask: (178,),
            segment_ids: (128,),
            label_id: (1,),
            img_feat: (50, 2054),
            q_id: (1,)
        )        
        '''
        return example

    def __len__(self):
        return len(self.examples)


def _load_dataset(args, name):  # args 'train'或'val'或'test'
    processor = processors[args.task_name]()  # GQAProcessor
    labels = processor.get_labels(args.label_file)  # datasets/gqa/trainval_testdev_all_ans2label.pkl
    # print(labels)  # 长度为1853的list
    '''
    [0, 1, 2, 3, 4, ..., 1852]
    '''

    if name == 'train':
        # if args.data_label_type == 'mask':
        if args.train_data_type == 'bal':
            """ PASS """
            examples = processor.get_train_examples(args.data_dir, 'gqa_bal_qla_train.json')
            """ PASS """
        else:  # args.train_data_type=all
            examples = processor.get_train_examples(args.data_dir, 'gqa_all_qla_train.json')  # datasets/gqa gqa_all_qla_train.json
            # print(examples)  # 16317209
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
    elif name == 'val':
        # if args.data_label_type == 'mask':
        if args.eval_data_type == 'bal':
            """ PASS """
            examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_val.json')
            """ PASS """
        else:  # args.eval_data_type=all
            examples = processor.get_dev_examples(args.data_dir, 'gqa_all_qla_val.json')  # datasets/gqa/gqa_all_qla_val.json
            # print(examples)  # 172174
            '''
            [
                InputInstance(
                    guid='dev-0', 
                    text_a='Do the shorts have dark color?', 
                    text_b='Person Footwear Tree Man Tree Tree Tree Tree Tree Tree', 
                    label=[1], 
                    score=None, 
                    img_key='n288870', 
                    q_id=0
                ),
                ...
            ]
            '''
    elif name == 'test':  # test-submission
        # if args.data_label_type == 'mask':
        if args.test_data_type == 'bal':
            """ PASS """
            examples = processor.get_test_examples(args.data_dir, 'gqa_bal_qla_submission.json')
            """ PASS """
        else:  # args.test_data_type=all
            examples = processor.get_test_examples(args.data_dir, 'gqa_all_qla_submission.json')  # datasets/gqa gqa_all_qla_submission.json
            # print(examples)  # 4237524
            '''
            [
                InputInstance(
                    guid='test-0', 
                    text_a='Do you see a bench to the right of her?', 
                    text_b='Person Human face Furniture Woman Furniture Personal care Glasses Bench Bench Trousers Training bench', 
                    label=None, 
                    score=None, 
                    img_key='2365049', 
                    q_id=11183447
                ),
                ...
            ]
            '''
    elif name == 'test-dev':  # test-dev set
        """ PASS """
        # if args.data_label_type == 'mask':
        if args.test_data_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_testdev.json')
        else:
            examples = processor.get_dev_examples(args.data_dir, 'gqa_all_qla_testdev.json')
        """ PASS """

    return examples, labels


def _load_img_features(args):
    t_start = time.time()
    if args.img_feature_type == 'faster_r-cnn':  # faster_r-cnn
        if args.img_feature_dim == 2048:  # object features: 2048
            """ PASS """
            feat_file_name = 'gqa_img_frcnn_obj_feats.pt'
            """ PASS """
        else:  # object + spatial features: 2054
            feat_file_name = 'gqa_img_frcnn_feats.pt'  # gqa_img_frcnn_feats.pt
    else:
        """ PASS """
        feat_file_name = 'gqa_img_feats.pt'
        """ PASS """
    img_features = torch.load(os.path.join(args.data_dir, feat_file_name))  # datasets/gqa/gqa_img_frcnn_feats.pt
    # print(img_features)  # 148854
    '''
    {
        ...,
        '2352065': tensor([[ 0.8760,  4.3490,  0.4122,  ...,  0.9983,  0.9449,  0.6555],
                           [ 0.5602,  0.0730,  2.6269,  ...,  0.9983,  0.4950,  0.7504],
                           [ 0.6406,  1.8985,  0.8664,  ...,  0.9552,  0.8789,  0.7888],
                           ...,
                           [ 0.0746, 10.0312,  1.2866,  ...,  0.9983,  0.3342,  0.4202],
                           [ 0.0000,  0.0000,  2.6000,  ...,  0.5594,  0.0615,  0.0673],
                           [ 0.0000,  0.0000,  4.0430,  ...,  0.5451,  0.0337,  0.0465]])
    }
    '''
    t_end = time.time()
    logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))
    '''
    03/19/2022 16:35:05 - INFO - __main__ - Info: loading gqa_img_frcnn_feats.pt features using 68.73 secs
    '''
    return img_features


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """

    # print(train_dataset)
    '''
    GQADataset(Dataset): 初始化参数为args, name, tokenizer, img_features
        self.args: args
        self.name: 'train'或'val'或'test'
        self.tokenizer: tokenizer

        self.output_mode: "classification"

        self.img_features:  {
                                ...,
                                '2352065': tensor([[ 0.8760,  4.3490,  0.4122,  ...,  0.9983,  0.9449,  0.6555],
                                                   [ 0.5602,  0.0730,  2.6269,  ...,  0.9983,  0.4950,  0.7504],
                                                   [ 0.6406,  1.8985,  0.8664,  ...,  0.9552,  0.8789,  0.7888],
                                                   ...,
                                                   [ 0.0746, 10.0312,  1.2866,  ...,  0.9983,  0.3342,  0.4202],
                                                   [ 0.0000,  0.0000,  2.6000,  ...,  0.5594,  0.0615,  0.0673],
                                                   [ 0.0000,  0.0000,  4.0430,  ...,  0.5451,  0.0337,  0.0465]])
                            }
        self.examples:  [
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
        self.labels: [0, 1, 2, 3, 4, ..., 1852]
        self.label_map: {
                            0: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            ...,
                            1852, 1852
                        }

        def __getitem__(index): # 对每一个self.examples[index]进行处理后得到:
                                (
                                    input_ids: (128,),
                                    input_mask: (178,),
                                    segment_ids: (128,),
                                    label_id: (1,),
                                    img_feat: (50, 2054),
                                    q_id: (1,)
                                )
        def __len__(): len(self.examples)
    '''

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # 16 * 1
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)  # RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  # num_workers=args.workers,  # 默认为0
                                  sampler=train_sampler,  # RandomSampler(train_dataset)
                                  batch_size=args.train_batch_size)  # 16
    # print(train_dataloader)  # 16317209 / 16 = 1019826
    '''
    [
        [
            torch.Size([16, 128]),
            torch.Size([16, 178]),
            torch.Size([16, 128]),
            torch.Size([16, 1]),
            torch.Size([16, 50, 2054]),
            torch.Size([16, 1]) 
        ],
        [
            torch.Size([16, 128]),
            torch.Size([16, 178]),
            torch.Size([16, 128]),
            torch.Size([16, 1]),
            torch.Size([16, 50, 2054]),
            torch.Size([16, 1]) 
        ],
        ...
    ]
    '''
    if args.max_steps > 0:  # args.max_step=-1
        """ PASS """
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        """ PASS """
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs  # 1019826 // 1 * 5.0 = 5099130.0

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }

    ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  # original
    if args.optim == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,  # 5e-5
                          eps=args.adam_epsilon)  # 1e-8
    elif args.optim == 'Adamax':
        optimizer = torch.optim.Adamax(optimizer_grouped_parameters,
                                       lr=args.learning_rate,
                                       eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)  # original
    if args.scheduler == "constant":  # constant warmup and decay
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":  # linear warmup and decay
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    """ PASS """
    # apex fp16 initialization
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    """ PASS """

    """ PASS """
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    """ PASS """

    """ PASS """
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    """ PASS """

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))  # 16317209
    logger.info("  Num Epochs = %d", args.num_train_epochs)  # 5.0
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)  # 16
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))  # 16 * 1 * 1
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)  # 1
    logger.info("  Total optimization steps = %d", t_total)  # 5099130
    '''
    04/02/2022 13:12:09 - INFO - __main__ - ***** Running training *****
    04/02/2022 13:12:09 - INFO - __main__ -   Num examples = 500
    04/02/2022 13:12:09 - INFO - __main__ -   Num Epochs = 2
    04/02/2022 13:12:09 - INFO - __main__ -   Instantaneous batch size per GPU = 16
    04/02/2022 13:12:09 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 16
    04/02/2022 13:12:09 - INFO - __main__ -   Gradient Accumulation steps = 1
    04/02/2022 13:12:09 - INFO - __main__ -   Total optimization steps = 64
    '''

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])  # 0~4

    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': model.state_dict(),  # copy.deepcopy(model)
        'optimizer_state': optimizer.state_dict()
    }

    # for epoch in range(int(args.num_train_epochs)):
    for epoch in train_iterator:  # 0~4

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])  # 0~1019825

        # total_loss = 0
        # total_norm = 0
        # count_norm = 0

        t_start = time.time()

        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(epoch_iterator):  # 0~1019825
            # print(batch)
            '''
            [
                torch.Size([16, 128])  # input_ids
                torch.Size([16, 178])  # input_mask
                torch.Size([16, 128])  # segment_ids
                torch.Size([16, 1])  # label_id
                torch.Size([16, 50, 2054])  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],  # input_ids: (16, 128)
                'attention_mask': batch[1],  # input_mask: (16, 178)
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 128)
                'labels': batch[3],  # label_ids: (16, 1)
                'img_feats': None if args.img_feature_dim == -1 else batch[4]  # img_feat: (16, 50, 2054)
            }
            outputs = model(**inputs)
            '''
            (
                loss: torch(数)
                logits: (16, 1853),
                hidden_states: tuple(13个(16, 178, 768)),
                attentions: tuple(12个(16, 12, 178, 178))
            )
            '''

            loss, logits = outputs[:2]
            # print(loss, logits.shape)  # torch(数) (16, 1853)

            """ PASS """
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            """ PASS """

            """ PASS """
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            """ PASS """

            if args.fp16:
                """ PASS """
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                """ PASS """
            else:
                loss.backward()
                # total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # count_norm += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 1.0

            tr_loss += loss.item()
            # total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # args.logging_steps=-1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:  # Log metrics
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()
                    # if args.local_rank in [-1, 0] and args.evaluate_during_training:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch + 1, global_step))
                        '''
                        
                        '''
                        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
                        # print(eval_result, eval_score)  # {"acc": acc} acc
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch + 1
                            best_model['model'] = copy.deepcopy(model)

                    logging_loss = tr_loss

                # args.save_steps=-1
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        logger.info("***** Epoch: {} *****".format(epoch + 1))
        logger.info("  Train Loss: {}".format(tr_loss / len(train_dataset)))
        '''
        04/02/2022 13:12:19 - INFO - __main__ - ***** Epoch: 1 *****
        04/02/2022 13:12:19 - INFO - __main__ -   Train Loss: 0.34573886251449587
        '''

        t_end = time.time()
        logger.info('  Train Time Cost: %.3f' % (t_end - t_start))
        '''
        04/02/2022 13:12:19 - INFO - __main__ - Train Time Cost: 9.656
        '''

        # 每个epoch结束时做一次evaluation并保存模型
        # evaluation
        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix='')
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(model)
            # best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())
        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch > 0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint {0} to {1}".format(epoch + 1, output_dir))
            '''
            04/02/2022 13:12:21 - INFO - __main__ - Saving model checkpoint 1 to model/gqa/teacher\checkpoint-1
            '''

        epoch_log = {'epoch': epoch + 1, 'eval_score': eval_score, 'best_score': best_score}
        log_json.append(epoch_log)

        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch + 1, t_end - t_start))
        logger.info('********************')
        '''
        04/02/2022 13:12:21 - INFO - __main__ - Epoch: 1, Train Time: 11.964
        04/02/2022 13:12:21 - INFO - __main__ - ********************
        '''

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # 所有epoch结束后保存最好的模型
    if args.local_rank in [-1, 0]:  # Save the final model checkpoint
        # output_dir = os.path.join(args.output_dir, 'best'.format(best_model['epoch']))  # results/gqa/best
        output_dir = args.output_dir  # model/gqa/teacher
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))
        '''
        04/02/2022 13:12:29 - INFO - __main__ - Saving the best model checkpoint epoch 1 to model/gqa/teacher
        '''

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    eval_task_names = (args.task_name,)  # ('gqa',)
    eval_outputs_dirs = (args.output_dir,)  # ('results/gqa',)

    results = {}
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # 'gqa' 'results/gqa'
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 16 * 1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     # num_workers=args.workers,  # 默认为0
                                     sampler=eval_sampler,  # SequentialSampler(eval_dataset)
                                     batch_size=args.eval_batch_size)  # 16
        # print(eval_dataloader)  # 172174/16=10761
        '''
        [
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ],
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ]
        ]
        '''

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))  # 172174
        logger.info("  Batch size = %d", args.eval_batch_size)  # 16
        '''
        04/02/2022 13:31:42 - INFO - __main__ - ***** Running evaluation  *****
        04/02/2022 13:31:42 - INFO - __main__ -   Num examples = 172174
        04/02/2022 13:31:42 - INFO - __main__ -   Batch size = 16
        '''

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # num_data = 0
        correct_num = 0
        # results_dict = {}

        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # print(batch)
            '''
            [
                torch.Size([16, 128])  # input_ids
                torch.Size([16, 178])  # input_mask
                torch.Size([16, 128])  # segment_ids
                torch.Size([16, 1])  # label_id
                torch.Size([16, 50, 2054])  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''
            # print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape, batch[5].shape)

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],  # input_ids: (16, 128)
                    'attention_mask': batch[1],  # input_mask: (16, 178)
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 128)
                    'labels': batch[3],  # label_id: (16, 1)
                    'img_feats': None if args.img_feature_dim == -1 else batch[4]  # img_feat: (16, 50, 2054)
                }
                outputs = model(**inputs)
                # print(outputs)
                '''
                (
                    loss: torch(数)
                    logits: (16, 1853),
                    hidden_states: tuple(13个(16, 178, 768)),
                    attentions: tuple(12个(16, 12, 178, 178))
                )
                '''
                tmp_eval_loss, logits = outputs[:2]
                # print(tmp_eval_loss, logits.shape)  # tensor(数) (16, 1853)

                eval_loss += tmp_eval_loss.mean().item()

                # 法一
                # val, idx = logits.max(1)  # (values: (16,), indices: (16,))
                # batch_acc = torch.sum(idx == batch[3].view(-1)).item()
                # correct_num += batch_acc
                # 法二
                correct = logits.argmax(1) == batch[3].view(-1)
                correct_num += correct.sum().item()

                # num_data += logits.size(0)

            nb_eval_steps += 1
            # if preds is None:
            #    preds = logits.detach().cpu().numpy()
            #    out_label_ids = inputs['labels'].detach().cpu().numpy()
            # else:
            #    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logger.info("  Eval Loss = %f" % eval_loss)
        '''
        04/02/2022 13:42:17 - INFO - __main__ -   Eval Loss = 1.059539
        '''
        # if args.output_mode == "classification":
        #    preds = np.argmax(preds, axis=1)
        # elif args.output_mode == "regression":
        #    preds = np.squeeze(preds)
        # result = compute_metrics(eval_task, preds, out_label_ids)
        # results.update(result)
        #
        # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #    logger.info("***** Eval results {} *****".format(prefix))
        #    for key in sorted(result.keys()):
        #        logger.info("  %s = %s", key, str(result[key]))
        #        writer.write("%s = %s\n" % (key, str(result[key])))

        acc = float(correct_num) / len(eval_dataloader.dataset)
        logger.info("  Eval Accuracy: {}".format(100 * acc))
        logger.info("  EVALERR: {:.2f}%".format(100 * acc))
        '''
        04/02/2022 13:42:17 - INFO - __main__ -   Eval Accuracy: 80.4198078687839
        04/02/2022 13:42:17 - INFO - __main__ -   EVALERR: 80.42%
        '''
        results.update({'acc': acc})

        # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results {} *****".format(prefix))
        #     for key in sorted(results.keys()):
        #         logger.info("  %s = %s", key, str(results[key]))
        #         writer.write("%s = %s\n" % (key, str(results[key])))

        # with open(os.path.join(args.data_dir, 'val_results.json'), 'w') as f:
        #     json.dump(results_dict, f)

    t_end = time.time()
    logger.info('  Eval Time Cost: %.3f' % (t_end - t_start))
    '''
    04/02/2022 13:42:17 - INFO - __main__ -   Eval Time Cost: 634.178
    '''

    return results, acc


def test(args, model, eval_dataset=None, prefix=""):
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    eval_task_names = (args.task_name,)  # ('gqa',)
    eval_outputs_dirs = (args.output_dir,)  # ('results/gqa',)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))  # datasets/gqa/trainval_testdev_all_label2ans.pkl
    # print(label2ans)  # 长度为1853的dict
    '''
    {
        0: 'no', 
        1: 'yes', 
        2: 'towel', 
        3: 'man', 
        4: 'sidewalk', 
        ..., 
        1851: 'lab coat', 
        1852: 'bottle cap'
    }
    '''
    logger.info('label2ans: %d' % (len(label2ans)))  # 1853
    '''
    03/19/2022 17:42:53 - INFO - __main__ - label2ans: 1853
    '''

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # gqa results/gqa
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 16 * 1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,  # SequentialSampler(eval_dataset)
                                     batch_size=args.eval_batch_size)  # 16
        # print(eval_dataloader)  # 4237524/16=264846
        '''
        [
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ],
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ]
        ]
        '''

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Test!
        logger.info("***** Running test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))  # 4237524
        logger.info("  Batch size = %d", args.eval_batch_size)  # 16
        '''
        
        '''

        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # print(batch)
            '''
            [
                torch.Size([16, 128])  # input_ids
                torch.Size([16, 178])  # input_mask
                torch.Size([16, 128])  # segment_ids
                torch.Size([16, 1])  # label_id
                torch.Size([16, 50, 2054])  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],  # input_ids: (16, 128)
                    'attention_mask': batch[1],  # input_mask: (16, 178)
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 128)
                    'labels': None,
                    'img_feats': None if args.img_feature_dim == -1 else batch[4]  # img_feat: (16, 50, 2054)
                }
                outputs = model(**inputs)
                # print(outputs)
                '''
                (
                    logits: (16, 1853),
                    hidden_states: tuple(13个(16, 178, 768)),
                    attentions: tuple(12个(16, 12, 178, 178))
                )
                '''
                logits = outputs[0]  # (16, 1853)

                val, idx = logits.max(1)  # (values: (16,), indices: (16,))
                # logger.info('idx: %s, batch[5]: %s' % (str(idx.shape), str(batch[5].shape)))  # idx为预测值, batch[5]为q_id: (16, 1)

                for i in range(idx.size(0)):  # idx.size(0)=16
                    # print(eval_dataset.labels)
                    '''
                    [0, 1, 2, 3, 4, ..., 1852]
                    '''

                    """ [{"questionId": str, "prediction": str}, ...] """
                    result = {
                        'questionId': str(batch[5][i].item()),  # q_id
                        'prediction': label2ans[eval_dataset.labels[idx[i].item()]]
                    }
                    results.append(result)

                    # if len(results) % 2000 == 0:
                    #     logger.info("PROGRESS: {}%".format(round(100 * len(results) / len(eval_dataset), 4)))
                    #     '''
                    #     3/19/2022 17:43:04 - INFO - __main__ - PROGRESS: 0.0472%
                    #     '''

        with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:  # model/gqa/teacher/test_results.json
            json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))
    '''
    
    '''


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    # Text
    parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")
    # parser.add_argument("--data_label_type", default='mask', type=str, help="faster or mask")
    parser.add_argument("--train_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--test_data_type", default='bal', type=str, help="bal or all")

    # Image
    # parser.add_argument("--img_feat_dir", default=None, type=str, help="The input img_feat_dir.")
    # parser.add_argument("--img_feat_format", default='pt', type=str, help="img_feat_format: pt or tsv.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    # Dataset
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')

    # Model configuration
    parser.add_argument("--loss_type", default='ce', type=str, help="kl or bce or ce")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    # parser.add_argument("--use_img_layernorm", action='store_true', help="use img_layernorm")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--optim", default='AdamW', type=str, help="AdamW or Adamax")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=-1, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")

    args = parser.parse_args()

    """ PASS """
    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)
    """ PASS """

    # Create or inspect output dir
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")

    """ PASS """
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    """ PASS """

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")  # device(type='cuda')
        # print(device)  # cuda
        args.n_gpu = torch.cuda.device_count()  # 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        """ PASS """
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
        """ PASS """
    args.device = device  # device(type='cuda')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    '''
    04/02/2022 13:08:47 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
    '''

    # Set seed
    set_seed(args.seed, args.n_gpu)  # 42 1

    # Prepare GLUE task
    # print(processors)
    '''
    {
        "vqa_text": VQATextProcessor,
        "vqa_text_a": VQATextAProcessor,
        "gqa": GQAProcessor,
        "nlvr": NLVRProcessor
    }
    '''
    # print(output_modes)
    '''
    {
        "vqa_text": "classification",
        "vqa_text_a": "classification",
        "gqa": "classification",
        "nlvr": "classification"
    }
    '''
    args.task_name = args.task_name.lower()  # gqa
    """ PASS """
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    """ PASS """
    processor = processors[args.task_name]()  # GQAProcessor
    args.output_mode = output_modes[args.task_name]  # "classification"
    label_list = processor.get_labels(args.label_file)  # datasets/gqa/trainval_testdev_all_ans2label.pkl
    # print(label_list)  # 长度为1853的list
    '''
    [0, 1, 2, 3, 4, ..., 1852]
    '''
    num_labels = len(label_list)  # 1853
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))  # gqa 1853
    '''
    04/02/2022 13:08:47 - INFO - __main__ - Task Name: gqa, #Labels: 1853
    '''

    """ PASS """
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    """ PASS """

    args.model_type = args.model_type.lower()  # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (BertConfig, ImageBertForSequenceClassification, BertTokenizer)
    config = config_class.from_pretrained(  # BertConfig
        args.config_name if args.config_name else args.model_name_or_path,  # pretrained_models/base-vg-labels/ep_107_1192087
        num_labels=num_labels,  # 1853
        finetuning_task=args.task_name,  # gqa
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,  # pretrained_models/base-vg-labels/ep_107_1192087
        do_lower_case=args.do_lower_case,  # True
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # new config: discrete code
    config.img_feature_dim = args.img_feature_dim  # 2054
    config.img_feature_type = args.img_feature_type  # faster_r-cnn
    config.code_voc = args.code_voc  # 512
    config.hidden_dropout_prob = args.drop_out  # 0.3
    config.loss_type = args.loss_type  # ce
    config.classifier = args.classifier  # linear
    config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # config.use_img_layernorm = args.use_img_layernorm

    """ PASS """
    # load discrete code
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Load discrete code from: {}'.format(args.data_dir))
        t_start = time.time()
        train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
        t_end = time.time()
        logger.info('Load time: %.3f' % (t_end - t_start))
        if args.code_level == 'top':
            config.code_dim = train_code['embeddings_t'].shape[0]
            config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
        elif args.code_level == 'bottom':
            config.code_dim = train_code['embeddings_b'].shape[0]
            config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
        elif args.code_level == 'both':
            config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]
    """ PASS """

    model = model_class.from_pretrained(  # ImageBertForSequenceClassification
        args.model_name_or_path,  # pretrained_models/base-vg-labels/ep_107_1192087
        from_tf=bool('.ckpt' in args.model_name_or_path),  # False
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    """ PASS """
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Initializing the code embedding with {}'.format(args.code_level))
        if args.code_level == 'top':
            model.init_code_embedding(train_code['embeddings_t'].t())
        elif args.code_level == 'bottom':
            model.init_code_embedding(train_code['embeddings_b'].t())
    """ PASS """

    """ PASS """
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    """ PASS """

    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Model Parameters: {}'.format(total_params))
    '''
    04/02/2022 13:08:50 - INFO - __main__ - Model Parameters: 112485437
    '''

    model.to(args.device)  # device(type='cuda')

    logger.info("Training/evaluation parameters %s", args)
    '''
    04/02/2022 12:57:28 - INFO - __main__ - Training/evaluation parameters Namespace(
        adam_epsilon=1e-08, 
        cache_dir='', classifier='linear', cls_hidden_scale=3, code_level='top', code_voc=512, config_name='', 
        data_dir='datasets/gqa', device=device(type='cuda'), do_eval=False, do_lower_case=True, do_test=False, do_test_dev=False, do_train=True, do_train_val=False, drop_out=0.3, 
        eval_all_checkpoints=False, eval_data_type='all', evaluate_during_training=True, 
        fp16=False, fp16_opt_level='O1', 
        gradient_accumulation_steps=1, 
        img_feature_dim=2054, img_feature_type='faster_r-cnn', 
        label2ans_file=None, label_file='datasets/gqa/trainval_testdev_all_ans2label.pkl', learning_rate=5e-05, local_rank=-1, logging_steps=-1, loss_type='ce', 
        max_grad_norm=1.0, max_img_seq_length=45, max_seq_length=165, max_steps=-1, model_name_or_path='pretrained_models/base-vg-labels/ep_107_1192087', model_type='bert', 
        n_gpu=1, no_cuda=False, num_train_epochs=5.0, 
        optim='AdamW', output_dir='model/gqa/teacher', output_mode='classification', overwrite_cache=False, overwrite_output_dir=False, 
        per_gpu_eval_batch_size=16, per_gpu_train_batch_size=16, philly=False, 
        save_after_epoch=-1, save_epoch=1, save_steps=-1, scheduler='linear', seed=88, server_ip='', server_port='', 
        task_name='gqa', test_data_type='all', tokenizer_name='', train_data_type='all', 
        warmup_steps=0, weight_decay=0.05, workers=4
    )
    '''

    # load image features
    img_features = _load_img_features(args)
    # print(img_features)  # 148854
    '''
    {
        ...,
        '2352065': tensor([[ 0.8760,  4.3490,  0.4122,  ...,  0.9983,  0.9449,  0.6555],
                          [ 0.5602,  0.0730,  2.6269,  ...,  0.9983,  0.4950,  0.7504],
                          [ 0.6406,  1.8985,  0.8664,  ...,  0.9552,  0.8789,  0.7888],
                          ...,
                          [ 0.0746, 10.0312,  1.2866,  ...,  0.9983,  0.3342,  0.4202],
                          [ 0.0000,  0.0000,  2.6000,  ...,  0.5594,  0.0615,  0.0673],
                          [ 0.0000,  0.0000,  4.0430,  ...,  0.5451,  0.0337,  0.0465]])
    }
    '''

    # Training (on 'train' set)
    if args.do_train:  # 构建训练集'train'
        train_dataset = GQADataset(args, 'train', tokenizer, img_features)  # 构建训练集'train'
        eval_dataset = GQADataset(args, 'val', tokenizer, img_features)  # 构建验证集'val'
        '''
        GQADataset(Dataset): 初始化参数为args, name, tokenizer, img_features
            self.args: args
            self.name: 'train'或'val'或'test'
            self.tokenizer: tokenizer

            self.output_mode: "classification"

            self.img_features:  {
                                    ...,
                                    '2352065': tensor([[ 0.8760,  4.3490,  0.4122,  ...,  0.9983,  0.9449,  0.6555],
                                                       [ 0.5602,  0.0730,  2.6269,  ...,  0.9983,  0.4950,  0.7504],
                                                       [ 0.6406,  1.8985,  0.8664,  ...,  0.9552,  0.8789,  0.7888],
                                                       ...,
                                                       [ 0.0746, 10.0312,  1.2866,  ...,  0.9983,  0.3342,  0.4202],
                                                       [ 0.0000,  0.0000,  2.6000,  ...,  0.5594,  0.0615,  0.0673],
                                                       [ 0.0000,  0.0000,  4.0430,  ...,  0.5451,  0.0337,  0.0465]])
                                }
            self.examples:  [
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
            self.labels: [0, 1, 2, 3, 4, ..., 1852]
            self.label_map: {
                                0: 0,
                                1: 1,
                                2: 2,
                                3: 3,
                                ...,
                                1852, 1852
                            }

            def __getitem__(index): # 对每一个self.examples[index]进行处理后得到:
                                    (
                                        input_ids: (128,),
                                        input_mask: (178,),
                                        segment_ids: (128,),
                                        label_id: (1,),
                                        img_feat: (50, 2054),
                                        q_id: (1,)
                                    )
            def __len__(): len(self.examples)
        '''

        # 在训练集'train'上做Training, 在验证集'val'上做Evaluation
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        '''
        04/02/2022 13:12:29 - INFO - __main__ -  global_step = 64, average loss = 4.378270532935858
        '''

    """ PASS """
    # Training (on 'train+val' set)
    if args.do_train_val:  # 构建训练集'train+val'
        train_dataset = GQADataset(args, 'train+val', tokenizer, img_features)  # 构建训练集'train+val'
        eval_dataset = GQADataset(args, 'val', tokenizer, img_features)  # 构建验证集'val'

        # 在训练集'train+val'上做Training, 在验证集'val'上做Evaluation
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    """ PASS """

    # Evaluation (on 'val' set)
    # results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        eval_dataset = GQADataset(args, 'val', tokenizer, img_features)  # 构建验证集'val'

        checkpoints = [args.output_dir]  # ["results/gqa/best"]
        """ PASS """
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        """ PASS """
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        '''
        04/02/2022 13:31:38 - INFO - __main__ - Evaluate the following checkpoints: ['model/gqa/teacher']
        '''
        for checkpoint in checkpoints:  # results/gqa/best
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result, score = evaluate(args, model, eval_dataset, prefix=global_step)
            # result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            # results.update(result)

    """ PASS """
    # Testing (on 'test-dev' set)
    if args.do_test_dev and args.local_rank in [-1, 0]:
        test_dev_dataset = GQADataset(args, 'test-dev', tokenizer, img_features)  # 构建测试集'test-dev'

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result, score = evaluate(args, model, test_dev_dataset, prefix=global_step)
            # test(args, model, test_dev_dataset, prefix=global_step)
    """ PASS """

    # Testing (on 'test' set)
    if args.do_test and args.local_rank in [-1, 0]:
        test_dataset = GQADataset(args, 'test', tokenizer, img_features)  # 构建测试集'test'

        checkpoints = [args.output_dir]  # ["results/gqa/best"]
        """ PASS """
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        """ PASS """
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        '''
        
        '''
        for checkpoint in checkpoints:  # "results/gqa/best"
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            test(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
