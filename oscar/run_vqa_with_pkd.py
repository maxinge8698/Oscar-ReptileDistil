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
import gc

import _pickle as cPickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')  # 新添加的目录会优先于其他目录被import检查

from transformers.pytorch_transformers import BertConfig, BertTokenizer
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


class VQADataset(Dataset):
    """ VQA Dataset """

    def __init__(self, args, name, tokenizer):  # args 'train'或'val'或'test2015' tokenizer
        super(VQADataset, self).__init__()

        assert name in ['train', 'val', 'test2015', 'test-dev2015', 'train+val']

        self.args = args
        self.name = name  # 'train'或'val'或'test2015'
        self.tokenizer = tokenizer

        self.output_mode = output_modes[args.task_name]  # "classification"

        # load image features
        self.img_features = _load_img_features(args, name)  # args, 'train'或'val'或'test2015'
        # print(self.img_features) # 121287 或 2000 或 81434
        '''
        {
            ...,
            9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                       [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                       [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                       ...,
                       [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                       [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                       [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),
            ...
        }
        '''

        self.examples, self.labels = _load_dataset(args, name)  # args 'train'或'val'或'test2015'
        # print(self.examples)  # 647480->634516 或 10631->10402 或 447793->447793
        '''
        [
            InputInstance(
                guid='train-0', 
                text_a='How many cookies can be seen?', 
                text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple', 
                label=[1504], 
                score=[1.0], 
                img_key=9, 
                q_id=0
            ), 
            ...
        ]
        '''
        # print(self.labels)  # 3129
        '''
        [0, 835, 2421, 1, 78, 2, 2379, ...]
        '''

        self.label_map = {label: i for i, label in enumerate(self.labels)}
        # print(self.label_map)
        '''
        {
            0: 0, 
            835: 1, 
            2421: 2, 
            1: 3, 
            78: 4, 
            2: 5,
            ...
        }
        '''

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))  # 'train'或'val'或'test2015' 644516或10402或447793
        '''
        03/19/2022 13:10:17 - INFO - __main__ - train Data Examples: 634516
        或
        03/19/2022 13:09:37 - INFO - __main__ - val Data Examples: 10402
        或
        03/14/2022 14:17:07 - INFO - __main__ - test2015 Data Examples: 447793
        '''

    def tensorize_example(self,
                          example,  # InputInstance(guid='train-0', text_a='How many cookies can be seen?', text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple', label=[1504], score=[1.0], img_key=9, q_id=0)
                          cls_token_at_end=False,  # False
                          pad_on_left=False,  # False
                          cls_token='[CLS]',  # '[CLS]'
                          sep_token='[SEP]',  # '[SEP]'
                          cls_token_segment_id=1,  # 0
                          pad_token_segment_id=0,  # 0
                          pad_token=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1,
                          mask_padding_with_zero=True):

        # print(example)
        """
        InputInstance(
            guid='train-0',
            text_a='How many cookies can be seen?',
            text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple',
            label=[1504],
            score=[1.0],
            img_key=9,
            q_id=0
        )
        """

        tokens_a = self.tokenizer.tokenize(example.text_a)  # ['how', 'many', 'cookies', 'can', 'be', 'seen', '?']

        tokens_b = None
        if example.text_b:  # 'bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple'
            tokens_b = self.tokenizer.tokenize(example.text_b)  # ['bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple']
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            """ PASS """
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]  # Account for [CLS] and [SEP] with "- 2"
            """ PASS """

        tokens = tokens_a + [sep_token]  # ['how', 'many', 'cookies', 'can', 'be', 'seen', '?', '[SEP]']
        segment_ids = [sequence_a_segment_id] * len(tokens)  # [0, 0, 0, 0, 0, 0, 0, 0]

        if tokens_b:  # ['bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple']
            tokens += tokens_b + [sep_token]  # ['how', 'many', 'cookies', 'can', 'be', 'seen', '?', '[SEP]', 'bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple', '[SEP]']
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if cls_token_at_end:
            """ PASS """
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            """ PASS """
        else:
            tokens = [cls_token] + tokens  # ['[CLS]', 'how', 'many', 'cookies', 'can', 'be', 'seen', '?', '[SEP]', 'bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple', '[SEP]']
            segment_ids = [cls_token_segment_id] + segment_ids  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # [101, 2129, 2116, 16324, 2064, 2022, 2464, 1029, 102, 4605, 22953, 21408, 3669, 4605, 4605, 4605, 15642, 4605, 9850, 4605, 2123, 4904, 9850, 4605, 7759, 2795, 6207, 102]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)  # 128 - 28 = 100
        if pad_on_left:
            """ PASS """
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            """ PASS """
        else:
            input_ids = input_ids + ([pad_token] * padding_length)  # [101, 2129, 2116, 16324, 2064, 2022, 2464, 1029, 100, 4605, 22953, 21408, 3669, 4605, 4605, 4605, 15642, 4605, 9850, 4605, 2123, 4904, 9850, 4605, 7759, 2795, 6207, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
            # print(self.img_features)  # 121287 或 2000 或 81434
            '''
            {
                ...,
                9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                           [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                           [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                           ...,
                           [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                           [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                           [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),
                ...
            }
            '''
            img_feat = self.img_features[example.img_key]  # 9
            # print(img_feat)  # (20, 2054)
            '''
            tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                    [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                    [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                    ...,
                    [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                    [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                    [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]])
            '''
            if img_feat.shape[0] > self.args.max_img_seq_length:  # 20 > 50
                img_feat = img_feat[0:self.args.max_img_seq_length, ]  # (50, 2054)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+50
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:  # 20 < 50
                if self.args.max_img_seq_length > 0:  # 50 > 0
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+20: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]  # 128+20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))  # (50-20, 2054)
                img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((20, 2054), (30, 2054), dim=0) -> (50, 2054)
                if self.args.max_img_seq_length > 0:  # 50 > 0
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])  # 148+30: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]  # 148+30: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print(len(input_ids), len(input_mask), len(segment_ids), img_feat.shape)  # 128 178 128 (50, 2054)

        if self.args.output_mode == "classification":  # "classification"
            if example.label is None:  # 测试集example.label=None, example.score=None
                label_id = [0]  # 测试集的label_id=[0]
                score = [0]  # 测试集的score=[0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:  # 训练集和验证集example.label=[1504], example.score=[1.0]
                label_id = [self.label_map[l] for l in example.label]  # [1504] -> [2698]
                score = example.score  # [1.0]
        elif self.args.output_mode == "regression":
            """ PASS """
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
            """ PASS """
        else:
            raise KeyError(self.args.output_mode)

        new_scores = target_tensor(len(self.labels), label_id, score)  # 3129 [2689] [1.0]
        # print(new_scores)
        '''
        [0, 0, 0, 0, 0, 0, ..., 1.0, ..., 0]
        '''

        """ PASS """
        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            # img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)
        """ PASS """

        return (torch.tensor(input_ids, dtype=torch.long),  # (128,)
                torch.tensor(input_mask, dtype=torch.long),  # (178,)
                torch.tensor(segment_ids, dtype=torch.long),  # (128,)
                torch.tensor([label_id[0]], dtype=torch.long),  # (1,): tensor([2698])
                torch.tensor(new_scores, dtype=torch.float),  # (3129,): tensor([0, 0, 0, 0, 0, 0, ..., 1.0, ..., 0])
                img_feat,  # (50, 2054)
                torch.tensor([example.q_id], dtype=torch.long))  # (1,): tensor([0])

    def __getitem__(self, index):
        # print(self.examples)
        """
        [
            InputInstance(
                guid='train-0',
                text_a='How many cookies can be seen?',
                text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple',
                label=[1504],
                score=[1.0],
                img_key=9,
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
            text_a='How many cookies can be seen?', 
            text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple', 
            label=[1504], 
            score=[1.0], 
            img_key=9, 
            q_id=0
        )
        '''
        example = self.tensorize_example(entry,  # InputInstance(guid='train-0', text_a='How many cookies can be seen?', text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple', label=[1504], score=[1.0], img_key=9, q_id=0)
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
            label_id: (1,)
            new_score: (3129,)
            img_feat: (50, 2054),
            q_id: (1,)
        )
        '''
        return example

    def __len__(self):
        return len(self.examples)


def _load_dataset(args, name):  # args 'train'或'val'或'test-2015'
    processor = processors[args.task_name]()  # VQATextProcessor
    labels = processor.get_labels(args.label_file)  # datasets/vqa/trainval_ans2label.pkl
    # print(labels)  # 长度为3129的list
    '''
    [0, 835, 2421, 1, 78, ..., 3128]
    '''

    if name == 'train':
        if args.data_label_type == 'mask':
            if args.use_vg:  # 使用train2014_vg_qla_mrcnn.json
                """ PASS """
                examples = processor.get_train_examples(args.data_dir, 'train2014_vg_qla_mrcnn.json')
                """ PASS """
            else:  # 使用train2014_qla_mrcnn.json
                examples = processor.get_train_examples(args.data_dir, 'train2014_qla_mrcnn.json')  # datasets/vqa/ train2014_qla_mrcnn.json
                # print(examples)  # 647480 -> 634516
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
        else:  # 使用train2014_qla.json
            """ PASS """
            examples = processor.get_train_examples(args.data_dir, 'train2014_qla.json')
            """ PASS """
    elif name == 'train+val':
        """ PASS """
        if args.data_label_type == 'mask':  # 使用train+val2014_qla_mrcnn.json
            examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla_mrcnn.json')  # 658111
        else:  # 使用train+val2014_qla.json
            examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla.json')
        """ PASS """
    elif name == 'val':
        if args.data_label_type == 'mask':
            if args.use_vg_dev:  # 使用vg_qla_mrcnn.json
                """ PASS """
                examples = processor.get_dev_examples(args.data_dir, 'vg_qla_mrcnn.json')
                """ PASS """
            else:  # 使用val2014_qla_mrcnn.json
                examples = processor.get_dev_examples(args.data_dir, 'val2014_qla_mrcnn.json')  # datasets/vqa/ val2014_qla_mrcnn.json
                # print(examples)  # 10631 -> 10402
                '''
                [
                    InputInstance(
                        guid="dev-0", 
                        text_a="What is he sitting on?", 
                        text_b="person person bottle cup person cup remote couch handbag couch frisbee couch person potted plant person",
                        label=[487, 2969, 2898]
                        score=[0.9, 0.6, 1.0], 
                        img_key=241, 
                        q_id=0
                    ),
                    ...
                ]
                '''
        else:  # 使用val2014_qla.json
            """ PASS """
            examples = processor.get_dev_examples(args.data_dir, 'val2014_qla.json')
            """ PASS """
    elif name == 'test2015':
        if args.data_label_type == 'mask':  # 使用test2015_qla_mrcnn.json
            examples = processor.get_test_examples(args.data_dir, 'test2015_qla_mrcnn.json')  # datasets/vqa test2015_qla_mrcnn.json
            # print(examples)  # 447793 -> 447793
            """
            [
                InputInstance(
                    guid="test-0", 
                    text_a="Is the ball flying towards the batter?", 
                    text_b="helmet bat line pant grass umpire shirt dirt shirt glove mound uniform pant ball field baseball pant net shirt advertisement home plate base player pitcher player cap field spectator glove leg uniform wall glove sign man leg man glove jersey sign plate dirt player hair man batter man pitchers mound person grass person",
                    label=None
                    score=None 
                    img_key=262144, 
                    q_id=262144000
                ),
                ...
            ]
            """
        else:  # 使用test2014_qla.json
            """ PASS """
            examples = processor.get_test_examples(args.data_dir, 'test2014_qla.json')
            """ PASS """
    elif name == 'test-dev2015':
        """ PASS """
        if args.data_label_type == 'mask':  # 使用test-dev2015_qla_mrcnn.json
            examples = processor.get_test_examples(args.data_dir, 'test-dev2015_qla_mrcnn.json')
        else:  # 使用test2014_qla.json
            examples = processor.get_test_examples(args.data_dir, 'test2014_qla.json')
        """ PASS """

    return examples, labels


def _load_img_features(args, name):
    t_start = time.time()
    if args.img_feature_type == 'faster_r-cnn':  # faster_r-cnn
        if args.img_feature_dim == 2048:  # object features: 2048
            """ PASS """
            feat_file_name = '{}_img_frcnn_obj_feats.pt'.format(name)
            """ PASS """
        else:  # object + spatial features: 2054
            feat_file_name = '{}_img_frcnn_feats.pt'.format(name)  # train_img_frcnn_feats.pt或val_img_frcnn_feats.pt或test_img_frcnn_feats.pt
    else:
        """ PASS """
        feat_file_name = '{}_img_feats.pt'.format(name)
        """ PASS """
    img_features = torch.load(os.path.join(args.data_dir, feat_file_name))  # datasets/vqa/train_img_frcnn_feats.pt或datasets/vqa/val_img_frcnn_feats.pt或datasets/vqa/test_img_frcnn_feats.pt
    # print(img_features)  # 121287 或 2000 或 81434
    '''
    {
        ...,
        9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                   [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                   [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                   ...,
                   [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                   [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                   [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),
        ...
    }
    '''
    t_end = time.time()
    logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))
    '''
    03/19/2022 13:09:37 - INFO - __main__ - Info: loading val_img_frcnn_feats.pt features using 0.24 secs
    或
    03/19/2022 13:10:10 - INFO - __main__ - Info: loading train_img_frcnn_feats.pt features using 32.80 secs
    或
    03/19/2022 14:42:02 - INFO - __main__ - Info: loading test2015_img_frcnn_feats.pt features using 16.02 secs
    '''
    return img_features


def target_tensor(len, labels, scores):  # 3129 [2698] [1.0]
    """ create the target by labels and scores """
    target = [0] * len  # [0, 0, 0, 0, 0, 0, ..., 0, ...]
    for id, l in enumerate(labels):
        target[l] = scores[id]  # target[2698] = score[0]
    # print(target)
    '''
    [0, 0, 0, 0, 0, 0, ..., 1.0, ...]
    '''
    return target


def compute_score_with_logits(logits,  # logits: (16, 3129)
                              labels):  # new_scores: (16, 3129)
    logits = torch.max(logits, dim=1)[1].data  # argmax: (16,)
    one_hots = torch.zeros(*labels.size()).cuda()  # (16, 3129)
    one_hots.scatter_(dim=1, index=logits.view(-1, 1), value=1)  # (16, 3129): [index]位置处的值被[value]指定的值替代
    scores = (one_hots * labels)  # (16, 3129)*(16, 3129)=(16, 3129)
    return scores


def train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer):
    """ Train the model """

    # print(train_dataset)  # 647480过滤后剩634516
    '''
    VQADataset(Dataset): 初始化参数为args, name, tokenizer
        self.args: args
        self.name: 'train'或'val'或'test2015'
        self.tokenizer: tokenizer
        
        self.output_mode: "classification"
        
        self.img_features: {
                                ...,
                                9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                                           [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                                           [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                                           ...,
                                           [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                                           [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                                           [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),
                                ...
                            }
        self.examples:  [
                            InputInstance(
                                guid='train-0', 
                                text_a='How many cookies can be seen?', 
                                text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple', 
                                label='[1504]', 
                                score='[1.0]', 
                                img_key=9, 
                                q_id=0
                            ), 
                            ...
                        ]
        self.labels: [0, 835, 2421, 1, 78, 2, ...]
        self.label_map: {
                            0: 0,
                            835: 1,
                            2421: 2,
                            1: 3,
                            78: 4,
                            2: 5,
                            ...
                        }
            
        def __getitem__(index): # 对每一个self.examples[index]进行处理后得到:
                                (
                                    input_ids: (128,),
                                    input_mask: (178,),
                                    segment_ids: (128,),
                                    label_id: (1,)
                                    new_score: (3129,)
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
    # print(train_dataloader)  # 634516/16=39658
    '''
    [
        [
            torch.Size([16, 128]),
            torch.Size([16, 178]),
            torch.Size([16, 128]),
            torch.Size([16, 1]),
            torch.Size([16, 3129]),
            torch.Size([16, 50, 2054]),
            torch.Size([16, 1]) 
        ],
        [
            torch.Size([16, 128]),
            torch.Size([16, 178]),
            torch.Size([16, 128]),
            torch.Size([16, 1]),
            torch.Size([16, 3129]),
            torch.Size([16, 50, 2054]),
            torch.Size([16, 1])
        ],
        ...
    ]
    '''
    if args.max_steps > 0:  # args.max_steps=-1
        """ PASS """
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        """ PASS """
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs  # 39658 // 1 * 25.0 = 991450.0

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
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

    # """ PASS """
    # # apex fp16 initialization
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # """ PASS """
    #
    # """ PASS """
    # # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:  # args.n_gpu=1
    #     model = torch.nn.DataParallel(model)
    # """ PASS """
    #
    # """ PASS """
    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # """ PASS """

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))  # 634516
    logger.info("  Num Epochs = %d", args.num_train_epochs)  # 25
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)  # 16
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))  # 16 * 1 * 1
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)  # 1
    logger.info("  Total optimization steps = %d", t_total)  # 991450
    '''
    03/19/2022 13:10:17 - INFO - __main__ - ***** Running training *****
    03/19/2022 13:10:17 - INFO - __main__ -   Num examples = 634516
    03/19/2022 13:10:17 - INFO - __main__ -   Num Epochs = 25
    03/19/2022 13:10:17 - INFO - __main__ -   Instantaneous batch size per GPU = 16
    03/19/2022 13:10:17 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 16
    03/19/2022 13:10:17 - INFO - __main__ -   Gradient Accumulation steps = 1
    03/19/2022 13:10:17 - INFO - __main__ -   Total optimization steps = 991450
    '''

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    student_model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])  # 0~24

    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': student_model.state_dict(),  # copy.deepcopy(student_model)
        'optimizer_state': optimizer.state_dict()
    }

    # for epoch in range(int(args.num_train_epochs)):
    for epoch in train_iterator:  # 0~24

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])  # 0~39657

        # total_loss = 0
        # train_score = 0
        # total_norm = 0
        # count_norm = 0

        t_start = time.time()

        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(epoch_iterator):  # 0~39657
            # print(batch)
            '''
            [
                torch.Size([16, 128])  # input_ids
                torch.Size([16, 178])  # input_mask
                torch.Size([16, 128])  # segment_ids
                torch.Size([16, 1])  # label_id
                torch.Size([16, 3129])  # new_scores
                torch.Size([16, 50, 2054])  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],  # input_ids: (16, 128)
                'attention_mask': batch[1],  # input_mask: (16, 178)
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 128)
                'labels': batch[4],  # new_scores: (16, 3129)
                'img_feats': None if args.img_feature_dim == -1 else batch[5]  # img_feat: (16, 50, 2054)
            }

            # BERT-PKD
            student_model.train()
            teacher_model.eval()

            task_specific_loss, student_logits, student_reps, student_atts = student_model(input_ids=inputs['input_ids'],  # (16, 128)
                                                                                           token_type_ids=inputs['token_type_ids'],  # (16, 128)
                                                                                           attention_mask=inputs['attention_mask'],  # (16, 178)
                                                                                           labels=inputs['labels'],  # (16, 3129)
                                                                                           img_feats=inputs['img_feats'])  # 新增: (16, 50, 2054)
            # print(task_specific_loss)  # torch(数)
            # print(student_logits)  # (16, 3129)
            # print(student_reps)  # tuple(7个(16, 178, 768))或tuple(5个(16, 178, 768))
            # print(student_atts)  # tuple(6个(16, 12, 178, 178))或tuple(4个(16, 12, 178, 178))

            with torch.no_grad():
                teacher_logits, teacher_reps, teacher_atts = teacher_model(input_ids=inputs['input_ids'],  # (16, 128)
                                                                           token_type_ids=inputs['token_type_ids'],  # (16, 128)
                                                                           attention_mask=inputs['attention_mask'],  # (16, 178))
                                                                           img_feats=inputs['img_feats'])  # 新增: (16, 50, 2054)
            # print(teacher_logits)  # (16, 3129)
            # print(teacher_reps)  # tuple(13个(16, 178, 768))
            # print(teacher_atts)  # tuple(12个(16, 12, 178, 178))

            # PKD-skip
            teacher_layer_num = len(teacher_atts)  # 12
            student_layer_num = len(student_atts)  # 6或4
            assert teacher_layer_num % student_layer_num == 0  # 12%6=0或12%4=0
            layers_per_block = int(teacher_layer_num / student_layer_num)  # int(12/6)=2或int(12/4)=3

            teacher_cls_outputs = [
                teacher_reps[i * layers_per_block][:, 0] for i in range(student_layer_num + 1)  # 0,2,4,6,8,10,12: hidden_states的第1,3,5,7,9,11,13层(即embedding层+transformer的第2,4,6,8,10,12层) 或 0,3,6,9,12: hidden_states的第1,4,7,10,13层(即embedding层+transformer的第3,6,9,12层)
            ]  # list(7个(16, 768))或list(5个(16, 768))
            student_cls_outputs = [student_reps[i][:, 0] for i in range(len(student_reps))]  # list(7个(16, 768))或list(5个(16, 768))

            teacher_patience = torch.stack(teacher_cls_outputs[:-1]).transpose(0, 1)  # list(7个(16, 768))或list(5个(16, 768)) -> (6, 16, 768)或(4, 16, 768) -> (16, 6, 768)或(16, 4, 768)
            student_patience = torch.stack(student_cls_outputs[:-1]).transpose(0, 1)  # list(7个(16, 768))或list(5个(16, 768)) -> (6, 16, 768)或(4, 16, 768) -> (16, 6, 768)或(16, 4, 768)

            teacher_patience = F.normalize(teacher_patience, p=2, dim=-1)  # (16, 6, 768)或(16, 4, 768)
            student_patience = F.normalize(student_patience, p=2, dim=-1)  # (16, 6, 768)或(16, 4, 768)

            # L_pkd
            pkd_loss = F.mse_loss(teacher_patience, student_patience).half()  # torch(数)

            # L_vanilla_kd
            T = args.temperature
            vanilla_kd_loss = nn.KLDivLoss()(F.log_softmax(student_logits / T, dim=-1),
                                             F.softmax(teacher_logits / T, dim=-1)) * T * T
            # L_task_specific
            # task_specific_loss = nn.BCEWithLogitsLoss()(student_logits, inputs['labels'])

            # L_s = α * L_vanilla_kd + (1 - α) * L_task_specific + β * L_pkd
            loss = (1 - args.alpha) * task_specific_loss + args.alpha * vanilla_kd_loss + args.beta * pkd_loss

            loss.backward()  # ▽_θ_s L(x;θ_s;θ_t)

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 1.0

            # # print(compute_score_with_logits(logits), batch[4])  # (16, 3129)
            # batch_score = compute_score_with_logits(student_logits, batch[4]).sum()  # torch(数)
            # train_score += batch_score.item()

            tr_loss += loss.item()
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
            break

        logger.info("***** Epoch: {} *****".format(epoch + 1))
        logger.info("  Train Loss: {}".format(tr_loss / len(train_dataset)))
        '''
        
        '''

        t_end = time.time()
        logger.info('  Train Time Cost: %.3f' % (t_end - t_start))
        '''

        '''

        # 每个epoch结束时做一次evaluation并保存模型
        # evaluation
        eval_result, eval_score = evaluate(args, student_model, eval_dataset, prefix='')
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(student_model)
            # best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())
        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch > 0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            base_output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))  # ./model/vqa/student/checkpoint-1
            # teacher_output_dir = os.path.join(base_output_dir, 'teacher')  # ./model/vqa/student/chechpoint-1/teacher
            student_output_dir = os.path.join(base_output_dir, 'student')  # ./model/vqa/student/chechpoint-1/student
            # for output_dir in [teacher_output_dir, student_output_dir]:
            for output_dir in [student_output_dir]:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            # model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training
            student_model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Take care of distributed/parallel training
            student_model_to_save.save_pretrained(student_output_dir)
            torch.save(args, os.path.join(student_output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(student_output_dir)
            logger.info("Saving student model checkpoint {0} to {1}".format(epoch + 1, student_output_dir))
            '''
            03/10/2022 11:22:24 - INFO - __main__ - Saving model checkpoint 0 to results/vqa/checkpoint-0
            '''
            # teacher_model_to_save = teacher_model.module if hasattr(teacher_model, 'module') else teacher_model  # Take care of distributed/parallel training
            # teacher_model_to_save.save_pretrained(teacher_output_dir)
            # torch.save(args, os.path.join(teacher_output_dir, 'training_args.bin'))
            # tokenizer.save_pretrained(teacher_output_dir)
            # logger.info("Saving teacher model checkpoint {0} to {1}".format(epoch + 1, teacher_output_dir))
            # '''
            #
            # '''

        epoch_log = {'epoch': epoch + 1, 'eval_score': eval_score, 'best_score': best_score}
        log_json.append(epoch_log)

        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch + 1, t_end - t_start))
        logger.info('********************')
        '''
        
        '''

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # 所有epoch结束后保存最好的模型
    if args.local_rank in [-1, 0]:  # Save the final model checkpoint
        # output_dir = os.path.join(args.output_dir, 'best'.format(best_model['epoch']))  # model/vqa/student/best
        output_dir = args.output_dir  # model/vqa/student
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(student_model, 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))
        '''
        
        '''

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    eval_task_names = (args.task_name,)  # ('vqa_text',)
    eval_outputs_dirs = (args.output_dir,)  # ('results/vqa',)

    results = {}
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # 'vqa_text' 'results/vqa'
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 16 * 1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     # num_workers=args.workers,  # 默认为0
                                     sampler=eval_sampler,  # SequentialSampler(eval_dataset)
                                     batch_size=args.eval_batch_size)  # 16
        # print(eval_dataloader)  # 10402/16=651
        '''
        [
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 3129])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ],
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 3129])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ],
            ...
        ]
        '''

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)  # debug: single-gpu or multi-gpus

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))  # 10402
        logger.info("  Batch size = %d", args.eval_batch_size)  # 16
        '''
        03/10/2022 09:20:39 - INFO - __main__ - ***** Running evaluation  *****
        03/10/2022 09:20:39 - INFO - __main__ -   Num examples = 10402
        03/10/2022 09:20:39 - INFO - __main__ -   Batch size = 16
        '''

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # num_data = 0
        score = 0
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
                torch.Size([16, 3129])  # new_scores
                torch.Size([16, 50, 2054])  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''
            # print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape, batch[5].shape, batch[6].shape)

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],  # input_ids: (16, 128)
                    'attention_mask': batch[1],  # input_mask: (16, 178)
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 128)
                    'labels': batch[4],  # new_scores: (16, 3129)
                    'img_feats': None if args.img_feature_dim == -1 else batch[5]  # img_feat: (16, 50, 2054)
                }
                outputs = model(**inputs)
                # print(outputs)
                '''
                (
                    loss: torch(数)
                    logits: (16, 3129),
                    hidden_states: tuple(13个(16, 178, 768)),
                    attentions: tuple(12个(16, 12, 178, 178))
                )
                '''
                tmp_eval_loss, logits = outputs[:2]
                # print(tmp_eval_loss, logits.shape)  # tensor(数) (16, 3129)

                eval_loss += tmp_eval_loss.mean().item()

                # print(compute_score_with_logits(logits, batch[4]))  # (16, 3129)
                # batch_score = compute_score_with_logits(logits, batch[4]).sum()  # torch(数)
                batch_score = torch.sum(compute_score_with_logits(logits, batch[4]), dim=1)  # (16,)
                # # update results_dict
                # # print(batch[6].view(-1).tolist())  # q_id: (16, 1) -> (16,) -> 长度为16的list
                # # print(batch_score.tolist())  # (16,) -> 长度为16的list
                # results_dict.update(
                #     {
                #         qa_ind: score for qa_ind, score in zip(batch[6].view(-1).tolist(), batch_score.tolist())
                #     }
                # )
                score += batch_score.sum().item()

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

        score = score / len(eval_dataloader.dataset)
        logger.info("  Eval Score: {}".format(100 * score))
        logger.info("  EVALERR: {:.2f}%".format(100 * score))
        '''
        
        '''
        results.update({'score': score})

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
    03/10/2022 09:35:04 - INFO - __main__ -   Eval Time Cost: 55.993
    '''

    return results, score


def test(args, model, eval_dataset=None, prefix=""):
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    eval_task_names = (args.task_name,)  # ('vqa_text',)
    eval_outputs_dirs = (args.output_dir,)  # ('results/vqa',)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))  # datasets/vqa/trainval_label2ans.pkl
    # print(label2ans)  # 长度为3129的list
    '''
    ['', 'name', 'plain', 'museum', 'steeple', 'grape', ...]
    '''
    logger.info('label2ans: %d' % (len(label2ans)))  # 3129
    '''
    03/19/2022 14:42:07 - INFO - __main__ - label2ans: 3129
    '''

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # vqa_text results/vqa
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 16 * 1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,  # SequentialSampler(eval_dataset)
                                     batch_size=args.eval_batch_size)  # 16
        # print(eval_dataloader)  # 447793/16=28988
        '''
        [
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 3129])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ],
            [
                torch.Size([16, 128])
                torch.Size([16, 178])
                torch.Size([16, 128])
                torch.Size([16, 1])
                torch.Size([16, 3129])
                torch.Size([16, 50, 2054])
                torch.Size([16, 1]) 
            ],
            ...
        ]
        '''

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Test!
        logger.info("***** Running test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))  # 447793
        logger.info("  Batch size = %d", args.eval_batch_size)  # 16
        '''
        03/19/2022 14:42:07 - INFO - __main__ - ***** Running test  *****
        03/19/2022 14:42:07 - INFO - __main__ -   Num examples = 447793
        03/19/2022 14:42:07 - INFO - __main__ -   Batch size = 16
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
                torch.Size([16, 3129])  # new_scores
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
                    'img_feats': None if args.img_feature_dim == -1 else batch[5]  # img_feat: (16, 50, 2054)
                }
                outputs = model(**inputs)
                # print(outputs)
                '''
                (
                    logits: (16, 3129),
                    hidden_states: tuple(13个(16, 178, 768)),
                    attentions: tuple(12个(16, 12, 178, 178))
                )
                '''

                logits = outputs[0]  # (16, 3129)

                val, idx = logits.max(dim=1)  # (values: (16,), indices: (16,))
                # logger.info('idx: %s, batch[6]: %s' % (str(idx.shape), str(batch[6].shape)))  # batch[6]为q_id: (16, 1)

                for i in range(idx.size(0)):  # idx.size(0)=16
                    # print(eval_dataset.labels)
                    '''
                    [0, 835, 2421, 1, 78, 2, ...]
                    '''

                    """ [{"question_id": int, "answer": str}, ...] """
                    result = {
                        'question_id': batch[6][i].item(),  # q_id
                        'answer': label2ans[eval_dataset.labels[idx[i].item()]]
                    }
                    results.append(result)

                    # if len(results) % 2000 == 0:
                    #     logger.info("PROGRESS: {}%".format(round(100 * len(results) / len(eval_dataset), 4)))
                    #     '''
                    #     3/19/2022 14:42:22 - INFO - __main__ - PROGRESS: 0.4466%
                    #     '''

        with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:  # model/vqa/teacher/test2015_results.json
            json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))
    '''
    03/14/2022 15:00:39 - INFO - __main__ - # questions: 447793
    03/14/2022 15:00:39 - INFO - __main__ - Test Time Cost: 2610.081
    '''


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    # Text
    parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")
    parser.add_argument("--data_label_type", default='faster', type=str, help="faster or mask")
    parser.add_argument("--use_vg", action='store_true', help="Use VG-QA or not.")
    parser.add_argument("--use_vg_dev", action='store_true', help="Use VG-QA as validation.")

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
    parser.add_argument("--loss_type", default='bce', type=str, help="kl or bce or ce")
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

    #
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, required=True, help="The student model dir.")
    parser.add_argument('--alpha', default=0.5, type=float, help="Vanilla knowledge distillation loss radio.")
    parser.add_argument("--temperature", default=5.0, type=float, help="Distillation temperature for soft target.")
    parser.add_argument('--num_hidden_layers', default=6, type=int)
    # parser.add_argument("--teacher_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam of Teacher model.")
    # parser.add_argument("--student_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam of Student model.")
    # parser.add_argument("--strategy", default="first", type=str, help="first | last | skip | both")
    parser.add_argument('--beta', default=500, type=float, help="patience loss radio.")

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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    '''
    03/19/2022 13:09:34 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
    '''

    # Set seed
    set_seed(args.seed, args.n_gpu)  # 42  1

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
    args.task_name = args.task_name.lower()  # vqa_text
    """ PASS """
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    """ PASS """
    processor = processors[args.task_name]()  # VQATextProcessor
    args.output_mode = output_modes[args.task_name]  # "classification"
    label_list = processor.get_labels(args.label_file)  # datasets/vqa/trainval_ans2label.pkl
    # print(label_list)  # 长度为3129的list
    '''
    [0, 835, 2421, 1, 78, 2, 2379, ...]
    '''
    num_labels = len(label_list)  # 3129
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))  # vqa_text 3129
    '''
    03/19/2022 13:09:34 - INFO - __main__ - Task Name: vqa_text, #Labels: 3129
    '''

    """ PASS """
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    """ PASS """

    args.model_type = args.model_type.lower()  # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (BertConfig, ImageBertForSequenceClassification, BertTokenizer)
    tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
        args.teacher_model,  # model/vqa/teacher
        do_lower_case=args.do_lower_case  # True
    )
    teacher_config = config_class.from_pretrained(  # BertConfig
        args.teacher_model,  # model/vqa/teacher
        num_labels=num_labels,  # 3129
        finetuning_task=args.task_name,  # vqa_text
    )
    student_config = config_class.from_pretrained(  # BertConfig
        args.student_model,  # pretrained_models/base-vg-labels/ep_107_1192087
        num_hidden_layers=args.num_hidden_layers,  # 6
        num_labels=num_labels,  # 3129
        finetuning_task=args.task_name,  # vqa_text
    )

    # new config: discrete code
    teacher_config.img_feature_dim = args.img_feature_dim  # 2054
    teacher_config.img_feature_type = args.img_feature_type  # faster_r-cnn
    teacher_config.code_voc = args.code_voc  # 512
    teacher_config.hidden_dropout_prob = args.drop_out  # 0.3
    teacher_config.loss_type = args.loss_type  # bce
    teacher_config.classifier = args.classifier  # linear
    teacher_config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # teacher_config.use_img_layernorm = args.use_img_layernorm
    # 输出hidden_states和attentions
    teacher_config.output_hidden_states = True
    teacher_config.output_attentions = True

    student_config.img_feature_dim = args.img_feature_dim  # 2054
    student_config.img_feature_type = args.img_feature_type  # faster_r-cnn
    student_config.code_voc = args.code_voc  # 512
    student_config.hidden_dropout_prob = args.drop_out  # 0.3
    student_config.loss_type = args.loss_type  # bce
    student_config.classifier = args.classifier  # linear
    student_config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # student_config.use_img_layernorm = args.use_img_layernorm
    # 输出hidden_states和attentions
    student_config.output_hidden_states = True
    student_config.output_attentions = True

    # """ PASS """
    # # load discrete code
    # if args.img_feature_type in ['dis_code', 'dis_code_t']:
    #     logger.info('Load discrete code from: {}'.format(args.data_dir))
    #     t_start = time.time()
    #     train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
    #     t_end = time.time()
    #     logger.info('Load time: %.3f' % (t_end - t_start))
    #     if args.code_level == 'top':
    #         config.code_dim = train_code['embeddings_t'].shape[0]
    #         config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
    #     elif args.code_level == 'bottom':
    #         config.code_dim = train_code['embeddings_b'].shape[0]
    #         config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
    #     elif args.code_level == 'both':
    #         config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]
    # """ PASS """

    teacher_model = model_class.from_pretrained(  # ImageBertForSequenceClassification
        args.teacher_model,  # model/vqa/teacher
        from_tf=bool('.ckpt' in args.teacher_model),  # False
        config=teacher_config
    )
    student_model = model_class.from_pretrained(  # ImageBertForSequenceClassification
        args.student_model,  # pretrained_models/base-vg-labels/ep_107_1192087,
        from_tf=bool('.ckpt' in args.student_model),  # False
        config=student_config
    )
    # """ PASS """
    # if args.img_feature_type in ['dis_code', 'dis_code_t']:
    #     logger.info('Initializing the code embedding with {}'.format(args.code_level))
    #     if args.code_level == 'top':
    #         model.init_code_embedding(train_code['embeddings_t'].t())
    #     elif args.code_level == 'bottom':
    #         model.init_code_embedding(train_code['embeddings_b'].t())
    # """ PASS """

    """ PASS """
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    """ PASS """

    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info('Teacher Model Parameters: {}'.format(teacher_total_params))
    student_total_params = sum(p.numel() for p in student_model.parameters())
    logger.info('Student Model Parameters: {}'.format(student_total_params))
    '''
    03/30/2022 22:12:45 - INFO - __main__ - Teacher Model Parameters: 113466681
    03/30/2022 22:12:45 - INFO - __main__ - Student Model Parameters: 70939449
    '''

    # ###########
    # for n, p in teacher_model.named_parameters():
    #     print(n)
    # for n, p in student_model.named_parameters():
    #     print(n)
    # ###########

    teacher_model.to(args.device)  # device(type='cuda')
    student_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    '''
    
    '''

    # Training (on 'train' set)
    if args.do_train:
        train_dataset = VQADataset(args, 'train', tokenizer)  # 构建训练集'train'
        eval_dataset = VQADataset(args, 'val', tokenizer)  # 构建验证集'val'
        '''
        VQADataset(Dataset): 初始化参数为args, name, tokenizer
            self.args: args
            self.name: 'train'或'val'或'test2015'
            self.tokenizer: tokenizer
            
            self.output_mode: "classification"
            
            self.img_features: {
                                    ...,
                                    9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                                               [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                                               [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                                               ...,
                                               [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                                               [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                                               [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),
                                    ...
                                }
            self.examples:  [
                                InputInstance(
                                    guid='train-0', 
                                    text_a='How many cookies can be seen?', 
                                    text_b='bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple', 
                                    label='[1504]', 
                                    score='[1.0]', 
                                    img_key=9, 
                                    q_id=0
                                ), 
                                ...
                            ]
            self.labels: [0, 835, 2421, 1, 78, 2, ...]
            self.label_map: {
                                0: 0,
                                835: 1,
                                2421: 2,
                                1: 3,
                                78: 4,
                                2: 5,
                                ...
                            }
            
            def __getitem__(index): # 对每一个self.examples[index]进行处理后得到:
                                    (
                                        input_ids: (128,),
                                        input_mask: (178,),
                                        segment_ids: (128,),
                                        label_id: (1,)
                                        new_score: (3129,)
                                        img_feat: (50, 2054),
                                        q_id: (1,)
                                    )
            def __len__(): len(self.examples)
        '''

        # 在训练集'train'上做Training, 在验证集'val'上做Evaluation
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        '''
        
        '''

    """ PASS """
    # Training (on 'train+val' set)
    if args.do_train_val:
        train_dataset = VQADataset(args, 'train+val', tokenizer)  # 构建训练集'train+val'
        eval_dataset = VQADataset(args, 'val', tokenizer)  # 构建验证集'val'

        # 在训练集'train+val'上做Training, 在验证集'val'上做Evaluation
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    """ PASS """

    # # Evaluation (on 'val' set)
    # # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     eval_dataset = VQADataset(args, 'val', tokenizer)  # 构建验证集'val'
    #
    #     checkpoints = [args.output_dir]  # ["results/vqa/best"]
    #     """ PASS """
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     """ PASS """
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     '''
    #
    #     '''
    #     for checkpoint in checkpoints:  # results/vqa/best
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint, config=config)
    #         model.to(args.device)
    #         result, score = evaluate(args, model, eval_dataset, prefix=global_step)
    #         # result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         # results.update(result)

    # """ PASS """
    # # Testing (on 'test-dev2015' set)
    # if args.do_test_dev and args.local_rank in [-1, 0]:
    #     test_dev_dataset = VQADataset(args, 'test-dev2015', tokenizer)  # 构建测试集'test-dev2015'
    #
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:  # results
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         test(args, model, test_dev_dataset, prefix=global_step)
    # """ PASS """

    # # Testing (on 'test2015' set)
    # if args.do_test and args.local_rank in [-1, 0]:
    #     test_dataset = VQADataset(args, 'test2015', tokenizer)  # 构建测试集'test2015'
    #
    #     checkpoints = [args.output_dir]  # ["results/vqa/best"]
    #     """ PASS """
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     """ PASS """
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     '''
    #
    #     '''
    #     for checkpoint in checkpoints:  # "results/vqa/best"
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         test(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
