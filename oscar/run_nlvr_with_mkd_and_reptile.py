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

from oscar.modeling.modeling_bert import ImageBertForMultipleChoice, ImageBertForSequenceClassification
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


class NLVRDataset(Dataset):
    """ NLVR2 Dataset """

    def __init__(self, args, name, tokenizer, img_features):  # args 'train'或'val'或'test1' tokenizer img_features
        super(NLVRDataset, self).__init__()

        assert name in ['train', 'val', 'test1', 'val+test1']

        self.args = args
        self.name = name  # 'train'或'val'或'test1'
        self.tokenizer = tokenizer

        self.output_mode = output_modes[args.task_name]  # "classification"

        # load image features
        self.img_features = img_features
        # print(self.img_features)  # 119354
        '''
        {
            ...,
            'nlvr2_test1-999-3-img0': tensor([[0.2093, 0.1726, 0.0000,  ..., 0.8653, 0.3037, 0.5747],
                                              [0.0487, 0.0000, 0.0314,  ..., 0.9089, 0.9983, 0.8539],
                                              [0.7593, 0.0000, 0.0000,  ..., 0.4331, 0.2360, 0.1643],
                                              ...,
                                              [0.0000, 0.0000, 0.0000,  ..., 0.7089, 0.2960, 0.1254],
                                              [0.0000, 0.0000, 0.0000,  ..., 0.8396, 0.2864, 0.1631],
                                              [0.0000, 0.0000, 0.0000,  ..., 0.4790, 0.2997, 0.1500]]), 
            'nlvr2_test1-999-3-img1': tensor([[0.6039, 2.2122, 0.0000,  ..., 0.7338, 0.4896, 0.4970],
                                              [0.0000, 2.9828, 0.0000,  ..., 0.7337, 0.2934, 0.5637],
                                              [0.0000, 0.0000, 0.0000,  ..., 0.7202, 0.2274, 0.0714],
                                              ...,
                                              [0.0000, 0.0000, 0.0000,  ..., 0.3436, 0.1675, 0.1095],
                                              [1.4307, 5.4042, 0.0000,  ..., 0.7686, 0.2727, 0.5442],
                                              [0.0000, 0.3192, 0.0000,  ..., 0.6729, 0.2998, 0.2141]])
        }
        '''

        self.examples, self.labels = _load_dataset(args, name)  # args 'train'或'val'或'test1'
        # print(self.examples)  # 86373 或 6982 或 6967
        '''
        [
            InputInstance(
                guid="train-0", 
                text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", 
                text_b={
                    "left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", 
                    "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"
                }, 
                label=0, 
                score=None, 
                img_key={
                    "left": "nlvr2_train-10171-0-img0", 
                    "right": "nlvr2_train-10171-0-img1"
                }, 
                q_id=0
            ),
            ...
        ]
        '''
        # print(self.labels)  # 2
        '''
        [0, 1]
        '''

        self.label_map = {label: i for i, label in enumerate(self.labels)}
        # print(self.label_map)
        '''
        {
            0: 0,
            1: 1
        }
        '''

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))  # 'train'或'val'或'test1' 86373或6982或6967
        '''
        03/27/2022 19:59:27 - INFO - __main__ - train Data Examples: 86373
        或
        03/27/2022 19:59:26 - INFO - __main__ - val Data Examples: 6982
        或
        03/15/2022 16:54:00 - INFO - __main__ - test1 Data Examples: 6967
        '''

    def tensorize_example(self,
                          example,  # InputInstance(guid="train-0", text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", text_b={"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}, label=0, score=None, img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, q_id=0)
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
            guid="train-0",
            text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.",
            text_b={
                "left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket",
                "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"
            },
            label=0,
            score=None,
            img_key={
                "left": "nlvr2_train-10171-0-img0",
                "right": "nlvr2_train-10171-0-img1"
            },
            q_id=0
        )
        """

        tokens_a = self.tokenizer.tokenize(example.text_a)  # ['an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.']

        tokens_b = None
        if example.text_b:  # {"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}
            text_b = example.text_b['left'] + ' ' + example.text_b['right']  # 'bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket book background case umbrella umbrella wallet case screw case wallet screw wallet wallet'
            tokens_b = self.tokenizer.tokenize(text_b)  # ['bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket', 'book', 'background', 'case', 'umbrella', 'umbrella', 'wallet', 'case', 'screw', 'case', 'wallet', 'screw', 'wallet', 'wallet']
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            """ PASS """
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]  # Account for [CLS] and [SEP] with "- 2"
            """ PASS """

        tokens = tokens_a + [sep_token]  # ['an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.', '[SEP]']
        segment_ids = [sequence_a_segment_id] * len(tokens)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if tokens_b:  # ['bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket', 'book', 'background', 'case', 'umbrella', 'umbrella', 'wallet', 'case', 'screw', 'case', 'wallet', 'screw', 'wallet', 'wallet']
            tokens += tokens_b + [sep_token]  # ['an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.', '[SEP]', 'bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket', 'book', 'background', 'case', 'umbrella', 'umbrella', 'wallet', 'case', 'screw', 'case', 'wallet', 'screw', 'wallet', 'wallet', '[SEP]']
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if cls_token_at_end:
            """ PASS """
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            """ PASS """
        else:
            tokens = [cls_token] + tokens  # ['[CLS]', 'an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.', '[SEP]', 'bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket', 'book', 'background', 'case', 'umbrella', 'umbrella', 'wallet', 'case', 'screw', 'case', 'wallet', 'screw', 'wallet', 'wallet', '[SEP]']
            segment_ids = [cls_token_segment_id] + segment_ids  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # [101, 2019, 3746, 3065, 2028, 5898, 14745, 2553, 1010, 6913, 2330, 2007, 3015, 22164, 9332, 2503, 1012, 102, 4524, 4524, 15882, 4524, 4524, 16195, 15882, 2553, 4979, 16195, 5047, 4920, 5047, 2553, 16195, 4979, 2338, 4281, 2553, 12977, 12977, 15882, 2553, 11224, 2553, 15882, 11224, 15882, 15882, 102]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)  # 128 - 48 = 80
        if pad_on_left:
            """ PASS """
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            """ PASS """
        else:
            input_ids = input_ids + ([pad_token] * padding_length)  # [101, 2019, 3746, 3065, 2028, 5898, 14745, 2553, 1010, 6913, 2330, 2007, 3015, 22164, 9332, 2503, 1012, 102, 4524, 4524, 15882, 4524, 4524, 16195, 15882, 2553, 4979, 16195, 5047, 4920, 5047, 2553, 16195, 4979, 2338, 4281, 2553, 12977, 12977, 15882, 2553, 11224, 2553, 15882, 11224, 15882, 15882, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
            # print(self.img_features) # 119354
            '''
            {
                ...,
                'nlvr2_test1-999-3-img0': tensor([[0.2093, 0.1726, 0.0000,  ..., 0.8653, 0.3037, 0.5747],
                                                    [0.0487, 0.0000, 0.0314,  ..., 0.9089, 0.9983, 0.8539],
                                                    [0.7593, 0.0000, 0.0000,  ..., 0.4331, 0.2360, 0.1643],
                                                    ...,
                                                    [0.0000, 0.0000, 0.0000,  ..., 0.7089, 0.2960, 0.1254],
                                                    [0.0000, 0.0000, 0.0000,  ..., 0.8396, 0.2864, 0.1631],
                                                    [0.0000, 0.0000, 0.0000,  ..., 0.4790, 0.2997, 0.1500]]), 
                'nlvr2_test1-999-3-img1': tensor([[0.6039, 2.2122, 0.0000,  ..., 0.7338, 0.4896, 0.4970],
                                                    [0.0000, 2.9828, 0.0000,  ..., 0.7337, 0.2934, 0.5637],
                                                    [0.0000, 0.0000, 0.0000,  ..., 0.7202, 0.2274, 0.0714],
                                                    ...,
                                                    [0.0000, 0.0000, 0.0000,  ..., 0.3436, 0.1675, 0.1095],
                                                    [1.4307, 5.4042, 0.0000,  ..., 0.7686, 0.2727, 0.5442],
                                                    [0.0000, 0.3192, 0.0000,  ..., 0.6729, 0.2998, 0.2141]])
            }
            '''
            img_key_left = example.img_key['left']  # "nlvr2_train-10171-0-img0"
            img_key_right = example.img_key['right']  # "nlvr2_train-10171-0-img1"
            img_feat_left = self.img_features[img_key_left]  # "nlvr2_train-10171-0-img0"
            # print(img_feat_left)  # (10, 2054)
            '''
            tensor([[0.0000, 0.0000, 0.0000,  ..., 0.3612, 0.0423, 0.0496],
                    [0.6168, 0.3325, 0.0000,  ..., 0.9983, 0.6842, 0.9818],
                    [0.0000, 0.7713, 0.0000,  ..., 0.8973, 0.4749, 0.3308],
                    ...,
                    [0.0883, 0.4411, 0.0000,  ..., 0.8131, 0.5053, 0.4717],
                    [0.0000, 0.0000, 0.0000,  ..., 0.7353, 0.2073, 0.1400],
                    [1.3575, 0.5308, 0.0813,  ..., 0.9983, 0.7785, 0.6147]])
            '''
            img_feat_right = self.img_features[img_key_right]  # "nlvr2_train-10171-0-img1"
            # print(img_feat_right)  # (19, 2054)
            '''
            tensor([[1.7650, 0.1707, 0.0000,  ..., 0.9340, 0.9012, 0.8448],
                    [0.0000, 0.0000, 0.0000,  ..., 0.4667, 0.2140, 0.1828],
                    [0.5411, 0.2517, 0.0000,  ..., 0.7580, 0.3374, 0.4781],
                    ...,
                    [1.9352, 0.0291, 0.0000,  ..., 0.8133, 0.4658, 0.5230],
                    [0.9745, 0.0000, 0.0000,  ..., 0.9202, 0.3481, 0.6354],
                    [0.0000, 0.0000, 0.0000,  ..., 0.9146, 0.3118, 0.1945]])
            '''
            img_feat = torch.cat((img_feat_left, img_feat_right), 0)  # torch.cat((10, 2054), (19, 2054), dim=0) -> (29, 2054)
            if img_feat.shape[0] > 2 * self.args.max_img_seq_length:  # 29 > 2*50
                img_feat = img_feat[0: 2 * self.args.max_img_seq_length, ]  # (50, 2054)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+100
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:  # 29 < 2*50
                if self.args.max_img_seq_length > 0:  # 50 > 0
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+29: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((2 * self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))  # (2*50-29, 2054)
                img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((29, 2054), (71, 2054), dim=0) -> (100, 2054)
                if self.args.max_img_seq_length > 0:  # 50 > 0
                    input_mask = input_mask + (
                                [0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])  # 157+71: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]
        # print(len(input_ids), len(input_mask), len(segment_ids), img_feat.shape)  # 128 228 128 (100, 2054)

        if self.args.output_mode == "classification":  # "classification"
            if example.label is None:  # 测试集example.label=0, example.score=None
                label_id = [0]
                # score = [0]
            else:  # example.label=0, example.score=None
                # label_id = [self.label_map[l] for l in example.label]  # 0 -> [0]
                label_id = [example.label]  # [0]
                # score = [0]
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

        return (torch.tensor(input_ids, dtype=torch.long),  # (128,): tensor([101, 2019, 3746, 3065, 2028, 5898, 14745, 2553, 1010, 6913, 2330, 2007, 3015, 22164, 9332, 2503, 1012, 102, 4524, 4524, 15882, 4524, 4524, 16195, 15882, 2553, 4979, 16195, 5047, 4920, 5047, 2553, 16195, 4979, 2338, 4281, 2553, 12977, 12977, 15882, 2553, 11224, 2553, 15882, 11224, 15882, 15882, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                torch.tensor(input_mask, dtype=torch.long),  # (228,): tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                torch.tensor(segment_ids, dtype=torch.long),  # (128,): tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                torch.tensor([label_id[0]], dtype=torch.long),  # (1,): tensor([0])
                img_feat,  # (100, 2054)
                torch.tensor([example.q_id], dtype=torch.long))  # (1,): tensor([0])

    def tensorize_example_pair(self,
                               example,  # InputInstance(guid="train-0", text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", text_b={"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}, label=0, score=0, img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, q_id=0)
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
            guid="train-0",
            text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.",
            text_b={
                "left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket",
                "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"
            },
            label=0,
            score=None,
            img_key={
                "left": "nlvr2_train-10171-0-img0",
                "right": "nlvr2_train-10171-0-img1"
            },
            q_id=0
        )
        """

        # Text Processor
        tokens_a = self.tokenizer.tokenize(example.text_a)  # ['an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.']

        choices = []
        for choice_key in example.img_key:  # ("left", "right")
            tokens_b = None
            if example.text_b:  # {"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}
                tokens_b = self.tokenizer.tokenize(example.text_b[choice_key])  # ['bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket']
                # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
                _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
            else:
                """ PASS """
                if len(tokens_a) > self.args.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.args.max_seq_length - 2)]  # Account for [CLS] and [SEP] with "- 2"
                """ PASS """

            tokens = tokens_a + [sep_token]  # ['an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.', '[SEP]']
            segment_ids = [sequence_a_segment_id] * len(tokens)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            if tokens_b:  # ['bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket']
                tokens += tokens_b + [sep_token]  # ['an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.', '[SEP]', 'bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket', '[SEP]']
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            if cls_token_at_end:
                """ PASS """
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
                """ PASS """
            else:
                tokens = [cls_token] + tokens  # ['[CLS]', 'an', 'image', 'shows', 'one', 'leather', 'pencil', 'case', ',', 'displayed', 'open', 'with', 'writing', 'implements', 'tucked', 'inside', '.', '[SEP]', 'bag', 'bag', 'wallet', 'bag', 'bag', 'strap', 'wallet', 'case', 'pocket', 'strap', 'handle', 'hole', 'handle', 'case', 'strap', 'pocket', '[SEP]']
                segment_ids = [cls_token_segment_id] + segment_ids  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # [101, 2019, 3746, 3065, 2028, 5898, 14745, 2553, 1010, 6913, 2330, 2007, 3015, 22164, 9332, 2503, 1012, 102, 4524, 4524, 15882, 4524, 4524, 16195, 15882, 2553, 4979, 16195, 5047, 4920, 5047, 2553, 16195, 4979, 102]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)  # [[1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1]]
            input_mask_txt = [1 if mask_padding_with_zero else 0] * len(input_ids)  # [[1], 1, 1, 1, ..., 1, [1], 1, 1, 1, ..., 1, [1]]
            input_mask_img = [0 if mask_padding_with_zero else 0] * len(input_ids)  # [[0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [0]]

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_length - len(input_ids)  # 128 - 35 = 93
            if pad_on_left:
                """ PASS """
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                """ PASS """
            else:
                input_ids = input_ids + ([pad_token] * padding_length)  # [101, 2019, 3746, 3065, 2028, 5898, 14745, 2553, 1010, 6913, 2330, 2007, 3015, 22164, 9332, 2503, 1012, 102, 4524, 4524, 15882, 4524, 4524, 16195, 15882, 2553, 4979, 16195, 5047, 4920, 5047, 2553, 16195, 4979, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  # [[1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                input_mask_txt = input_mask_txt + ([0 if mask_padding_with_zero else 1] * padding_length)  # [[1], 1, 1, 1, ..., 1, [1], 1, 1, 1, ..., 1, [1], 0, 0, 0, ..., 0]
                input_mask_img = input_mask_img + ([0 if mask_padding_with_zero else 1] * padding_length)  # [[0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0]
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(input_mask_txt) == self.args.max_seq_length
            assert len(input_mask_img) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            # Image Processor
            img_key = example.img_key[choice_key]  # "nlvr2_train-10171-0-img0"
            img_feat = self.img_features[img_key]
            # print(img_feat)  # (10, 2054)
            '''
            tensor([[0.0000, 0.0000, 0.0000,  ..., 0.3612, 0.0423, 0.0496],
                    [0.6168, 0.3325, 0.0000,  ..., 0.9983, 0.6842, 0.9818],
                    [0.0000, 0.7713, 0.0000,  ..., 0.8973, 0.4749, 0.3308],
                    ...,
                    [0.0883, 0.4411, 0.0000,  ..., 0.8131, 0.5053, 0.4717],
                    [0.0000, 0.0000, 0.0000,  ..., 0.7353, 0.2073, 0.1400],
                    [1.3575, 0.5308, 0.0813,  ..., 0.9983, 0.7785, 0.6147]])
            '''
            if img_feat.shape[0] > self.args.max_img_seq_length:  # 10 > 50
                img_feat = img_feat[0: self.args.max_img_seq_length, ]  # (50, 2054)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+50
                    input_mask_txt = input_mask_txt + [0 if mask_padding_with_zero else 0] * img_feat.shape[0]  # [[1], 1, 1, 1, ..., 1, [1], 1, 1, 1, ..., 1, [1], 0, 0, 0, ..., 0, [0, 0, 0, ..., 0]]
                    input_mask_img = input_mask_img + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # [[0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [1, 1, 1, ..., 1]]
            else:  # 10 < 50
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+10: [[1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    input_mask_txt = input_mask_txt + [0 if mask_padding_with_zero else 0] * img_feat.shape[0]  # [[1], 1, 1, 1, ..., 1, [1], 1, 1, 1, ..., 1, [1], 0, 0, 0, ..., 0, [0, 0, 0, ..., 0]]
                    input_mask_img = input_mask_img + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # [[0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [1, 1, 1, ..., 1]]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))  # (50-10, 2054)
                img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((10, 2054), (40, 2054), dim=0) -> (50, 2054)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])  # 138+40: [[1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    input_mask_txt = input_mask_txt + [0 if mask_padding_with_zero else 0] * padding_matrix.shape[0]  # [[1], 1, 1, 1, ..., 1, [1], 1, 1, 1, ..., 1, [1], 0, 0, 0, ..., 0, [0, 0, 0, ..., 0], 0, 0, 0, ..., 0]
                    input_mask_img = input_mask_img + [0 if mask_padding_with_zero else 0] * padding_matrix.shape[0]  # [[0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [0], 0, 0, 0, ..., 0, [1, 1, 1, ..., 1], 0, 0, 0, ..., 0]

            # print(len(input_ids), len(input_mask), len(segment_ids), img_feat.shape)  # 128 178 128 (50, 2054)
            # print(len(input_mask_txt), len(input_mask_img))

            choices.append((tokens, input_ids, input_mask, segment_ids, img_feat,
                            #
                            input_mask_txt,
                            input_mask_img))
        # print(choices)
        '''
        [
            (
                tokens: 长度为35的list,
                input_ids: 长度为128的list,
                input_mask: 长度为178的list,
                segment_ids: 长度为128的list,
                img_feat: (50, 2054),
                input_mask_txt: 长度为178的list,
                input_mask_img: 长度为178的list,
            ),
            (
                tokens: 长度为32的list,
                input_ids: 长度为128的list,
                input_mask: 长度为178的list,
                segment_ids: 长度为128的list,
                img_feat: (50, 2054),
                input_mask_txt: 长度为178的list,
                input_mask_img: 长度为178的list,
            )
        ]
        '''
        if self.args.output_mode == "classification":  # "classification"
            if example.label is None:
                label_id = [0]
                score = [0]
            else:  # example.label=0, example.score=None
                # label_id = [self.label_map[l] for l in example.label]  # 0 -> [0]
                label_id = [example.label]  # [0]
                score = [0]
        elif self.args.output_mode == "regression":
            """ PASS """
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
            """ PASS """
        else:
            raise KeyError(self.args.output_mode)

        choice_input_ids = [choice[1] for choice in choices]  # [长度为128的list, 长度为128的list]
        choice_input_mask = [choice[2] for choice in choices]  # [长度为178的list, 长度为178的list]
        choice_input_segs = [choice[3] for choice in choices]  # [长度为128的list, 长度为128的list]
        choice_input_imgs = [choice[4] for choice in choices]  # [(50, 2054), (50, 2054)]
        #
        choice_input_mask_txt = [choice[5] for choice in choices]
        choice_input_mask_img = [choice[6] for choice in choices]

        choice_img_feats = torch.stack(choice_input_imgs, dim=0)  # torch.stack([(50, 2054), (50, 2054)], dim=0) -> (2, 50, 2054)

        return (torch.tensor(choice_input_ids, dtype=torch.long),  # (2, 128)
                torch.tensor(choice_input_mask, dtype=torch.long),  # (2, 178)
                torch.tensor(choice_input_segs, dtype=torch.long),  # (2, 128)
                torch.tensor(label_id, dtype=torch.long),  # (1,): tensor([0])
                choice_img_feats,  # (2, 50, 2054)
                torch.tensor([example.q_id], dtype=torch.long),  # (1,): tensor([0])
                #
                torch.tensor(choice_input_mask_txt, dtype=torch.long),  # (2, 178)
                torch.tensor(choice_input_mask_img, dtype=torch.long))  # (2, 178)

    def __getitem__(self, index):
        # print(self.examples)
        """
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
        """
        entry = self.examples[index]
        # print(entry)
        '''
        InputInstance(
            guid="train-0", 
            text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", 
            text_b={"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}, 
            label=0, 
            score=0, 
            img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, 
            q_id=0
        )
        '''
        if self.args.use_pair:  # args.use_pair=True
            example = self.tensorize_example_pair(entry,  # InputInstance(guid="train-0", text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", text_b={"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}, label=0, score=0, img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, q_id=0)
                                                  cls_token_at_end=bool(self.args.model_type in ['xlnet']),  # False
                                                  pad_on_left=bool(self.args.model_type in ['xlnet']),  # False
                                                  cls_token=self.tokenizer.cls_token,  # '[CLS]'
                                                  sep_token=self.tokenizer.sep_token,  # '[SEP]'
                                                  cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,  # 0
                                                  pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)  # 0
            # print(example)
            '''
            (
                input_ids: (2, 128),
                input_mask: (2, 178),
                segment_ids: (2, 128),
                label_id: (1,),
                img_feat: (2, 50, 2054),
                q_id: (1,),
                input_mask_txt: (2, 178),
                input_mask_img: (2, 178)
            )
            '''
        else:
            """ PASS """
            example = self.tensorize_example(entry,  # InputInstance(guid="train-0", text_a="An image shows one leather pencil case, displayed open with writing implements tucked inside.", text_b={"left": "bag bag wallet bag bag strap wallet case pocket strap handle hole handle case strap pocket", "right": "book background case umbrella umbrella wallet case screw case wallet screw wallet wallet"}, label=0, score=0, img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, q_id=0)
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
                input_mask: (228,),
                segment_ids: (128,),
                label_id: (1,),
                img_feat: (100, 2054),
                q_id: (1,)
            )
            '''
            """ PASS """
        return example

    def __len__(self):
        return len(self.examples)


def _load_dataset(args, name):  # args 'train'或'val'或'test1'
    processor = processors[args.task_name]()  # NLVRProcessor
    labels = processor.get_labels()  # None
    # print(labels)  # 长度为2的list
    '''
    [0, 1]
    '''

    if name == 'train':
        examples = processor.get_train_examples(args.data_dir, args.use_label_seq, 'nlvr2_train.json')  # datasets/nlvr2 True nlvr2_train.json
        # print(examples)  # 86373
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
    elif name == 'val':
        if args.eval_data_type == 'bal':
            """ PASS """
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_balanced_dev.json')
            """ PASS """
        elif args.eval_data_type == 'unbal':
            """ PASS """
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_unbalanced_dev.json')
            """ PASS """
        else:  # args.eval_data_type=all
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_dev.json')  # datasets/nlvr2 True nlvr2_dev.json
            # print(examples)  # 6982
            '''
            [
                InputInstance(
                    guid="dev-0", 
                    text_a="The right image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage.", 
                    text_b={"left": "garden ground rocks plants mulch rock ground block barrel leaves wall plant barrel wall plants ground dirt rock plant", "right": "bottles bottles bottle grass bottle bottles bottle bottle bottle bottle bottle bottle label vase bottle label bottle wine bottle label bottles label bottle bottle bottle bottle bottle bucket meter bottle bottle bottles bag bottle wine bottle bottles bottle bottle bottle label bottle bottle label bottles"},
                    label=0, 
                    score=None, 
                    img_key={"left": "nlvr2_train-10171-0-img0", "right": "nlvr2_train-10171-0-img1"}, 
                    q_id=0
                ),
                ...
            ]
            '''
    elif name == 'test1':  # test-submission
        if args.test_data_type == 'bal':
            """ PASS """
            examples = processor.get_test_examples(args.data_dir, args.use_label_seq, 'nlvr2_balanced_test1.json')
            """ PASS """
        elif args.test_data_type == 'unbal':
            """ PASS """
            examples = processor.get_test_examples(args.data_dir, args.use_label_seq, 'nlvr2_unbalanced_test1.json')
            """ PASS """
        else:  # args.test_data_type=all
            examples = processor.get_test_examples(args.data_dir, args.use_label_seq, 'nlvr2_test1.json')  # datasets/nlvr2 True nlvr2_test1.json
            # print(examples)  # 6967
            '''
            [
                InputInstance(
                    guid="test-0", 
                    text_a="There is an empty glass.", 
                    text_b={"left": "wall bottle table label glass wall glass sticker wood pole circle lid label shadow skateboard water wood neck neck", "right": "drink wall picture sign glass wine picture glass man table sign window bottle window table reflection glass letter bowl wall face liquid letter sign letter sign hair"},
                    label=0, 
                    score=None, 
                    img_key={"left": "nlvr2_test1-0-1-img0", "right": "nlvr2_test1-0-1-img1"}, 
                    q_id=0
                ),
                ...
            ]
            '''
    elif name == 'val+test1':
        """ PASS """
        if args.eval_data_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_balanced_dev.json')
        elif args.eval_data_type == 'unbal':
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_unbalanced_dev.json')
        else:
            examples = processor.get_dev_examples(args.data_dir, args.use_label_seq, 'nlvr2_dev_test1.json')
        """ PASS """

    return examples, labels


def _load_img_features(args):
    t_start = time.time()
    if args.img_feature_type == 'faster_r-cnn':  # faster_r-cnn
        if args.img_feature_dim == 2048:  # object features: 2048
            """ PASS """
            feat_file_name = 'nlvr2_img_frcnn_obj_feats.pt'
            """ PASS """
        else:  # object + spatial features: 2054
            feat_file_name = 'nlvr2_img_frcnn_feats.pt'  # nlvr2_img_frcnn_feats.pt
    else:
        """ PASS """
        feat_file_name = 'nlvr2_img_feats.pt'
        """ PASS """
    img_features = torch.load(os.path.join(args.data_dir, feat_file_name))  # datasets/nlvr2/nlvr2_img_frcnn_feats.pt
    # print(img_features)  # 119354
    '''
    {
        ...,
        'nlvr2_test1-999-3-img0': tensor([[0.2093, 0.1726, 0.0000,  ..., 0.8653, 0.3037, 0.5747],
                                          [0.0487, 0.0000, 0.0314,  ..., 0.9089, 0.9983, 0.8539],
                                          [0.7593, 0.0000, 0.0000,  ..., 0.4331, 0.2360, 0.1643],
                                          ...,
                                          [0.0000, 0.0000, 0.0000,  ..., 0.7089, 0.2960, 0.1254],
                                          [0.0000, 0.0000, 0.0000,  ..., 0.8396, 0.2864, 0.1631],
                                          [0.0000, 0.0000, 0.0000,  ..., 0.4790, 0.2997, 0.1500]]), 
        'nlvr2_test1-999-3-img1': tensor([[0.6039, 2.2122, 0.0000,  ..., 0.7338, 0.4896, 0.4970],
                                          [0.0000, 2.9828, 0.0000,  ..., 0.7337, 0.2934, 0.5637],
                                          [0.0000, 0.0000, 0.0000,  ..., 0.7202, 0.2274, 0.0714],
                                          ...,
                                          [0.0000, 0.0000, 0.0000,  ..., 0.3436, 0.1675, 0.1095],
                                          [1.4307, 5.4042, 0.0000,  ..., 0.7686, 0.2727, 0.5442],
                                          [0.0000, 0.3192, 0.0000,  ..., 0.6729, 0.2998, 0.2141]])
    }

    '''
    t_end = time.time()
    logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))
    '''
    03/27/2022 19:59:26 - INFO - __main__ - Info: loading nlvr2_img_frcnn_feats.pt features using 14.09 secs
    '''
    return img_features


def train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer):
    """ Train the model """

    # print(train_dataset)  # 86373
    '''
    NLVRDataset(Dataset): 初始化参数为args, name, tokenizer, img_features
        self.args: args
        self.name: 'train'或'val'或'test1'
        self.tokenizer: tokenizer

        self.output_mode: "classification"
        
        self.img_features: {
                                ...,
                                'nlvr2_test1-999-3-img0': tensor([[0.2093, 0.1726, 0.0000,  ..., 0.8653, 0.3037, 0.5747],
                                                                  [0.0487, 0.0000, 0.0314,  ..., 0.9089, 0.9983, 0.8539],
                                                                  [0.7593, 0.0000, 0.0000,  ..., 0.4331, 0.2360, 0.1643],
                                                                  ...,
                                                                  [0.0000, 0.0000, 0.0000,  ..., 0.7089, 0.2960, 0.1254],
                                                                  [0.0000, 0.0000, 0.0000,  ..., 0.8396, 0.2864, 0.1631],
                                                                  [0.0000, 0.0000, 0.0000,  ..., 0.4790, 0.2997, 0.1500]]), 
                                'nlvr2_test1-999-3-img1': tensor([[0.6039, 2.2122, 0.0000,  ..., 0.7338, 0.4896, 0.4970],
                                                                  [0.0000, 2.9828, 0.0000,  ..., 0.7337, 0.2934, 0.5637],
                                                                  [0.0000, 0.0000, 0.0000,  ..., 0.7202, 0.2274, 0.0714],
                                                                  ...,
                                                                  [0.0000, 0.0000, 0.0000,  ..., 0.3436, 0.1675, 0.1095],
                                                                  [1.4307, 5.4042, 0.0000,  ..., 0.7686, 0.2727, 0.5442],
                                                                  [0.0000, 0.3192, 0.0000,  ..., 0.6729, 0.2998, 0.2141]])
                            }
        self.examples:  [
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
        self.labels: [0, 1]
        self.label_map: {
                            0: 0,
                            1: 1
                        }

        def __getitem__(index): # 对每一个self.examples[index]进行处理后得到:
                                (
                                    input_ids: (2, 128),
                                    input_mask: (2, 178),
                                    segment_ids: (2, 128),
                                    label_id: (1,),
                                    img_feat: (2, 50, 2054),
                                    q_id: (1,),
                                    input_mask_txt: (2, 178),
                                    input_mask_img: (2, 178)
                                )
        def __len__(): len(self.examples)
    '''

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # 32 * 1
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)  # RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  # num_workers=args.workers,  # 默认为0
                                  sampler=train_sampler,  # RandomSampler(train_dataset)
                                  batch_size=args.train_batch_size)  # 32
    # print(train_dataloader)  # 86373/32=2700
    '''
    [
        [
            torch.Size([16, 2, 128]),
            torch.Size([16, 2, 178]),
            torch.Size([16, 2, 128]),
            torch.Size([16, 1]),
            torch.Size([16, 2, 50, 2054]),
            torch.Size([16, 1]),
            torch.Size([16, 2, 178]),
            torch.Size([16, 2, 178])
        ],
        [
            torch.Size([16, 2, 128]),
            torch.Size([16, 2, 178]),
            torch.Size([16, 2, 128]),
            torch.Size([16, 1]),
            torch.Size([16, 2, 50, 2054]),
            torch.Size([16, 1]),
            torch.Size([16, 2, 178]),
            torch.Size([16, 2, 178])
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
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs  # 2700 // 1 * 20.0 = 54000.0

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    teacher_optimizer_grouped_parameters = [
        {
            'params': [p for n, p in teacher_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in teacher_model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    student_optimizer_grouped_parameters = [
        {
            'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    if args.optim == 'AdamW':
        teacher_optimizer = AdamW(teacher_optimizer_grouped_parameters,
                                  lr=args.teacher_learning_rate,  # 3e-5
                                  eps=args.adam_epsilon)  # 1e-8
        student_optimizer = AdamW(student_optimizer_grouped_parameters,
                                  lr=args.student_learning_rate,  # 3e-5
                                  eps=args.adam_epsilon)  # 1e-8
    elif args.optim == 'Adamax':
        teacher_optimizer = torch.optim.Adamax(teacher_optimizer_grouped_parameters,
                                               lr=args.teacher_learning_rate,
                                               eps=args.adam_epsilon)
        student_optimizer = torch.optim.Adamax(student_optimizer_grouped_parameters,
                                               lr=args.student_learning_rate,
                                               eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)  # original
    if args.scheduler == "constant":  # constant warmup and decay
        teacher_scheduler = WarmupConstantSchedule(teacher_optimizer, warmup_steps=args.warmup_steps)
        student_scheduler = WarmupConstantSchedule(student_optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":  # linear warmup and decay
        teacher_scheduler = WarmupLinearSchedule(teacher_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        student_scheduler = WarmupLinearSchedule(student_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

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
    # if args.n_gpu > 1:
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
    logger.info("  Num examples = %d", len(train_dataset))  # 86373
    logger.info("  Num Epochs = %d", args.num_train_epochs)  # 20
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)  # 32
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))  # 32 * 1 * 1
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)  # 1
    logger.info("  Total optimization steps = %d", t_total)  # 54000
    '''
    03/29/2022 21:59:06 - INFO - __main__ - ***** Running training *****
    03/29/2022 21:59:06 - INFO - __main__ -   Num examples = 86373
    03/29/2022 21:59:06 - INFO - __main__ -   Num Epochs = 20
    03/29/2022 21:59:06 - INFO - __main__ -   Instantaneous batch size per GPU = 32
    03/29/2022 21:59:06 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 32
    03/29/2022 21:59:06 - INFO - __main__ -   Gradient Accumulation steps = 1
    03/29/2022 21:59:06 - INFO - __main__ -   Total optimization steps = 54000
    '''

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    student_model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])  # 0~19

    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': student_model.state_dict(),  # copy.deepcopy(model)
        'optimizer_state': student_optimizer.state_dict()
    }

    # for epoch in range(int(args.num_train_epochs)):
    for epoch in train_iterator:  # 0~19

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])  # 0~2699

        # total_loss = 0
        # train_score = 0
        # total_norm = 0
        # count_norm = 0

        t_start = time.time()

        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(epoch_iterator):  # 0~2699
            # print(batch)
            '''
            [
                torch.Size([16, 2, 128]),  # input_ids
                torch.Size([16, 2, 178]),  # input_mask
                torch.Size([16, 2, 128]),  # segment_ids
                torch.Size([16, 1]),  # label_id
                torch.Size([16, 2, 50, 2054]),  # img_feat
                torch.Size([16, 1]),  # q_id
                torch.Size([16, 2, 178]),  # input_mask_txt
                torch.Size([16, 2, 178])  # input_mask_img
            ]
            '''

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],  # input_ids: (16, 2, 128)
                'attention_mask': batch[1],  # input_mask: (16, 2, 178)
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 2, 128)
                'labels': batch[3],  # label_id: (16, 1)
                'img_feats': None if args.img_feature_dim == -1 else batch[4],  # img_feat: (16, 2, 50, 2054)
                #
                'attention_mask_txt': batch[6],  # input_mask_txt: (16, 2, 178)
                'attention_mask_img': batch[7]  # input_mask_img: (16, 2, 178)
                #
            }

            # 1. θ_s' ← θ_s' - λ * ▽_θ_s' L(x;θ_s';θ_t)
            student_model.train()
            teacher_model.eval()

            sum_gradients = {}

            # θ_s' = θ_s
            fast_model = copy.deepcopy(student_model)
            fast_model.to(args.device)

            fast_optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in fast_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay
                },
                {
                    'params': [p for n, p in fast_model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]
            if args.optim == 'AdamW':
                fast_optimizer = AdamW(fast_optimizer_grouped_parameters,
                                       lr=args.student_learning_rate,  # 3e-5
                                       eps=args.adam_epsilon)  # 1e-8
            elif args.optim == 'Adamax':
                fast_optimizer = torch.optim.Adamax(fast_optimizer_grouped_parameters,
                                                    lr=args.student_learning_rate,
                                                    eps=args.adam_epsilon)

            if args.scheduler == "constant":  # constant warmup and decay
                fast_scheduler = WarmupConstantSchedule(fast_optimizer, warmup_steps=args.warmup_steps)
            elif args.scheduler == "linear":  # linear warmup and decay
                fast_scheduler = WarmupLinearSchedule(fast_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

            fast_model.train()
            fast_model.to(args.device)

            # text only
            fast_outputs_txt = fast_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                          token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                          attention_mask=inputs['attention_mask_txt'],  # (16, 2, 178)
                                          # labels=inputs['labels'],  # (16, 1)
                                          img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
            # image only
            fast_outputs_img = fast_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                          token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                          attention_mask=inputs['attention_mask_img'],  # (16, 2, 178)
                                          # labels=inputs['labels'],  # (16, 1)
                                          img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
            # text + image
            fast_outputs = fast_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                      token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                      attention_mask=inputs['attention_mask'],  # (16, 2, 178)
                                      labels=inputs['labels'],  # (16, 1)
                                      img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
            '''
            (
                loss: torch(数)
                logits: (16, 2),
                hidden_states: tuple(13个(32, 178, 768)),
                attentions: tuple(12个(32, 12, 178, 178))
            )
            '''
            with torch.no_grad():
                # text only
                teacher_outputs_txt = teacher_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                                    token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                    attention_mask=inputs['attention_mask_txt'],  # (16, 2, 178))
                                                    img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
                # image only
                teacher_outputs_img = teacher_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                                    token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                    attention_mask=inputs['attention_mask_img'],  # (16, 2, 178))
                                                    img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
                # text + image
                teacher_outputs = teacher_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                                token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                attention_mask=inputs['attention_mask'],  # (16, 2, 178))
                                                img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
                '''
                 (
                    logits: (16, 2),
                    hidden_states: tuple(13个(32, 178, 768)),
                    attentions: tuple(12个(32, 12, 178, 178))
                )
                '''

            # L_task_specific
            task_specific_loss, fast_logits = fast_outputs[0:2]  # torch(数) (16, 2)
            # task_specific_loss_txt, fast_logits_txt = fast_outputs_txt[0:2]
            # task_specific_loss_img, fast_logits_img = fast_outputs_img[0:2]
            fast_logits_txt = fast_outputs_txt[0]
            fast_logits_img = fast_outputs_img[0]
            teacher_logits = teacher_outputs[0]  # (16, 2)
            teacher_logits_txt = teacher_outputs_txt[0]
            teacher_logits_img = teacher_outputs_img[0]

            # L_vanilla_kd
            T = args.temperature
            # text only
            vanilla_kd_loss_txt = F.kl_div(
                F.log_softmax(fast_logits_txt / T, dim=-1),
                F.softmax(teacher_logits_txt / T, dim=-1),
                reduction='batchmean'
            ) * T * T
            # image only
            vanilla_kd_loss_img = F.kl_div(
                F.log_softmax(fast_logits_img / T, dim=-1),
                F.softmax(teacher_logits_img / T, dim=-1),
                reduction='batchmean'
            ) * T * T
            # text + image
            vanilla_kd_loss = F.kl_div(
                F.log_softmax(fast_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * T * T

            # L = (1 - α) * L_task_specific + α * L_vanilla_kd
            # loss = (1 - args.alpha) * (0.5 * task_specific_loss + 0.25 * task_specific_loss_txt + 0.25 * task_specific_loss_img) + args.alpha * (0.5 * vanilla_kd_loss + 0.25 * vanilla_kd_loss_txt + 0.25 * vanilla_kd_loss_img)  # L(x;θ_s';θ_t)
            loss = (1 - args.alpha) * task_specific_loss + args.alpha * (0.5 * vanilla_kd_loss + 0.25 * vanilla_kd_loss_txt + 0.25 * vanilla_kd_loss_img)  # L(x;θ_s';θ_t)

            # """ PASS """
            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # """ PASS """
            #
            # """ PASS """
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps
            # """ PASS """
            #
            # if args.fp16:
            #     """ PASS """
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            #     """ PASS """
            # else:
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            loss.backward()  # ▽_θ_s' L(x;θ_s';θ_t)
            # total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 1.0
            # count_norm += 1
            torch.nn.utils.clip_grad_norm_(fast_model.parameters(), args.max_grad_norm)  # 1.0

            # if (step + 1) % args.gradient_accumulation_steps == 0:
            fast_optimizer.step()
            fast_scheduler.step()  # Update learning rate schedule
            fast_optimizer.zero_grad()

            # 2. θ_t ← θ_t - μ * (θ_t - θ_s')
            # student_model.train()
            teacher_model.train()

            teacher_weights = {}
            for n, p in teacher_model.named_parameters():
                teacher_weights[n] = p

            fast_weights = {}
            """
            layer-wise parameters updating strategies
            """
            # first: 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3, 4 -> 4, 5 -> 5
            if args.strategy == 'first':
                for fast_n, fast_p in fast_model.named_parameters():
                    fast_weights[fast_n] = fast_p
            # last: 0 -> 6, 1 -> 7, 2 -> 8, 3 -> 9, 4 -> 10, 5 -> 11
            elif args.strategy == 'last':
                for fast_n, fast_p in fast_model.named_parameters():
                    if 'bert.encoder.layer.0' in fast_n:
                        new_fast_n = fast_n.replace('0', '6')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.1' in fast_n:
                        new_fast_n = fast_n.replace('1', '7')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.2' in fast_n:
                        new_fast_n = fast_n.replace('2', '8')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.3' in fast_n:
                        new_fast_n = fast_n.replace('3', '9')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.4' in fast_n:
                        new_fast_n = fast_n.replace('4', '10')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.5' in fast_n:
                        new_fast_n = fast_n.replace('5', '11')
                        fast_weights[new_fast_n] = fast_p
                    else:
                        fast_weights[fast_n] = fast_p
            # skip: 0 -> 1, 1 -> 3, 2 -> 5, 3 -> 7, 4 -> 9, 5 -> 11
            elif args.strategy == 'skip':
                for fast_n, fast_p in fast_model.named_parameters():
                    if 'bert.encoder.layer.0' in fast_n:
                        new_fast_n = fast_n.replace('0', '1')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.1' in fast_n:
                        new_fast_n = fast_n.replace('1', '3')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.2' in fast_n:
                        new_fast_n = fast_n.replace('2', '5')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.3' in fast_n:
                        new_fast_n = fast_n.replace('3', '7')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.4' in fast_n:
                        new_fast_n = fast_n.replace('4', '9')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.5' in fast_n:
                        new_fast_n = fast_n.replace('5', '11')
                        fast_weights[new_fast_n] = fast_p
                    else:
                        fast_weights[fast_n] = fast_p
            # both: 0 -> 0,1, 1 -> 2,3, 2 -> 4,5, 3 -> 6,7, 4 -> 8,9, 5 -> 10,11
            elif args.strategy == 'both':
                for fast_n, fast_p in fast_model.named_parameters():
                    if 'bert.encoder.layer.0.' in fast_n:
                        new_fast_n = fast_n.replace('0', '0')
                        fast_weights[new_fast_n] = fast_p
                        new_fast_n = fast_n.replace('0', '1')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.1' in fast_n:
                        new_fast_n = fast_n.replace('1', '2')
                        fast_weights[new_fast_n] = fast_p
                        new_fast_n = fast_n.replace('1', '3')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.2' in fast_n:
                        new_fast_n = fast_n.replace('2', '4')
                        fast_weights[new_fast_n] = fast_p
                        new_fast_n = fast_n.replace('2', '5')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.3' in fast_n:
                        new_fast_n = fast_n.replace('3', '6')
                        fast_weights[new_fast_n] = fast_p
                        new_fast_n = fast_n.replace('3', '7')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.4' in fast_n:
                        new_fast_n = fast_n.replace('4', '8')
                        fast_weights[new_fast_n] = fast_p
                        new_fast_n = fast_n.replace('4', '9')
                        fast_weights[new_fast_n] = fast_p
                    elif 'bert.encoder.layer.5' in fast_n:
                        new_fast_n = fast_n.replace('5', '10')
                        fast_weights[new_fast_n] = fast_p
                        new_fast_n = fast_n.replace('5', '11')
                        fast_weights[new_fast_n] = fast_p
                    else:
                        fast_weights[fast_n] = fast_p
            else:
                raise NotImplementedError()

            for fast_n, fast_p in fast_weights.items():
                gradient = teacher_weights[fast_n] - fast_p  # θ_t - θ_s'
                if fast_n in sum_gradients:
                    sum_gradients[fast_n] += gradient
                else:
                    sum_gradients[fast_n] = gradient

            del fast_model, fast_optimizer, fast_scheduler
            torch.cuda.empty_cache()

            for n, p in teacher_model.named_parameters():
                if n in sum_gradients:
                    p.gard = sum_gradients[n]
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.max_grad_norm)  # 1.0
            teacher_optimizer.step()  # θ_t ← θ_t - μ * (θ_t - θ_s')
            teacher_scheduler.step()
            teacher_optimizer.zero_grad()

            del sum_gradients
            gc.collect()

            # 3. θ_s ← θ_s - λ * ▽_θ_s L(x;θ_s;θ_t)
            # student_model.train()
            teacher_model.eval()

            # text only
            student_outputs_txt = student_model(input_ids=inputs['input_ids'],
                                                token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                attention_mask=inputs['attention_mask_txt'],  # (16, 2, 178)
                                                # labels=inputs['labels'],  # (16, 1)
                                                img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
            # image only
            student_outputs_img = student_model(input_ids=inputs['input_ids'],
                                                token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                attention_mask=inputs['attention_mask_img'],  # (16, 2, 178)
                                                # labels=inputs['labels'],  # (16, 1)
                                                img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
            # text + image
            student_outputs = student_model(input_ids=inputs['input_ids'],
                                            token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                            attention_mask=inputs['attention_mask'],  # (16, 2, 178)
                                            labels=inputs['labels'],  # (16, 1)
                                            img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
            with torch.no_grad():
                # text only
                teacher_outputs_txt = teacher_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                                    token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                    attention_mask=inputs['attention_mask_txt'],  # (16, 2, 178))
                                                    img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
                # image only
                teacher_outputs_img = teacher_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                                    token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                    attention_mask=inputs['attention_mask_img'],  # (16, 2, 178))
                                                    img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)
                # text + image
                teacher_outputs = teacher_model(input_ids=inputs['input_ids'],  # (16, 2, 128)
                                                token_type_ids=inputs['token_type_ids'],  # (16, 2, 128)
                                                attention_mask=inputs['attention_mask'],  # (16, 2, 178))
                                                img_feats=inputs['img_feats'])  # 新增: (16, 2, 50, 2054)

            # L_task_specific
            task_specific_loss, student_logits = student_outputs[0:2]  # torch(数) (16, 2)
            # task_specific_loss_txt, student_logits_txt = student_outputs_txt[0:2]
            # task_specific_loss_img, student_logits_img = student_outputs_img[0:2]
            student_logits_txt = student_outputs_txt[0]
            student_logits_img = student_outputs_img[0]
            teacher_logits = teacher_outputs[0]  # (16, 2)
            teacher_logits_txt = teacher_outputs_txt[0]
            teacher_logits_img = teacher_outputs_img[0]

            # L_vanilla_kd
            # text only
            vanilla_kd_loss_txt = F.kl_div(
                F.log_softmax(student_logits_txt / T, dim=-1),
                F.softmax(teacher_logits_txt / T, dim=-1),
                reduction='batchmean'
            ) * T * T
            # image only
            vanilla_kd_loss_img = F.kl_div(
                F.log_softmax(student_logits_img / T, dim=-1),
                F.softmax(teacher_logits_img / T, dim=-1),
                reduction='batchmean'
            ) * T * T
            # text + image
            vanilla_kd_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * T * T

            # L = (1 - α) * L_task_specific + α * L_vanilla_kd
            # loss = (1 - args.alpha) * (0.5 * task_specific_loss + 0.25 * task_specific_loss_txt + 0.25 * task_specific_loss_img) + args.alpha * (0.5 * vanilla_kd_loss + 0.25 * vanilla_kd_loss_txt + 0.25 * vanilla_kd_loss_img)  # L(x;θ_s;θ_t)
            loss = (1 - args.alpha) * task_specific_loss + args.alpha * (0.5 * vanilla_kd_loss + 0.25 * vanilla_kd_loss_txt + 0.25 * vanilla_kd_loss_img)  # L(x;θ_s;θ_t)

            loss.backward()  # ▽_θ_s L(x;θ_s;θ_t)

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 1.0

            tr_loss += loss.item()
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            student_optimizer.step()
            student_scheduler.step()
            student_optimizer.zero_grad()
            global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        logger.info("***** Epoch: {} *****".format(epoch + 1))
        logger.info("  Train Loss: {}".format(tr_loss / len(train_dataset)))
        '''
        03/27/2022 20:14:51 - INFO - __main__ - ***** Epoch: 1 *****
        03/27/2022 20:14:51 - INFO - __main__ -   Train loss: 0.04532386765648091
        '''

        t_end = time.time()
        logger.info('  Train Time Cost: %.3f' % (t_end - t_start))
        '''
        03/27/2022 20:14:51 - INFO - __main__ -   Train Time Cost: 924.019
        '''

        # 每个epoch结束时做一次evaluation并保存模型
        # evaluation
        eval_result, eval_score = evaluate(args, student_model, eval_dataset, prefix='')
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(student_model)
            # best_model['optimizer'] = copy.deepcopy(student_optimizer.state_dict())
        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch > 0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            base_output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))  # ./model/nlvr2/student/chechpoint-1
            # teacher_output_dir = os.path.join(base_output_dir, 'teacher')  # ./model/nlvr2/student/chechpoint-1/teacher
            student_output_dir = os.path.join(base_output_dir, 'student')  # ./model/nlvr2/student/chechpoint-1/student
            # for output_dir in [teacher_output_dir, student_output_dir]:
            for output_dir in [student_output_dir]:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            student_model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Take care of distributed/parallel training
            student_model_to_save.save_pretrained(student_output_dir)
            torch.save(args, os.path.join(student_output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(student_output_dir)
            logger.info("Saving student model checkpoint {0} to {1}".format(epoch + 1, student_output_dir))
            '''
            03/27/2022 20:15:22 - INFO - __main__ - Saving student model checkpoint 1 to model/nlvr2/student\checkpoint-1
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
        03/27/2022 20:15:22 - INFO - __main__ - Epoch: 1, Train Time: 955.006
        03/27/2022 20:15:22 - INFO - __main__ - ********************
        '''

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # 所有epoch结束后保存最好的模型
    if args.local_rank in [-1, 0]:  # Save the final model checkpoint
        # output_dir = os.path.join(args.output_dir, 'best'.format(best_model['epoch']))  # model/nlvr2/student/best
        output_dir = args.output_dir  # model/nlvr2/student
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(student_model, 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))
        '''
        03/28/2022 01:15:52 - INFO - __main__ - Saving the best model checkpoint epoch 9 to model/nlvr2/student
        '''

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    eval_task_names = (args.task_name,)  # ('nlvr',)
    eval_outputs_dirs = (args.output_dir,)  # ('results/nlvr2',)

    results = {}
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # 'nlvr' 'results/nlvr2'
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 32 * 1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     # num_workers=args.workers,  # 默认为0
                                     sampler=eval_sampler,  # SequentialSampler(eval_dataset)
                                     batch_size=args.eval_batch_size)  # 32
        # print(eval_dataloader)  # 6982/32=219
        '''
        [
            [
                torch.Size([16, 2, 128]),
                torch.Size([16, 2, 178]),
                torch.Size([16, 2, 128]),
                torch.Size([16, 1]),
                torch.Size([16, 2, 50, 2054]),
                torch.Size([16, 1])
            ],
            [
                torch.Size([16, 2, 128]),
                torch.Size([16, 2, 178]),
                torch.Size([16, 2, 128]),
                torch.Size([16, 1]),
                torch.Size([16, 2, 50, 2054]),
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
        logger.info("  Num examples = %d", len(eval_dataset))  # 6982
        logger.info("  Batch size = %d", args.eval_batch_size)  # 32
        '''
        03/27/2022 20:14:51 - INFO - __main__ - ***** Running evaluation 5399 *****
        03/27/2022 20:14:51 - INFO - __main__ -   Num examples = 6982
        03/27/2022 20:14:51 - INFO - __main__ -   Batch size = 32
        '''
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # num_data = 0
        correct_num = 0
        # results_dict = {}

        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):  # 0~218
            # print(batch)
            '''
            [
                torch.Size([16, 2, 128]),  # input_ids
                torch.Size([16, 2, 178]),  # input_mask
                torch.Size([16, 2, 128]),  # segment_ids
                torch.Size([16, 1]),  # label_id
                torch.Size([16, 2, 50, 2054]),  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''
            # print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape, batch[5].shape)

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],  # input_ids: (16, 2, 128)
                    'attention_mask': batch[1],  # input_mask: (16, 2, 178)
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 2, 128)
                    'labels': batch[3],  # label_id: (16, 1)
                    'img_feats': None if args.img_feature_dim == -1 else batch[4]  # img_feat: (16, 2, 50, 2054)
                }
                outputs = model(**inputs)
                # print(outputs)
                '''
                (
                    loss: torch(数),
                    logits: (16, 2),
                    hidden_states: tuple(13个(32, 178, 768)),
                    attentions: tuple(12个(32, 12, 178, 178))
                )
                '''
                tmp_eval_loss, logits = outputs[:2]
                # print(tmp_eval_loss, logits.shape)  # tensor(数) (16, 2)

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
        03/27/2022 20:15:22 - INFO - __main__ -   Eval Accuracy: 65.62589515898023
        03/27/2022 20:15:22 - INFO - __main__ -   EVALERR: 65.62589515898023%
        '''
        results.update({"acc": acc})

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
    03/27/2022 20:15:22 - INFO - __main__ -   Eval Time Cost: 30.432
    '''

    return results, acc


def test(args, model, eval_dataset=None, prefix=""):
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    eval_task_names = (args.task_name,)  # ('nlvr',)
    eval_outputs_dirs = (args.output_dir,)  # ('results/nlvr2',)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # nlvr results/nlvr2
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 32 * 1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,  # SequentialSampler(eval_dataset)
                                     batch_size=args.eval_batch_size)  # 32
        # print(eval_dataloader)  # 6967/32=218
        '''
        [
            [
                torch.Size([16, 2, 128]),
                torch.Size([16, 2, 178]),
                torch.Size([16, 2, 128]),
                torch.Size([16, 1]),
                torch.Size([16, 2, 50, 2054]),
                torch.Size([16, 1])
            ],
            [
                torch.Size([16, 2, 128]),
                torch.Size([16, 2, 178]),
                torch.Size([16, 2, 128]),
                torch.Size([16, 1]),
                torch.Size([16, 2, 50, 2054]),
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
        logger.info("  Num examples = %d", len(eval_dataset))  # 6967
        logger.info("  Batch size = %d", args.eval_batch_size)  # 32
        '''
        
        '''

        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):  # 0~217
            # print(batch)
            '''
            [
                torch.Size([16, 2, 128]),  # input_ids
                torch.Size([16, 2, 178]),  # input_mask
                torch.Size([16, 2, 128]),  # segment_ids
                torch.Size([16, 1]),  # label_id
                torch.Size([16, 2, 50, 2054]),  # img_feat
                torch.Size([16, 1])  # q_id
            ]
            '''

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],  # input_ids: (16, 2, 128)
                    'attention_mask': batch[1],  # input_mask: (16, 2, 178)
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # segment_ids: (16, 2, 128)
                    'labels': None,
                    'img_feats': None if args.img_feature_dim == -1 else batch[4]  # img_feat: (16, 2, 50, 2054)
                }
                outputs = model(**inputs)
                # print(outputs)
                '''
                (
                    logits: (16, 2),
                    hidden_states: tuple(13个(32, 178, 768)),
                    attentions: tuple(12个(32, 12, 178, 178))
                )
                '''
                logits = outputs[0]  # (16, 2)

                val, idx = logits.max(dim=1)  # (values: (16,), indices: (16,))
                # logger.info('idx: %s, batch[5]: %s' % (str(idx.shape), str(batch[5].shape)))  # idx为预测值, batch[5]为q_id: (16, 1)

                for i in range(idx.size(0)):  # idx.size(0)=16
                    result = {
                        'questionId': str(batch[6][i].item()),
                        'prediction': label2ans[eval_dataset.labels[idx[i].item()]]
                    }
                    results.append(result)

                    # if len(results) % 2000 == 0:
                    #     logger.info("PROGRESS: {}%".format(round(100 * len(results) / len(eval_dataset), 4)))
                    # logger.info('q_id: {0}, answer: {1}'.format(result['question_id'], result['answer']))

        with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:  # results/nlvr2/best/xxx_results.json
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
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    # Text
    # parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    # parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")
    # parser.add_argument("--data_label_type", default='faster', type=str, help="faster or mask")
    parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or unbal or all")
    parser.add_argument("--test_data_type", default='bal', type=str, help="bal or unbal or all")

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
    parser.add_argument("--loss_type", default='ce', type=str, help="bce or ce")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    # parser.add_argument("--use_img_layernorm", action='store_true', help="use_img_layernorm")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--optim", default='AdamW', type=str, help="optim: AdamW, Adamax")
    parser.add_argument("--use_pair", action='store_true', help="use_pair")
    parser.add_argument("--use_label_seq", action='store_true', help="use_label_seq")
    parser.add_argument("--num_choice", default=2, type=int, help="num_choice")

    # Other Parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
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
    parser.add_argument("--teacher_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam of Teacher model.")
    parser.add_argument("--student_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam of Student model.")
    parser.add_argument("--strategy", default="first", type=str, help="first | last | skip | both")

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
    03/27/2022 19:59:10 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
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
        "nlvr": NLVRProcessor,
        "vcr_q_a": VCR_Q_A_Processor,
        "vcr_qa_r": VCR_QA_R_Processor,
        "vcr_qar": VCR_QAR_Processor,
    }
    '''
    # print(output_modes)
    '''
    {
        "vqa_text": "classification",
        "vqa_text_a": "classification",
        "gqa": "classification",
        "nlvr": "classification",
        "vcr_q_a": "classification",
        "vcr_qa_r": "classification",
        "vcr_qar": "classification",
    }
    '''
    args.task_name = args.task_name.lower()  # nlvr
    """ PASS """
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    """ PASS """
    processor = processors[args.task_name]()  # NLVRProcessor
    args.output_mode = output_modes[args.task_name]  # "classification"
    label_list = processor.get_labels()  # None
    # print(label_list)  # 长度为2的list
    '''
    [0, 1]
    '''
    num_labels = len(label_list)  # 2
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))  # nlvr 2
    '''
    03/27/2022 19:59:10 - INFO - __main__ - Task Name: nlvr, #Labels: 2
    '''

    """ PASS """
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    """ PASS """

    args.model_type = args.model_type.lower()  # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (BertConfig, ImageBertForSequenceClassification, BertTokenizer)
    if args.use_pair:  # args.use_pair=True则使用ImageBertForMultipleChoice
        model_class = ImageBertForMultipleChoice  # (BertConfig, ImageBertForMultipleChoice, BertTokenizer)
    tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
        args.teacher_model,  # model/nlvr2/teacher
        do_lower_case=args.do_lower_case  # True
    )
    teacher_config = config_class.from_pretrained(  # BertConfig
        args.teacher_model,  # model/nlvr2/teacher
        num_labels=num_labels,  # 2
        finetuning_task=args.task_name  # nlvr
    )
    student_config = config_class.from_pretrained(
        args.student_model,  # pretrained_models/base-vg-labels/ep_107_1192087
        num_hidden_layers=args.num_hidden_layers,  # 6
        num_labels=num_labels,  # 2
        finetuning_task=args.task_name  # nlvr
    )

    # new config: discrete code
    teacher_config.img_feature_dim = args.img_feature_dim  # 2054
    teacher_config.img_feature_type = args.img_feature_type  # faster_r-cnn
    teacher_config.code_voc = args.code_voc  # 512
    teacher_config.hidden_dropout_prob = args.drop_out  # 0.3
    teacher_config.loss_type = args.loss_type  # ce
    teacher_config.classifier = args.classifier  # mlp
    teacher_config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # teacher_config.use_img_layernorm = args.use_img_layernorm  # False
    teacher_config.num_choice = args.num_choice  # 2

    student_config.img_feature_dim = args.img_feature_dim  # 2054
    student_config.img_feature_type = args.img_feature_type  # faster_r-cnn
    student_config.code_voc = args.code_voc  # 512
    student_config.hidden_dropout_prob = args.drop_out  # 0.3
    student_config.loss_type = args.loss_type  # ce
    student_config.classifier = args.classifier  # mlp
    student_config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # student_config.use_img_layernorm = args.use_img_layernorm  # False
    student_config.num_choice = args.num_choice  # 2

    # """ PASS """
    # # load discrete code
    # if args.img_feature_type in ['dis_code', 'dis_code_t']:
    #     logger.info('Load discrete code from: {}'.format(args.data_dir))
    #     t_start = time.time()
    #     train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
    #     t_end = time.time()
    #     logger.info('Load time: %.3f' % (t_end - t_start))
    #
    #     if args.code_level == 'top':
    #         config.code_dim = train_code['embeddings_t'].shape[0]
    #         config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
    #     elif args.code_level == 'bottom':
    #         config.code_dim = train_code['embeddings_b'].shape[0]
    #         config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
    #     elif args.code_level == 'both':
    #         config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]
    # """ PASS """

    teacher_model = model_class.from_pretrained(  # ImageBertForMultipleChoice
        args.teacher_model,  # models/nlvr2/teacher
        from_tf=bool('.ckpt' in args.teacher_model),  # False
        config=teacher_config
    )
    student_model = model_class.from_pretrained(  # ImageBertForMultipleChoice
        args.teacher_model,  # pretrained_models/base-vg-labels/ep_107_1192087
        from_tf=bool('.ckpt' in args.teacher_model),  # False
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
    03/29/2022 21:58:50 - INFO - __main__ - Teacher Model Parameters: 114606338
    03/29/2022 21:58:50 - INFO - __main__ - Student Model Parameters: 72079106
    '''

    # ###########
    # for n, p in teacher_model.named_parameters():
    #     print(n)
    # for n, p in student_model.named_parameters():
    #     print(n)
    # ###########

    teacher_model.to(args.device)  # device(type='cuda')
    student_model.to(args.device)

    logger.info("Training/Evaluation parameters %s", args)
    '''
    03/29/2022 21:58:50 - INFO - __main__ - Training/Evaluation parameters Namespace(
        adam_epsilon=1e-08, alpha=0.5, c
        ache_dir='', classifier='mlp', cls_hidden_scale=3, code_level='top', code_voc=512, config_name='', 
        data_dir='datasets/nlvr2', device=device(type='cuda'), do_eval=False, do_lower_case=True, do_test=False, do_train=True, drop_out=0.3, 
        eval_all_checkpoints=False, eval_data_type='all', evaluate_during_training=True, fp16=False, fp16_opt_level='O1', 
        gradient_accumulation_steps=1, 
        img_feature_dim=2054, img_feature_type='faster_r-cnn', 
        local_rank=-1, logging_steps=-1, loss_type='ce', 
        max_grad_norm=1.0, max_img_seq_length=40, max_seq_length=55, max_steps=-1, model_type='bert', 
        n_gpu=1, no_cuda=False, num_choice=2, num_hidden_layers=6, num_train_epochs=10.0, 
        optim='AdamW', output_dir='model/nlvr2/student', output_mode='classification', overwrite_cache=False, overwrite_output_dir=False, 
        per_gpu_eval_batch_size=32, per_gpu_train_batch_size=32, philly=False, 
        save_after_epoch=-1, save_epoch=1, save_steps=-1, scheduler='linear', seed=88, server_ip='', server_port='', 
        strategy='skip', student_learning_rate=3e-05, student_model='pretrained_models/base-vg-labels/ep_107_1192087', 
        task_name='nlvr', teacher_learning_rate=3e-05, teacher_model='model/nlvr2/teacher', temperature=5.0, test_data_type='all', tokenizer_name='', 
        use_label_seq=True, use_pair=True, 
        warmup_steps=10000, weight_decay=0.05, workers=4)

    '''

    # load image features
    img_features = _load_img_features(args)
    # print(img_features)  # 119354
    '''
    {
        ...,
        'nlvr2_test1-999-3-img0': tensor([[0.2093, 0.1726, 0.0000,  ..., 0.8653, 0.3037, 0.5747],
                                          [0.0487, 0.0000, 0.0314,  ..., 0.9089, 0.9983, 0.8539],
                                          [0.7593, 0.0000, 0.0000,  ..., 0.4331, 0.2360, 0.1643],
                                          ...,
                                          [0.0000, 0.0000, 0.0000,  ..., 0.7089, 0.2960, 0.1254],
                                          [0.0000, 0.0000, 0.0000,  ..., 0.8396, 0.2864, 0.1631],
                                          [0.0000, 0.0000, 0.0000,  ..., 0.4790, 0.2997, 0.1500]]), 
        'nlvr2_test1-999-3-img1': tensor([[0.6039, 2.2122, 0.0000,  ..., 0.7338, 0.4896, 0.4970],
                                          [0.0000, 2.9828, 0.0000,  ..., 0.7337, 0.2934, 0.5637],
                                          [0.0000, 0.0000, 0.0000,  ..., 0.7202, 0.2274, 0.0714],
                                          ...,
                                          [0.0000, 0.0000, 0.0000,  ..., 0.3436, 0.1675, 0.1095],
                                          [1.4307, 5.4042, 0.0000,  ..., 0.7686, 0.2727, 0.5442],
                                          [0.0000, 0.3192, 0.0000,  ..., 0.6729, 0.2998, 0.2141]])
    }
    '''

    # Training (on 'train' set)
    if args.do_train:
        train_dataset = NLVRDataset(args, 'train', tokenizer, img_features)  # 构建训练集'train'
        eval_dataset = NLVRDataset(args, 'val', tokenizer, img_features)  # 构建验证集'val'
        '''
        NLVRDataset(Dataset): 初始化参数为args, name, tokenizer, img_features
            self.args: args
            self.name: 'train'或'val'或'test1'
            self.tokenizer: tokenizer
            
            self.output_mode: "classification"
            
            self.img_features: {
                                    ...,
                                    'nlvr2_test1-999-3-img0': tensor([[0.2093, 0.1726, 0.0000,  ..., 0.8653, 0.3037, 0.5747],
                                                                      [0.0487, 0.0000, 0.0314,  ..., 0.9089, 0.9983, 0.8539],
                                                                      [0.7593, 0.0000, 0.0000,  ..., 0.4331, 0.2360, 0.1643],
                                                                      ...,
                                                                      [0.0000, 0.0000, 0.0000,  ..., 0.7089, 0.2960, 0.1254],
                                                                      [0.0000, 0.0000, 0.0000,  ..., 0.8396, 0.2864, 0.1631],
                                                                      [0.0000, 0.0000, 0.0000,  ..., 0.4790, 0.2997, 0.1500]]), 
                                    'nlvr2_test1-999-3-img1': tensor([[0.6039, 2.2122, 0.0000,  ..., 0.7338, 0.4896, 0.4970],
                                                                      [0.0000, 2.9828, 0.0000,  ..., 0.7337, 0.2934, 0.5637],
                                                                      [0.0000, 0.0000, 0.0000,  ..., 0.7202, 0.2274, 0.0714],
                                                                      ...,
                                                                      [0.0000, 0.0000, 0.0000,  ..., 0.3436, 0.1675, 0.1095],
                                                                      [1.4307, 5.4042, 0.0000,  ..., 0.7686, 0.2727, 0.5442],
                                                                      [0.0000, 0.3192, 0.0000,  ..., 0.6729, 0.2998, 0.2141]])
                                }
            self.examples:  [
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
            self.labels: [0, 1]
            self.label_map: {
                                0: 0,
                                1: 1
                            }
            
            def __getitem__(index): # 对每一个self.examples[index]进行处理后得到:
                                    (
                                        input_ids: (2, 128),
                                        input_mask: (2, 178),
                                        segment_ids: (2, 128),
                                        label_id: (1,),
                                        img_feat: (2, 50, 2054),
                                        q_id: (1,)
                                    )
            def __len__(): len(self.examples)
        '''

        # 在训练集'train'上做Training, 在验证集'val'上做Evaluation
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        '''
        
        '''

    # # Evaluation (on 'val' set)
    # # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     eval_dataset = GQADataset(args, 'val', tokenizer, img_features)  # 构建验证集'val'
    #
    #     checkpoints = [args.output_dir]  # ["results/gqa/best"]
    #     """ PASS """
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     """ PASS """
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     '''
    #     04/02/2022 13:31:38 - INFO - __main__ - Evaluate the following checkpoints: ['model/gqa/teacher']
    #     '''
    #     for checkpoint in checkpoints:  # results/gqa/best
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint, config=config)
    #         model.to(args.device)
    #         result, score = evaluate(args, model, eval_dataset, prefix=global_step)
    #         # result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         # results.update(result)

    # """ PASS """
    # # Testing (on 'test-dev' set)
    # if args.do_test_dev and args.local_rank in [-1, 0]:
    #     test_dev_dataset = GQADataset(args, 'test-dev', tokenizer, img_features)  # 构建测试集'test-dev'
    #
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint, config=config)
    #         model.to(args.device)
    #         result, score = evaluate(args, model, test_dev_dataset, prefix=global_step)
    #         # test(args, model, test_dev_dataset, prefix=global_step)
    # """ PASS """

    # # Testing (on 'test' set)
    # if args.do_test and args.local_rank in [-1, 0]:
    #     test_dataset = GQADataset(args, 'test', tokenizer, img_features)  # 构建测试集'test'
    #
    #     checkpoints = [args.output_dir]  # ["results/gqa/best"]
    #     """ PASS """
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     """ PASS """
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     '''
    #
    #     '''
    #     for checkpoint in checkpoints:  # "results/gqa/best"
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         test(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
