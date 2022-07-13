# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import os
import sys
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')  # 新添加的目录会优先于其他目录被import检查

from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers.pytorch_transformers import WEIGHTS_NAME

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from oscar.utils.misc import set_seed, mkdir
from oscar.utils.logger import setup_logger
from oscar.utils.task_utils import _truncate_seq_pair

#
import warnings

warnings.filterwarnings('ignore')


#


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""

    def __init__(self, args, tokenizer, split='train', is_train=True):
        super(RetrievalDataset, self).__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.output_mode = args.output_mode
        self.is_train = is_train

        feature_file = os.path.join(args.data_dir, '{}_img_{}_feats.pt'.format(split, args.img_feature_type))  # datasets/coco_ir/train_img_frcnn_feats.pt或datasets/coco_ir/minival_img_frcnn_feats.pt或datasets/coco_ir/test_img_frcnn_feats.pt
        self.features = torch.load(feature_file)
        # print(self.features)  # 113287 或 1000 或 5000
        '''
        {
            ...,
            134574: tensor([[ 0.0000,  2.1463,  0.0000,  ...,  0.9983,  0.7955,  0.9215],
                            [ 0.5593,  4.6677,  0.0000,  ...,  0.9983,  0.9056,  0.4651],
                            [ 1.5542,  9.7955,  1.2252,  ...,  0.4101,  0.0633,  0.2359],
                            ...,
                            [ 0.0000,  0.0896,  0.0000,  ...,  0.1881,  0.0950,  0.1881],
                            [ 0.7405,  3.7327,  0.0000,  ...,  0.4878,  0.6200,  0.3560],
                            [ 0.8907, 12.7547,  0.4234,  ...,  0.6314,  0.1375,  0.5062]]),
            418825: tensor([[3.4156e-01, 4.0807e-01, 0.0000e+00,  ..., 9.9875e-01, 9.1667e-01, 4.7146e-01],
                            [2.4551e-02, 2.3292e-02, 0.0000e+00,  ..., 7.6137e-01, 9.9833e-01, 5.7487e-01],
                            [1.3012e-01, 9.1158e-01, 0.0000e+00,  ..., 9.6293e-01, 6.8800e-01, 7.1294e-01],
                            ...,
                            [1.2698e-02, 6.2869e-01, 3.4424e-01,  ..., 8.6048e-01, 6.4186e-02, 5.3734e-02],
                            [4.6156e-01, 1.0837e-01, 8.9283e-04,  ..., 2.3653e-01, 4.0641e-01, 2.3653e-01],
                            [2.2118e+00, 3.3961e+00, 2.7367e-02,  ..., 5.0338e-01, 8.1081e-02, 8.9018e-02]])
        }
        '''

        caption_file = os.path.join(args.data_dir, '{}_captions.pt'.format(split))  # datasets/coco_ir/train_captions.pt或datasets/coco_ir/minival_captions.pt或datasets/coco_ir/test_captions.pt
        self.captions = torch.load(caption_file)  # 113287 或 1000 或 5000
        # print(self.captions)
        '''
        {
            ...,
            134574: '[
                "A table topped with four plates filled with food.", 
                "A number of plates with food and a glass on the table", 
                "A number of plates with food, spoon and glass", 
                "A variety of food on a dining room table. ", 
                "Plates of food on a table at a restaurant ", 
                "A picture of a couple plates of someone\'s lunch sitting on a table."
            ]'，
            418825: '[
                "Fruits and vegetables lay on a counter to prepare a meal.", 
                "A bag of strawberries on a table with tomatoes.", 
                "a bunch of food is laying out on a table", 
                "A table with several varieties of fruits and vegetables as well as flowers on top of it.", 
                "This kitchen table has fruits and vegetables on it"
            ]'
        }
        '''

        self.img_keys = list(self.features.keys())  # 长度为113287的list 或 长度为1000的list 或 长度为5000的list
        # print(self.img_keys)
        '''
        [57870, 384029, 222016, 520950, ..., 134574, 418825]
        '''

        if not type(self.captions[self.img_keys[0]]) == list:  # str != list
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}  # '[]'转[]
            # print(self.captions)  # 113287 或 1000 或 5000
            '''
            {
                ...,
                134574: [
                    "A table topped with four plates filled with food.", 
                    "A number of plates with food and a glass on the table", 
                    "A number of plates with food, spoon and glass", 
                    "A variety of food on a dining room table. ", 
                    "Plates of food on a table at a restaurant ", 
                    "A picture of a couple plates of someone\'s lunch sitting on a table."
                ]，
                418825: [
                    "Fruits and vegetables lay on a counter to prepare a meal.", 
                    "A bag of strawberries on a table with tomatoes.", 
                    "a bunch of food is laying out on a table", 
                    "A table with several varieties of fruits and vegetables as well as flowers on top of it.", 
                    "This kitchen table has fruits and vegetables on it"
                ]
            }
            '''
        assert len(self.features) == len(self.captions), "the length of image features and captions does not match!"

        if args.add_od_labels:  # args.add_od_labels=True
            label_file = os.path.join(args.data_dir, '{}_{}_labels.pt'.format(split, args.od_label_type))  # datasets/coco_ir/train_vg_labels.pt或datasets/coco_ir/minival_vg_labels.pt或datasets/coco_ir/test_vg_labels.pt
            self.labels = torch.load(label_file)  # 113287 或 1000 或 5000
            # print(self.labels)
            '''
            {
                ...,
                394940: 'table boy wall child mouth boy plate girl door face shirt head person nose eyes hand eye eye toothbrush knife handle watch handle hair eyebrow shirt hand hair food', 
                15335: 'wall people man woman bracelet head shirt watch man man napkin glass hair shirt shirt man head head man napkin hair shirt wall seat person hand hand man wrist'
            }
            '''

        if is_train:  # is_train=True
            self.num_captions_per_img = args.num_captions_per_img_train  # 5
        else:  # is_train=False
            self.num_captions_per_img = args.num_captions_per_img_val  # 20
            if args.eval_img_keys_file:  # test_img_keys_1k.tsv或test_img_keys.tsv
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(os.path.join(args.data_dir, args.eval_img_keys_file), 'r') as f:  # datasets/coco_ir/test_img_keys_1k.tsv或datasets/coco_ir/test_img_keys.tsv
                    img_keys = f.readlines()
                    # print(img_keys)  # 长度为1000的list 或 长度为5000的list
                    '''
                    ['317441\n', '415746\n', '45057\n', 444444\n', ..., '370678\n', '114684']   
                    '''
                self.img_keys = [int(k.strip()) for k in img_keys]  # 长度为1000的list 或 长度为5000的list
                # print(self.img_keys)
                '''
                [317441, 415746, 45057, 444444, ..., 370678, 114684]
                '''
                self.features = {k: self.features[k] for k in self.img_keys}
                # print(self.features)  # 1000 或 5000
                '''
                {
                    317441: (x, 2054),
                    ...,
                    114684: (x, 2054)
                }
                '''
                self.captions = {k: self.captions[k] for k in self.img_keys}
                # print(self.captions)  # 1000 或 5000
                '''
                {
                    317441: [
                        "A table topped with four plates filled with food.", 
                        "A number of plates with food and a glass on the table", 
                        "A number of plates with food, spoon and glass", 
                        "A variety of food on a dining room table. ", 
                        "Plates of food on a table at a restaurant ", 
                        "A picture of a couple plates of someone\'s lunch sitting on a table."
                    ],
                    ...
                    114684: [
                        "Fruits and vegetables lay on a counter to prepare a meal.", 
                        "A bag of strawberries on a table with tomatoes.", 
                        "a bunch of food is laying out on a table", 
                        "A table with several varieties of fruits and vegetables as well as flowers on top of it.", 
                        "This kitchen table has fruits and vegetables on it"
                    ]
                }
                '''
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}
                    # print(self.labels)  # 1000或5000
                    '''
                    {
                        317441: 'table boy wall child mouth boy plate girl door face shirt head person nose eyes hand eye eye toothbrush knife handle watch handle hair eyebrow shirt hand hair food', 
                        ...,
                        114684: 'wall people man woman bracelet head shirt watch man man napkin glass hair shirt shirt man head head man napkin hair shirt wall seat person hand hand man wrist'
                    }
                    '''

            if args.eval_caption_index_file:  # 指定了args.eval_caption_index_file=minival_caption_indexs_top20.pt则self.has_caption_indexs=True
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval  # 指定了args.eval_caption_index_file时需保证args.cross_image_eval=False
                caption_index_file = os.path.join(args.data_dir, args.eval_caption_index_file)  # datasets/coco_ir/minival_caption_indexs_top20.pt
                self.caption_indexs = torch.load(caption_index_file)
                # print(self.caption_indexs)
                '''
                {
                    184613: '[[184613, 1], [184613, 0], [414795, 3], [184613, 4], [184613, 3], [184613, 2], [57265, 4], [24097, 0], [173574, 3], [102159, 4], [24097, 2], [419144, 0], [24097, 1], [230454, 2], [423008, 2], [423008, 3], [358149, 3], [57265, 2], [173574, 2], [342146, 3]]', 
                    ...,
                    379842: '[[379842, 3], [379842, 4], [379842, 0], [379842, 2], [437564, 1], [140167, 4], [271986, 0], [304389, 3], [7320, 0], [322816, 0], [379842, 1], [265611, 1], [271986, 4], [571196, 2], [322816, 4], [265611, 4], [265611, 2], [29913, 2], [58254, 1], [257458, 1]]'
                }
                '''
                if not type(self.caption_indexs[self.img_keys[0]]) == list:  # str != list
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}  # '[]'转[]
                    # print(self.caption_indexs)
                    '''
                    {
                        184613: [[184613, 1], [184613, 0], [414795, 3], [184613, 4], [184613, 3], [184613, 2], [57265, 4], [24097, 0], [173574, 3], [102159, 4], [24097, 2], [419144, 0], [24097, 1], [230454, 2], [423008, 2], [423008, 3], [358149, 3], [57265, 2], [173574, 2], [342146, 3]], 
                        ...,
                        379842: [[379842, 3], [379842, 4], [379842, 0], [379842, 2], [437564, 1], [140167, 4], [271986, 0], [304389, 3], [7320, 0], [322816, 0], [379842, 1], [265611, 1], [271986, 4], [571196, 2], [322816, 4], [265611, 4], [265611, 2], [29913, 2], [58254, 1], [257458, 1]]
                    }
                    '''
            else:  # 未指定args.eval_caption_index_file则self.has_caption_indexs=False
                self.has_caption_indexs = False

    def get_image_caption_index(self, index):  # 0
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:  # is_train=False且cross_image_eval=True
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))  # index // (5 * 1000)或index // (5 * 5000)
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))  # index % (5 * 1000)或index // (5 * 5000)
            img_idx1 = cap_idx // self.num_captions_per_img  # (index % (5 * 1000)) // 5或(index // (5 * 5000)) // 5
            cap_idx1 = cap_idx % self.num_captions_per_img  # index % (5 * 1000)) % 5或(index // (5 * 5000)) % 5
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:  # is_train=False且has_caption_indexs=True
            img_idx = index // self.num_captions_per_img  # index // 5或index // 5
            cap_idx = index % self.num_captions_per_img  # index % 5或index % 5
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key):
        if self.args.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]  # 'table boy wall child mouth boy plate girl door face shirt head person nose eyes hand eye eye toothbrush knife handle watch handle hair eyebrow shirt hand hair food'
            else:
                od_labels = ' '.join([l['class'] for l in self.labels[img_key]])
            return od_labels

    def tensorize_example(self,
                          text_a,  # "Fruits and vegetables lay on a counter to prepare a meal."
                          img_feat,  # (39, 2054)
                          text_b=None,  # 'wall people man woman bracelet head shirt watch man man napkin glass hair shirt shirt man head head man napkin hair shirt wall seat person hand hand man wrist'
                          cls_token_segment_id=0,
                          pad_token_segment_id=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)  # ['fruits', 'and', 'vegetables', 'lay', 'on', 'a', 'counter', 'to', 'prepare', 'a', 'meal', '.']

        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)

        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.args.max_seq_length - len(tokens) - 1:
                tokens_b = tokens_b[: (self.args.max_seq_length - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.args.max_seq_length - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]  # 39
        if img_len > self.args.max_img_seq_length:
            img_feat = img_feat[0: self.args.max_img_seq_length, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:  # 39 < 50
            img_padding_len = self.args.max_img_seq_length - img_len  # 50-39=11
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))  # (11, 2054)
            img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((39, 2054), (11, 2054), dim=0) -> (50, 2054)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type  # args.att_mask_type=CLR
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + [1] * img_len + [0] * img_padding_len  # [1]*44 + [0]*26 + [1]*39 + [0]*11
        else:
            # use 2D mask to represent the attention
            max_len = self.args.max_seq_length + self.args.max_img_seq_length
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.args.max_seq_length, self.args.max_seq_length + img_len
            attention_mask[c_start: c_end, c_start: c_end] = 1
            attention_mask[l_start: l_end, l_start: l_end] = 1
            attention_mask[r_start: r_end, r_start: r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start: c_end, l_start: l_end] = 1
                attention_mask[l_start: l_end, c_start: c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start: c_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, c_start: c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start: l_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, l_start: l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)  # (70,)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)  # (120,)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)  # (70,)
        return input_ids, attention_mask, segment_ids, img_feat

    def __getitem__(self, index):
        if self.is_train:  # is_train=True
            img_idx, cap_idxs = self.get_image_caption_index(index)  # 113287 [15335, 0]
            img_key = self.img_keys[img_idx]  # 113287 -> 418825
            feature = self.features[img_key]  # (39, 2054)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]  # "Fruits and vegetables lay on a counter to prepare a meal."
            od_labels = self.get_od_labels(img_key)  # 'wall people man woman bracelet head shirt watch man man napkin glass hair shirt shirt man head head man napkin hair shirt wall seat person hand hand man wrist'
            example = self.tensorize_example(text_a=caption, img_feat=feature, text_b=od_labels)

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)
            if random.random() <= 0.5:
                # randomly select a negative caption from a different image.
                cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                example_neg = self.tensorize_example(text_a=caption_neg, img_feat=feature, text_b=od_labels)
            else:
                # randomly select a negative image 
                feature_neg = self.features[self.img_keys[img_idx_neg]]
                od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                example_neg = self.tensorize_example(text_a=caption, img_feat=feature_neg, text_b=od_labels_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair  # index, ([input_ids, attention_mask, segment_ids. img_feat, label, input_ids, attention_mask, segment_ids. img_feat, label])
        else:  # is_train=False
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature = self.features[img_key]
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(caption, feature, text_b=od_labels)

            label = 1 if img_key == cap_idxs[0] else 0
            return index, tuple(list(example) + [label])  # index, (input_ids, attention_mask, segment_ids. img_feat, label)

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:  # is_train=False且cross_image_eval=True
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.img_keys) * self.num_captions_per_img


def compute_score_with_logits(logits, labels):  # (32, 2) (32, 1)
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data  # argmax
        scores = logits == labels
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores


def compute_ranks(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while save_num < 10:
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, student_model, teacher_model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # 16 * 1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=args.num_workers)
    # print(train_dataloader)
    '''
    [
        [
            torch.Size([16, 128]),  # input_ids
            torch.Size([16, 178]),  # attention_mask
            torch.Size([16, 128]),  # segment_ids
            torch.Size([16, 50, 2054]),  # img_feat
            torch.Size([16, 1])  # label
        ],
        [
            torch.Size([16, 128]),
            torch.Size([16, 178]),
            torch.Size([16, 128]),
            torch.Size([16, 50, 2054]),
            torch.Size([16, 1])
        ],
        ...
    ]
    '''

    if args.max_steps > 0:  # args.max_steps=-1
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and scheduler
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
                          lr=args.learning_rate,  # 2e-5
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
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    '''
    2022-03-31 19:38:01,177 vlpretrain INFO: ***** Running training *****
    2022-03-31 19:38:01,186 vlpretrain INFO:   Num examples = 566435
    2022-03-31 19:38:01,189 vlpretrain INFO:   Num Epochs = 20
    2022-03-31 19:38:01,189 vlpretrain INFO:   Batch size per GPU = 64
    2022-03-31 19:38:01,191 vlpretrain INFO:   Total train batch size (w. parallel, & accumulation) = 64
    2022-03-31 19:38:01,191 vlpretrain INFO:   Gradient Accumulation steps = 1
    2022-03-31 19:38:01,191 vlpretrain INFO:   Total optimization steps = 177020
    '''

    global_step = 0
    tr_loss = 0.0
    train_acc = 0.0
    student_model.zero_grad()

    log_json = []

    best_score = 0

    #
    # Prepare loss function
    mse_loss_fn = nn.MSELoss()

    def soft_cross_entropy(predictions, targets):
        student_likelihood = F.log_softmax(predictions, dim=-1)
        targets_probs = F.softmax(targets, dim=-1)
        return (-targets_probs * student_likelihood).mean()

    #

    for epoch in range(int(args.num_train_epochs)):

        for step, (_, batch) in enumerate(train_dataloader):
            # print(batch)
            '''
            [
                torch.Size([16, 128]),  # input_ids
                torch.Size([16, 178]),  # input_mask
                torch.Size([16, 128]),  # segment_ids
                torch.Size([16, 50, 2054]),  # img_feat
                torch.Size([16, 1]),  # label_id
                torch.Size([16, 128]),  # input_ids
                torch.Size([16, 178]),  # input_mask
                torch.Size([16, 128]),  # segment_ids
                torch.Size([16, 50, 2054]),  # img_feat
                torch.Size([16, 1]),  # label_id
            ]
            '''

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': torch.cat((batch[0], batch[5]), dim=0),  # input_ids: (32, 70)
                'attention_mask': torch.cat((batch[1], batch[6]), dim=0),  # input_mask: (32, 120)
                'token_type_ids': torch.cat((batch[2], batch[7]), dim=0),  # segment_ids: (32, 70)
                'img_feats': torch.cat((batch[3], batch[8]), dim=0),  # img_feats: (32, 50, 2054)
                'labels': torch.cat((batch[4], batch[9]), dim=0)  # label_id: (32, 1)
            }

            # TinyBERT/BERT-EMD
            student_model.train()
            teacher_model.eval()

            task_specific_loss, student_logits, student_reps, student_atts = student_model(input_ids=inputs['input_ids'],  # (32, 70)
                                                                                           token_type_ids=inputs['token_type_ids'],  # (32, 70)
                                                                                           attention_mask=inputs['attention_mask'],  # (32, 120)
                                                                                           labels=inputs['labels'],  # (32, 2)
                                                                                           img_feats=inputs['img_feats'])  # 新增: (32, 50, 2054)
            # print(task_specific_loss)  # torch(数)
            # print(student_logits)  # (32, 2)
            # print(student_reps)  # tuple(7个(16, 178, 768))或tuple(5个(16, 178, 768))
            # print(student_atts)  # list(6个(16, 12, 178, 178))或tuple(4个(16, 12, 178, 178))

            with torch.no_grad():
                teacher_logits, teacher_reps, teacher_atts = teacher_model(input_ids=inputs['input_ids'],  # (32, 70)
                                                                           token_type_ids=inputs['token_type_ids'],  # (32, 70)
                                                                           attention_mask=inputs['attention_mask'],  # (32, 120))
                                                                           img_feats=inputs['img_feats'])  # 新增: (32, 50, 2054)
            # print(teacher_logits)  # (32, )
            # print(teacher_reps)  # list(13个(32, 178, 768))
            # print(teacher_atts)  # list(12个(32, 12, 178, 178))

            teacher_layer_num = len(teacher_atts)  # 12
            student_layer_num = len(student_atts)  # 6或4
            assert teacher_layer_num % student_layer_num == 0  # 12%6=0或12%4=0
            layers_per_block = int(teacher_layer_num / student_layer_num)  # int(12/6)=2或int(12/4)=3

            att_loss = 0.  # 6层attention的MSE loss
            rep_loss = 0.  # embedding层的MSE loss和6层hidden_state的MSE loss
            # cls_loss = 0.  # prediction层的soft_cross_entropy loss

            # L_attn
            new_teacher_atts = [
                teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)  # 1,3,5,7,9,11: attentions的第2,4,6,8,10,12层 或 2,5,8,11: attention的第3,6,9,12层
            ]
            # print(new_teacher_atts)  # list(6个(16, 12, 178, 178))或list(4个(16, 12, 178, 178))
            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device), student_att)  # (16, 12, 178, 178)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device), teacher_att)  # (16, 12, 178, 178)
                tmp_loss = mse_loss_fn(student_att, teacher_att)  # MSE((16,12,178,178), (16,12,178,178))
                att_loss += tmp_loss

            # L_embd + L_hidn
            new_teacher_reps = [
                teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)  # 0,2,4,6,8,10,12: hidden_states的第1,3,5,7,9,11,13层(即embedding层+transformer的第2,4,6,8,10,12层) 或 0,3,6,9,12: hidden_states的第1,4,7,10,13层(即embedding层+transformer的第3,6,9,12层)
            ]
            # print(new_teacher_reps)  # list(7个(16, 178, 768))
            for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                tmp_loss = mse_loss_fn(student_rep, teacher_rep)  # MSE((16,178,768), (16,178,768))
                rep_loss += tmp_loss

            # L_pred = α * L_vanilla_kd + (1 - α) * L_task_specific
            vanilla_kd_loss = soft_cross_entropy(student_logits / args.temperature, teacher_logits / args.temperature)
            # task_specific_loss = nn.BCEWithLogitsLoss()(student_logits, inputs['labels'])
            cls_loss = (1 - args.alpha) * task_specific_loss + args.alpha * vanilla_kd_loss  # L(x;θ_s;θ_t)

            # L_s = β * (L_embd + L_hidn + L_attn) + L_pred
            loss = args.beta * (rep_loss + att_loss) + cls_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 1.0

            batch_score = compute_score_with_logits(student_logits, inputs['labels']).sum()
            batch_acc = batch_score.item() / (args.train_batch_size * 2)

            tr_loss += loss.item()
            train_acc += batch_acc
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # args.logging_steps=-1
            if global_step % args.logging_steps == 0:
                logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), score: {:.4f} ({:.4f})".format(
                    epoch + 1, global_step, optimizer.param_groups[0]["lr"], loss, tr_loss / global_step, batch_acc, train_acc / global_step)
                )

            # args.save_steps=-1
            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                save_checkpoint(student_model, tokenizer, args, epoch + 1, global_step)
                # evaluation
                if args.evaluate_during_training:
                    logger.info("Perform evaluation at step: %d" % global_step)
                    test_result = test(args, student_model, val_dataset)
                    eval_result = evaluate(val_dataset, test_result)
                    rank_accs = eval_result['i2t_retrieval']
                    if rank_accs['R@1'] > best_score:
                        best_score = rank_accs['R@1']
                    epoch_log = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'R1': rank_accs['R@1'],
                        'R5': rank_accs['R@5'],
                        'R10': rank_accs['R@10'],
                        'best_R1': best_score
                    }
                    log_json.append(epoch_log)
                    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                        json.dump(log_json, fp)

    return global_step, tr_loss / global_step


def test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers)

    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    '''
    2022-03-31 16:52:53,553 vlpretrain INFO: Num examples = 5000000(COCO 1k test set)或125000000(COCO 5k test set)
    2022-03-31 16:52:53,554 vlpretrain INFO: Evaluation batch size = 64
    '''
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats': batch[3],
                'labels': batch[4]
            }
            _, logits = model(**inputs)[:2]
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1]  # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return results


def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def get_predict_file(args):
    cc = []
    data = os.path.basename(os.path.join(args.data_dir, '')[:-1])  # 'datasets/coco_ir' ->'datasets/coco_ir\\' -> 'datasets/coco_ir' -> 'coco_ir'
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)  # ['test']
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))  # ['test', 'wlabelsvg']
    return os.path.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc)))  # ./output/test.wlablesvg.results.pt


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(os.path.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 'max_img_seq_length', 'add_od_labels', 'od_label_type']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param, test_v, train_v))
                '''
                2022-03-31 16:43:50,587 vlpretrain WARNING: Override do_lower_case with train args: False -> True
                2022-03-31 16:43:50,588 vlpretrain WARNING: Override add_od_labels with train args: False -> True
                '''
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False, help="The input data dir with all required files.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False, help="The output directory to save checkpoint and test results.")
    # parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default='pretrained_models/base-vg-labels/ep_67_588997', type=str, required=False, help="Path to pre-trained model or model type. required for training.")

    # Text

    # Image
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str, help="Image feature type.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, help="The maximum total input image sequence length.")

    # Dataset
    parser.add_argument("--num_workers", default=0, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int, help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int, help="number of captions for each testing image.")
    parser.add_argument("--att_mask_type", default='CLR', type=str, help="attention mask type, support ['CL', 'CR', 'LR', 'CLR'] C: caption, L: labels, R: image regions; CLR is full attention by default. CL means attention between caption and labels. please pay attention to the order CLR, which is the default concat order.")

    # Model configuration
    parser.add_argument("--loss_type", default='ce', type=str, help="Loss function types: support kl, ce")
    # parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    # parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--output_mode", default='classification', type=str, help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--optim", default='AdamW', type=str, help="AdamW or Adamax")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=70, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. This number is calculated on COCO dataset, If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation. do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")

    # Training
    parser.add_argument("--eval_caption_index_file", default='', type=str, help="index of a list of (img_key, cap_idx) for each image. this is used to perform re-rank using hard negative samples. useful for validation set to monitor the performance during training.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, help="label type, support vg, gt, oid")
    # Inference
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str, help="image key tsv to select a subset of images for evaluation. This is useful in 5-folds evaluation. The topn index file is not needed in this case.")
    parser.add_argument("--cross_image_eval", action='store_true', help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--eval_model_dir", type=str, default='', help="Model directory for evaluation.")

    #
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, required=True, help="The student model dir.")
    parser.add_argument('--alpha', default=0.5, type=float, help="Vanilla knowledge distillation loss radio.")
    parser.add_argument("--temperature", default=5.0, type=float, help="Distillation temperature for soft target.")
    parser.add_argument('--num_hidden_layers', default=6, type=int)
    # parser.add_argument("--teacher_learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam of Teacher model.")
    # parser.add_argument("--student_learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam of Student model.")
    # parser.add_argument("--strategy", default="first", type=str, help="first | last | skip | both")
    parser.add_argument('--beta', default=0.01, type=float, help="intermediate features radio.")

    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
    '''
    2022-03-31 19:37:34,902 vlpretrain WARNING: Device: cuda, n_gpu: 1
    2022-03-31 19:37:34,903 vlpretrain INFO: output_mode: classification, #Labels: 2
    '''

    config_class, model_class, tokenizer_class = BertConfig, ImageBertForSequenceClassification, BertTokenizer
    if args.do_train:  # args.do_train=True
        tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
            args.teacher_model,  # model/coco_ir/teacher
            do_lower_case=args.do_lower_case  # True
        )
        teacher_config = config_class.from_pretrained(  # BertConfig
            args.teacher_model,  # model/coco_ir/teacher
            num_labels=args.num_labels,  # 2
            finetuning_task='ir'
        )
        student_config = config_class.from_pretrained(  # BertConfig
            args.student_model,  # pretrained_models/base-vg-labels/ep_67_588997
            num_hidden_layers=args.num_hidden_layers,  # 6
            num_labels=args.num_labels,  # 2
            finetuning_task='ir'
        )

        teacher_config.img_feature_dim = args.img_feature_dim  # 2054
        teacher_config.img_feature_type = args.img_feature_type  # frcnn
        teacher_config.hidden_dropout_prob = args.drop_out  # 0.1
        teacher_config.loss_type = args.loss_type  # ce
        # 输出hidden_states和attentions
        teacher_config.output_hidden_states = True
        teacher_config.output_attentions = True

        student_config.img_feature_dim = args.img_feature_dim  # 2054
        student_config.img_feature_type = args.img_feature_type  # frcnn
        student_config.hidden_dropout_prob = args.drop_out  # 0.1
        student_config.loss_type = args.loss_type  # ce
        # 输出hidden_states和attentions
        student_config.output_hidden_states = True
        student_config.output_attentions = True

        teacher_model = model_class.from_pretrained(  # ImageBertForSequenceClassification
            args.teacher_model,  # model/coco_ir/teacher
            from_tf=bool('.ckpt' in args.teacher_model),  # False
            config=teacher_config
        )
        student_model = model_class.from_pretrained(  # ImageBertForSequenceClassification
            args.student_model,  # pretrained_models/base-vg-labels/ep_67_588997
            from_tf=bool('.ckpt' in args.student_model),  # False
            config=student_config
        )
    else:  # args.do_eval=True或args.do_test=True
        checkpoint = args.eval_model_dir  # ./output
        assert os.path.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        '''
        2022-03-31 16:43:48,267 vlpretrain INFO: Evaluate the following checkpoint: ./output
        '''
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info('Teacher Model Parameters: {}'.format(teacher_total_params))
    student_total_params = sum(p.numel() for p in student_model.parameters())
    logger.info('Student Model Parameters: {}'.format(student_total_params))

    '''
    2022-03-31 19:37:37,295 vlpretrain INFO: Teacher Model Parameters: 111062018
    
    '''

    teacher_model.to(args.device)
    student_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    '''
    
    '''

    # Training
    if args.do_train:
        train_dataset = RetrievalDataset(args, tokenizer, 'train', is_train=True)
        if args.evaluate_during_training:
            eval_dataset = RetrievalDataset(args, tokenizer, 'minival', is_train=False)
        else:
            eval_dataset = None
        global_step, tr_loss = train(args, train_dataset, eval_dataset, student_model, teacher_model, tokenizer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
        '''
        
        '''

    # # Inference and evaluation
    # if args.do_test or args.do_eval:
    #     args = restore_training_settings(args)
    #     test_dataset = RetrievalDataset(args, tokenizer, args.test_split, is_train=False)  # 'test'
    #     checkpoint = args.eval_model_dir  # ./output
    #     assert os.path.isdir(checkpoint)
    #     logger.info("Evaluate the following checkpoint: %s", checkpoint)  # ./output
    #     '''
    #     2022-03-31 16:43:51,189 vlpretrain INFO: Evaluate the following checkpoint: ./output
    #     '''
    #     model = model_class.from_pretrained(checkpoint, config=config)
    #     model.to(args.device)
    #     if args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)
    #
    #     pred_file = get_predict_file(args)  # './output/test.wlablesvg.results.pt'
    #     if os.path.isfile(pred_file):
    #         logger.info("Prediction file exist, skip inference.")
    #         if args.do_eval:
    #             test_result = torch.load(pred_file)
    #     else:
    #         test_result = test(args, model, test_dataset)
    #         torch.save(test_result, pred_file)
    #         logger.info("Prediction results saved to {}.".format(pred_file))
    #
    #     if args.do_eval:
    #         eval_result = evaluate(test_dataset, test_result)
    #         result_file = os.path.splitext(pred_file)[0] + '.eval.json'  # './output/test.wlablesvg.results.pt' -> ('./output/test.wlablesvg.results', '.pt') -> './output/test.wlablesvg.results.eval.json'
    #         with open(result_file, 'w') as f:
    #             json.dump(eval_result, f)
    #         logger.info("Evaluation results saved to {}.".format(result_file))
    #         '''
    #
    #         '''


if __name__ == "__main__":
    main()
