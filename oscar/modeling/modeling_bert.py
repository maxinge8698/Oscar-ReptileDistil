# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.pytorch_transformers.modeling_bert import (BertEmbeddings,
                                                             BertSelfAttention, BertSelfOutput, BertAttention,
                                                             BertIntermediate, BertOutput,
                                                             BertLayer,
                                                             BertEncoder,
                                                             BertModel,
                                                             BertPooler,
                                                             BertLayerNorm,
                                                             BertPreTrainedModel,
                                                             BertPredictionHeadTransform)
from .modeling_utils import CaptionPreTrainedModel
from ..utils.cbs import ConstrainedBeamSearch, select_best_beam_with_constraints

logger = logging.getLogger(__name__)


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self,
                hidden_states,  # (16, 178, 768)
                attention_mask,  # (16, 1, 1, 178)
                #
                head_mask=None,  # None
                history_state=None):  # None
        if history_state is not None:
            """ PASS """
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
            """ PASS """
        else:
            # Q, K, V: 178×768 (n×d)
            mixed_query_layer = self.query(hidden_states)  # (16, 178, 768) -> (16, 178, 768)
            mixed_key_layer = self.key(hidden_states)  # (16, 178, 768) -> (16, 178, 768)
            mixed_value_layer = self.value(hidden_states)  # (16, 178, 768) -> (16, 178, 768)

        # QW, KW, VW: 178×64 (n×d/h)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (16, 178, 768) -> (16, 12, 178, 64)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (16, 178, 768) -> (16, 12, 178, 64)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (16, 178, 768) -> (16, 12, 178, 64)

        # Take the dot product between "query" and "key" to get the raw attention scores: (n×d/h)(d/h×n)=(n×n)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (16, 12, 178, 64)(16, 12, 64, 178)=(16, 12, 178, 178)
        # (n×n) / sqrt(h)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # (16, 12, 178, 178)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # (16, 12, 178, 178)+(16, 1, 1, 178)=(16, 12, 178, 178)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (16, 12, 178, 178)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # (16, 12, 178, 178)

        # Mask heads if we want to
        if head_mask is not None:
            """ PASS """
            attention_probs = attention_probs * head_mask
            """ PASS """

        # Softmax(QK^T/sqrt(d_k))V: (n×d/h)
        context_layer = torch.matmul(attention_probs, value_layer)  # (16, 12, 178, 178)(16, 12, 178, 64)=(16, 12, 178, 64)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (16, 12, 178, 64) -> (16, 178, 12, 64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # torch.Size([16, 178])+(768,)=torch.Size([16, 178, 768])
        context_layer = context_layer.view(*new_context_layer_shape)  # (16, 178, 12, 64)转(16, 178, 768)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        # print(outputs)
        '''
        (
            context_layer: (16, 178, 768),
        )
        或
        (
            context_layer: (16, 178, 768),
            attention_probs: (16, 12, 178, 178)
        )
        '''
        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)

        self.self = CaptionBertSelfAttention(config)  # 修改BertSelfAttention(nn.Module)为CaptionBertAttention(BertSelfAttention)
        self.output = BertSelfOutput(config)  # original

    def forward(self,
                input_tensor,  # (16, 178, 768)
                attention_mask,  # (16, 1, 1, 768)
                #
                head_mask=None,  # None
                history_state=None):  # None
        self_outputs = self.self(input_tensor,  # (16, 178, 768)
                                 attention_mask,  # (16, 1, 1, 768)
                                 #
                                 head_mask,  # None
                                 history_state)  # None
        # print(self_outputs)
        '''
        (
            context_layer: (16, 178, 768),
        )
        或
        (
            context_layer: (16, 178, 768),
            attention_probs: (16, 12, 178, 178)
        )
        '''

        attention_output = self.output(self_outputs[0],  # context_layer: (16, 128, 768)
                                       input_tensor)  # (16, 178, 768)
        # print(attention_output)  # (16, 178, 768)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # print(outputs)
        '''
        (
            attention_output: (16, 178, 768),
        )
        或
        (
            attention_output: (16, 178, 768),
            attention_probs: (16, 12, 178, 178)
        )
        '''
        return outputs


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)

        self.attention = CaptionBertAttention(config)  # 修改BertAttention(nn.Module)为CaptionBertAttention(BertAttention)
        self.intermediate = BertIntermediate(config)  # original
        self.output = BertOutput(config)  # original

    def forward(self,
                hidden_states,  # 第1层的输入为embedding_output: (16, 178, 768) | 第2-12层的输入为layer_output: (16, 178, 768)
                attention_mask,  # extended_attention_mask: (16, 1, 1, 178)
                #
                head_mask=None,  # None
                history_state=None):  # None
        attention_outputs = self.attention(hidden_states,  # (16, 178, 768)
                                           attention_mask,  # (16, 1, 1, 768)
                                           #
                                           head_mask,  # None
                                           history_state)  # None
        # print(attention_outputs)
        '''
        (
            attention_output: (16, 178, 768),
        )
        或
        (
            attention_output: (16, 178, 768),
            attention_probs: (16, 12, 178, 178)
        )
        '''
        attention_output = attention_outputs[0]  # (16, 178, 768)
        intermediate_output = self.intermediate(attention_output)  # (16, 178, 3072)
        layer_output = self.output(intermediate_output, attention_output)  # (16, 178, 768)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        # print(outputs)
        '''
        (
            layer_output: (16, 178, 768)，
        )
        或
        (
            layer_output: (16, 178, 768)，
            attention_probs: (16, 12, 178, 178)
        )
        '''
        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)

        self.output_attentions = config.output_attentions  # original
        self.output_hidden_states = config.output_hidden_states  # original
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])  # 修改BertLayer(nn.Module)为CaptionBertLayer(BertLayer)

    def forward(self,
                hidden_states,  # embedding_output: (16, 178, 768)
                attention_mask,  # extended_attention_mask: (16, 1, 1, 178)
                #
                head_mask=None,  # None
                encoder_history_states=None):  # None
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # 把上一层的hidden_state加入到hidden_states中(从Embedding层的embedding_output到第12个Transformer层的layer_output)

            history_state = None if encoder_history_states is None else encoder_history_states[i]  # 新增: None

            layer_outputs = layer_module(hidden_states,  # 把上一层的输出作为当前层的输入: (16, 178, 768)
                                         attention_mask,  # (16, 1, 1, 178)
                                         #
                                         head_mask[i],  # None
                                         history_state)  # None
            # print(layer_outputs)
            '''
            (
                layer_output: (16, 178, 768)，
            )
            或
            (
                layer_output: (16, 178, 768)，
                attention_probs: (16, 12, 178, 178)
            )
            '''

            # 当前层的输出layer_output即为该层的hidden_state
            hidden_states = layer_outputs[0]  # (16, 178, 768)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # 加入当前层的attention_probs

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # 把第12层Transformer的hidden_state加入到hidden_states中

        outputs = (hidden_states,)  # ((16, 178, 768), )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # print(outputs)
        '''
        (
            hidden_states: (16, 178, 768),
        )
        或
        (
            hidden_states: (16, 178, 768),
            all_hidden_states: tuple(13个(16, 128, 768)),
        )
        或
        (
            hidden_states: (16, 178, 768),
            all_hidden_states: tuple(13个(16, 128, 768)),
            all_attentions: tuple(12个(16, 12, 178, 178))
        )
        '''
        return outputs  # last_hidden_state, (hidden states), (attentions)


class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """

    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)  # original
        self.encoder = CaptionBertEncoder(config)  # 修改BertEncoder(nn.Module)为CaptionBertEncoder(BertEncoder)
        self.pooler = BertPooler(config)  # original

        self.img_dim = config.img_feature_dim  # 新增: 2054
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        '''
        03/17/2022 16:27:20 - INFO - oscar.modeling.modeling_bert - BertImgModel Image Dimension: 2054
        '''

        self.img_feature_type = config.img_feature_type  # 新增: faster_r-cnn

        # if hasattr(config, 'use_img_layernorm'):
        #     self.use_img_layernorm = config.use_img_layernorm
        # else:
        #     self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            """ PASS """
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
            """ PASS """
        elif config.img_feature_type == 'dis_code_t':  # transpose
            """ PASS """
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
            """ PASS """
        elif config.img_feature_type == 'dis_code_scale':  # scaled
            """ PASS """
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
            """ PASS """
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)  # 新增: 2054 -> 768
            self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 新增: 0.1
            # if self.use_img_layernorm:
            #     self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)  # original

    def forward(self,
                input_ids,  # (16, 128)
                token_type_ids=None,  # (16, 128)
                attention_mask=None,  # (16, 178)
                img_feats=None,  # 新增: (16, 50, 2054)
                #
                position_ids=None,  # None
                head_mask=None,  # None
                #
                encoder_history_states=None):  # None
        if attention_mask is None:  # original
            """ PASS """
            attention_mask = torch.ones_like(input_ids)
            """ PASS """
        if token_type_ids is None:  # original
            """ PASS """
            token_type_ids = torch.zeros_like(input_ids)
            """ PASS """

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (16, 178) -> (16, 1, 1, 178)
        elif attention_mask.dim() == 3:  # 新增
            """ PASS """
            extended_attention_mask = attention_mask.unsqueeze(1)
            """ PASS """
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            """ PASS """
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
            """ PASS """
        else:
            head_mask = [None] * self.config.num_hidden_layers  # [None, None, None, None, None, None, None, None, None, None, None, None]

        embedding_output = self.embeddings(input_ids,  # (16, 128)
                                           token_type_ids=token_type_ids,  # (16, 128)
                                           position_ids=position_ids)  # None
        if encoder_history_states:
            """ PASS """
            assert img_feats is None, "Cannot take image features while using encoder history states"
            """ PASS """

        if img_feats is not None:  # (16, 50, 2054)
            if self.img_feature_type == 'dis_code':
                """ PASS """
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
                """ PASS """
            elif self.img_feature_type == 'dis_code_t':  # transpose
                """ PASS """
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
                """ PASS """
            elif self.img_feature_type == 'dis_code_scale':  # left scaled
                """ PASS """
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
                """ PASS """
            else:  # self.img_feature_type=faster_r-cnn
                img_embedding_output = self.img_embedding(img_feats)  # (16, 20, 2054) -> (16, 50, 768)
                # if self.use_img_layernorm:
                #     img_embedding_output = self.LayerNorm(img_embedding_output)
                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)  # (16, 50, 768) -> (16, 50, 768)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)  # torch.cat((16, 128, 768), (16, 50, 768), dim=1) -> (16, 178, 768)

        encoder_outputs = self.encoder(embedding_output,  # (16, 178, 768)
                                       extended_attention_mask,  # (16, 1, 1, 178)
                                       #
                                       head_mask=head_mask,  # None
                                       encoder_history_states=encoder_history_states)  # 新增: None
        # print(encoder_outputs)
        '''
        (
            hidden_states: (16, 178, 768),
        )
        或
        (
            hidden_states: (16, 178, 768),
            all_hidden_states: tuple(13个(16, 128, 768)),
        )
        或
        (
            hidden_states: (16, 178, 768),
            all_hidden_states: tuple(13个(16, 128, 768)),
            all_attentions: tuple(12个(16, 12, 178, 178))
        )
        '''
        sequence_output = encoder_outputs[0]  # (16, 178, 768)
        pooled_output = self.pooler(sequence_output)  # (16, 768)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # print(outputs)
        '''
        (
            last_hidden_state: (16, 178, 768),
            pooled_output: (16, 768),
            all_hidden_states: tuple(13个(16, 178, 768)),
            all_attentions: tuple(12个(16, 12, 178, 178))
        )
        '''
        return outputs

    def _resize_token_embeddings(self, new_num_tokens):  # original
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):  # original
        """
        Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


# def instance_bce_with_logits(logits, labels, reduction='mean'):
#     assert logits.dim() == 2
#     loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
#     if reduction == 'mean':
#         loss *= labels.size(1)
#     return loss


class ImageBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """

    def __init__(self, config):
        super(ImageBertForSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels  # original: 3129

        self.loss_type = config.loss_type  # 新增: bce

        # self.config = config

        if config.img_feature_dim > 0:  # 新增: 2054
            self.bert = BertImgModel(config)  # 新增: ImageBert
        else:
            """ PASS """
            self.bert = BertModel(config)  # original: Bert
            """ PASS """

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # original: 0.3(默认0.1)

        if hasattr(config, 'classifier'):  # config.classifier=linear
            if not hasattr(config, 'cls_hidden_scale'):  # config.cls_hidden_scale=3
                """ PASS """
                config.cls_hidden_scale = 2
                """ PASS """

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # 768 -> 3129
            elif config.classifier == 'mlp':
                """ PASS """
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),  # 768 -> 768*3
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)  # 768*3 -> 3129
                )
                """ PASS """
        else:
            """ PASS """
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
            """ PASS """

        self.apply(self.init_weights)  # original

    def forward(self,
                input_ids,  # (16, 128)
                token_type_ids=None,  # (16, 128)
                attention_mask=None,  # (16, 178)
                labels=None,  # (16, 3129)
                img_feats=None,  # 新增: (16, 50, 2054)
                #
                position_ids=None,  # None
                head_mask=None):  # None
        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(input_ids,  # (16, 128)
                                token_type_ids=token_type_ids,  # (16, 128)
                                attention_mask=attention_mask,  # (16, 178)
                                img_feats=img_feats,  # (16, 50, 2054)
                                #
                                head_mask=head_mask,  # None
                                position_ids=position_ids)  # None
        else:
            """ PASS """
            outputs = self.bert(input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                #
                                position_ids=position_ids,
                                head_mask=head_mask)
            """ PASS """

        pooled_output = outputs[1]  # (16, 768)

        pooled_output = self.dropout(pooled_output)  # (16, 768) -> (16, 768)

        logits = self.classifier(pooled_output)  # (16, 768) -> (16, 3129)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print(outputs)
        '''
        (
            logits: (16, 3129),
            hidden_states: tuple(13个(16, 178, 768)),
            attentions: tuple(12个(16, 12, 178, 178))
        )
        '''

        if labels is not None:  # VQA:(16, 3129) | GQA:(16, 1)
            if self.num_labels == 1:  # doing regression
                """ PASS """
                loss_fct = nn.MSELoss(reduction='mean')
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
                """ PASS"""
            else:
                if self.loss_type == 'kl':
                    """ PASS """
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)  # (16, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)  # (16, 3129)
                    loss = loss_fct(reshaped_logits, labels.contiguous())  # torch(数)
                    """ PASS """
                elif self.loss_type == 'bce':  # [VQA]
                    # loss = instance_bce_with_logits(logits, labels)
                    loss_fct = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    loss = loss_fct(logits, labels)  # VQA:BCE((16, 3129), (16, 3129)) -> torch(数)
                    loss *= labels.size(1)  # torch(数) * 3129
                elif self.loss_type == 'ce':  # [GQA, Retrieval, Captioning]
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # GQA:CE((16, 1853), (16,)) -> torch(数)
                else:
                    raise NotImplementedError()
            outputs = (loss,) + outputs
            # print(outputs)
            '''
            (
                loss: torch(数)
                logits: (16, 3129),
                hidden_states: tuple(13个(16, 178, 768)),
                attentions: tuple(12个(16, 12, 178, 178))
            )
            '''
        '''
        (
            logits: (16, 3129),
            hidden_states: tuple(13个(16, 178, 768)),
            attentions: tuple(12个(16, 12, 178, 178))
        )
        或
        (
            loss: torch(数)
            logits: (16, 3129),
            hidden_states: tuple(13个(16, 178, 768)),
            attentions: tuple(12个(16, 12, 178, 178))
        )
        '''
        return outputs

    def init_code_embedding(self, em):  # 新增
        self.bert.code_embeddings.weight.data = em.clone()



class ImageBertForMultipleChoice(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """

    def __init__(self, config):
        super(ImageBertForMultipleChoice, self).__init__(config)

        self.loss_type = config.loss_type  # 新增: ce

        if config.img_feature_dim > 0:  # 新增: 2054
            self.bert = BertImgModel(config)  # 新增: ImageBert
        else:
            """ PASS """
            self.bert = BertModel(config)  # original: Bert
            """ PASS """

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # original: 0.3(默认0.1)

        if hasattr(config, 'classifier'):  # config.classifier=mlp
            if not hasattr(config, 'cls_hidden_scale'):  # config.cls_hidden_scale=3
                """ PASS """
                config.cls_hidden_scale = 2
                """ PASS """
            if config.classifier == 'linear':
                """ PASS """
                self.classifier = nn.Linear(config.num_choice * config.hidden_size, self.config.num_labels)  # 2*768 -> 2
                """ PASS """
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.num_choice * config.hidden_size, config.hidden_size * config.cls_hidden_scale),  # 2*768 -> 768*3
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)  # 768*3 -> 2
                )
        else:
            """ PASS """
            self.classifier = nn.Linear(config.num_choice * config.hidden_size, self.config.num_labels)  # original
            """ PASS """

        self.apply(self.init_weights)  # original

    def forward(self,
                input_ids,  # (16, 2, 128)
                token_type_ids=None,  # (16, 2, 128)
                attention_mask=None,  # (16, 2, 178)
                labels=None,  # (16, 1)
                img_feats=None,  # 新增: (16, 2, 50, 2054)
                #
                position_ids=None,  # None
                head_mask=None):  # None
        # num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # (16, 2, 128) -> (32, 128)
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None  # (16, 2, 128) -> (32, 128)
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None  # (16, 2, 178) -> (32, 178)

        flat_img_feats = img_feats.view(-1, img_feats.size(-2), img_feats.size(-1)) if img_feats is not None else None  # (16, 2, 50, 2054) -> (32, 50, 2054)

        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None  # None

        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(flat_input_ids,  # (32, 128)
                                token_type_ids=flat_token_type_ids,  # (32, 128)
                                attention_mask=flat_attention_mask,  # (32, 178)
                                img_feats=flat_img_feats,  # 新增: (32, 50, 2054)
                                #
                                position_ids=flat_position_ids,  # None
                                head_mask=head_mask)  # None
        else:
            """ PASS """
            outputs = self.bert(flat_input_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask,
                                #
                                position_ids=flat_position_ids,
                                head_mask=head_mask)
            """ PASS"""

        pooled_output = outputs[1]  # (32, 768)

        pooled_output = self.dropout(pooled_output)  # (32, 768) -> (32, 768)

        # reshaped_pool_output
        reshaped_pool_output = pooled_output.view(-1, self.config.num_choice * (pooled_output.shape[1]))  # (32, 768) -> (16, 2*768)

        logits = self.classifier(reshaped_pool_output)  # (16, 2*768) -> (16, 2)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print(outputs)
        '''
        (
            logits: (16, 2),
            hidden_states: tuple(13个(32, 178, 768)),
            attentions: tuple(12个(32, 12, 178, 178))
        )
        '''

        if labels is not None:
            if self.loss_type == 'bce':
                """ PASS """
                # loss = instance_bce_with_logits(logits, labels.view(-1, self.config.num_labels))
                loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
                loss = loss_fct(logits, labels.view(-1, self.config.num_labels))  # 出错: BCE((16, 2), (8, 2))
                loss *= labels.size(1)  # torch(数) * 3129
                """ PASS """
            elif self.loss_type == 'ce':
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(logits, labels.view(-1))  # GQA:CE((16, 2), (16,)) -> torch(数)
            else:
                raise NotImplementedError()
            outputs = (loss,) + outputs
            # print(outputs)
            '''
            (
                loss: torch(数),
                logits: (16, 2),
                hidden_states: tuple(13个(32, 178, 768)),
                attentions: tuple(12个(32, 12, 178, 178))
            )
            '''
        # print(outputs)
        '''
        (
            logits: (16, 2),
            hidden_states: tuple(13个(32, 178, 768)),
            attentions: tuple(12个(32, 12, 178, 178))
        )
        或
        (
            loss: torch(数),
            logits: (16, 2),
            hidden_states: tuple(13个(32, 178, 768)),
            attentions: tuple(12个(32, 12, 178, 178))
        )
        '''
        return outputs


class BertForImageCaptioning(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """

    def __init__(self, config):
        super(BertForImageCaptioning, self).__init__(config)
        self.config = config
        self.bert = BertImgModel(config)
        self.transform = BertPredictionHeadTransform(config)
        bert_embedding_weight = self.bert.embeddings.word_embeddings.weight
        self.decoder = nn.Linear(bert_embedding_weight.size(1),
                                 bert_embedding_weight.size(0), bias=False)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.drop_worst_ratio = 0.2

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask, masked_pos, masked_ids=None,
                       token_type_ids=None, position_ids=None, head_mask=None,
                       is_training=True, encoder_history_states=None):
        outputs = self.bert(input_ids, img_feats=img_feats, attention_mask=attention_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        sequence_output = outputs[0][:, :masked_pos.shape[-1], :]

        if is_training:
            # num_masks_in_batch * hidden_size
            sequence_output_masked = sequence_output[masked_pos == 1, :]
            transformed_output_masked = self.transform(sequence_output_masked)
            class_logits = self.decoder(transformed_output_masked)
            masked_ids = masked_ids[masked_ids != 0]  # remove padding masks
            masked_loss = self.loss(class_logits.float(), masked_ids)
            outputs = (masked_loss, class_logits,) + outputs[2:]
        else:
            class_logits = self.decoder(self.transform(sequence_output))
            outputs = (class_logits,) + outputs[2:]
        return outputs

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size,
                                                      full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                                                                         dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1] - row_end + row_start,
                                     t.shape[2] - col_end + col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start,
                                               seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            img_feats = self.img_feats

            if self.add_od_labels:
                assert self.od_label_ids.shape[1] == self.od_labels_len
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                    torch.cat([x[:, 2:, :], x[:, :start_pos, :]], dim=1)
                    for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                      :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                      self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                      :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                      self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                    [torch.cat([i2i, i2s], dim=2),
                     torch.cat([s2i, s2s], dim=2)],
                    dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                                            for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                             self.od_labels_len + self.img_seq_len + start_pos: self.od_labels_len + self.img_seq_len + end_pos,
                             :self.od_labels_len + self.img_seq_len + end_pos]

        return {'input_ids': input_ids, 'img_feats': img_feats,
                'masked_pos': masked_pos, 'attention_mask': attention_mask,
                'token_type_ids': token_type_ids, 'position_ids': position_ids,
                'is_training': False,
                'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.decoder

    def generate(self, img_feats, attention_mask, masked_pos, token_type_ids=None,
                 position_ids=None, head_mask=None, input_ids=None, max_length=None,
                 do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, bos_token_id=None, pad_token_id=None,
                 eos_token_ids=None, mask_token_id=None, length_penalty=None, num_return_sequences=None,
                 num_keep_best=1, is_decode=None,
                 add_od_labels=False, od_labels_start_posid=None,
                 use_cbs=False, fsm=None, num_constraints=None,
                 min_constraints_to_satisfy=None, use_hypo=False,
                 ):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        self.max_seq_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        if not use_cbs:
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b == batch_size and v == vocab_size and f1 == num_fsm_states

        self.add_od_labels = add_od_labels
        # avoid position_ids collision of caption and od labels
        self.od_labels_start_posid = max(od_labels_start_posid, self.max_seq_len)
        if self.add_od_labels:
            # get od labels part from input_ids
            assert input_ids.shape[0] == batch_size
            od_label_ids = input_ids[:, self.max_seq_len:]
            self.od_labels_len = input_ids.shape[1] - self.max_seq_len
            self.od_label_ids = self._expand_for_beams(od_label_ids, num_beams,
                                                       num_fsm_states)
            input_ids = None
        else:
            self.od_labels_len = 0
            self.od_label_ids = None
            assert input_ids.shape == (batch_size, self.max_seq_len)
            input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len
            if self.add_od_labels:
                od_labels_posids = torch.arange(
                    self.od_labels_start_posid,
                    self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, od_labels_posids])
                posids_len += self.od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        cur_len = input_ids.shape[1]
        assert num_return_sequences == 1, 'not supported num_return_sequences != 1'
        effective_batch_size = batch_size

        self.img_feats = self._expand_for_beams(img_feats, num_beams, num_fsm_states)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_beams, num_fsm_states)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_beams, num_fsm_states)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_beams, num_fsm_states)
        self.full_position_ids = self._expand_for_beams(position_ids, num_beams, num_fsm_states)
        self.full_head_mask = self._expand_for_beams(head_mask, num_beams, num_fsm_states)

        if not use_cbs:
            if num_beams > 1:
                output = self._generate_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size,
                )
            else:
                output = self._generate_no_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                )
        else:
            assert self.num_keep_best == 1, 'not supported n_best > 1 for CBS'
            searcher = ConstrainedBeamSearch(eos_token_ids, max_length,
                                             num_beams, use_hypo=use_hypo)
            curr_ids, sum_logprobs = searcher.search(
                input_ids,
                None,
                self._decode_step,
                fsm,
            )
            curr_ids, sum_logprobs = select_best_beam_with_constraints(
                curr_ids,
                sum_logprobs,
                num_constraints,
                min_constraints_to_satisfy,
            )
            # (batch_size, n_best, max_len), (batch_size, n_best)
            output = (curr_ids.unsqueeze(1), sum_logprobs.unsqueeze(1))

        return output

    def _expand_for_beams(self, x, num_beams, num_fsm_states):
        num_expand = num_beams * num_fsm_states
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_beams, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1
