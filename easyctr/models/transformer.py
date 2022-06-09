# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from .utils import feedforward, multihead_attention, layer_normalization
import collections

"""
part of code comes from https://github.com/qiaoguan/deep-ctr-prediction/blob/master/Transformer/transformer.py
A tensorflow implementation of Transformer
Ashish Vasmani et all.  "Attention is All You Need,"  In NIPS,2017.
"""


class Transformer(object):
    def __init__(self, num_units, num_blocks, num_heads, max_len, dropout_rate, pos_fixed=True, l2_reg=0.0):
        self.num_units = num_units        # embedding_size
        self.num_blocks = num_blocks      # the number of multi-head attention we use
        self.num_heads = num_heads
        self.max_len = max_len            # the max length of the sequence
        self.dropout_keep_prob = 1. - dropout_rate
        self.position_encoding_matrix = None
        self.pos_fixed = pos_fixed
        self.l2_reg = l2_reg

    def get_position_encoding(self, inputs, scope="pos_embedding/", reuse=None):
        '''
        Args:
            inputs: sequence embeddings, shape: (batch_size , max_len, embedding_size)
        Return:
            Output sequences which has the same shape with inputs
        '''
        #E = inputs.get_shape().as_list()[-1]  # static
        #N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=reuse):
            if self.position_encoding_matrix is None:
                # TODO: 不用numpy包
                encoded_vec = np.array(
                    [pos / np.power(10000, 2 * i / self.num_units) for pos in range(self.max_len) for i in
                     range(self.num_units)])
                encoded_vec[::2] = np.sin(encoded_vec[::2])   #从index为0开始，下一个元素index+2
                encoded_vec[1::2] = np.cos(encoded_vec[1::2])
                encoded_vec = tf.convert_to_tensor(encoded_vec.reshape([self.max_len, self.num_units]), dtype=tf.float32)
                self.position_encoding_matrix = encoded_vec  # (max_len, num_units)

            N = tf.shape(inputs)[0]  # batch_size
            T = tf.shape(inputs)[1]  # max_len
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (batch_size , max_len)
            position_encoding = tf.nn.embedding_lookup(self.position_encoding_matrix,
                                                       position_ind)  # (batch_size, len, num_units)
        return position_encoding

    # TODO: add mask
    def __call__(self, inputs):#, mask):
        '''
        Args:
            inputs: sequence embeddings (item_embeddings +  pos_embeddings) shape: (batch_size , max_len, embedding_size)
            mask:  deal with mask shape: (batch_size, max_len, 1)
        Return:
            Output sequences which has the same shape with inputs
        '''
        if self.pos_fixed:  # use sin/cos positional embedding
            position_encoding = self.get_position_encoding(inputs)  # (batch_size, len, num_units)
            inputs += position_encoding

        #inputs *= mask
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                inputs = multihead_attention(queries=layer_normalization(inputs),
                                             keys=inputs,
                                             num_units=self.num_units,
                                             num_heads=self.num_heads,
                                             dropout_keep_prob=self.dropout_keep_prob,
                                             causality=False,
                                             scope="self_attention")

                # Feed forward
                inputs = feedforward(layer_normalization(inputs), num_units=[self.num_units, self.num_units],
                                     dropout_keep_prob=self.dropout_keep_prob)

                #inputs *= mask
        outputs = layer_normalization(inputs)  # (batch_size, max_len, num_units)
        return outputs
