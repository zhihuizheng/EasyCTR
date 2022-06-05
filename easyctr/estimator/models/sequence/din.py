from collections import OrderedDict

import tensorflow as tf
from easyctr.estimator.models import BaseModel
from easyctr.estimator.utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope
# from easyctr.input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from easyctr.layers.core import DNN, PredictionLayer
# from easyctr.layers.sequence import AttentionSequencePoolingLayer
from easyctr.layers.utils import concat_func, NoMask
# from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
# from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
# from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


# def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
#     sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
#     user_behavior_input = OrderedDict()
#     for i, feat in enumerate(seq_feature_list):
#         user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)
#
#     return sparse_input, dense_input, user_behavior_input

def get_input(feature_name_dict, features, max_seq_len):
    numeric_cols = []
    categorical_cols = []
    sequence_cols = []
    for col in feature_name_dict['numeric_feature_names']:
        numeric_cols.append(features[col])
    # 离散特征
    for col in feature_name_dict['categorical_feature_names']:
        categorical_cols.append(features[col])
    # 序列特征
    for col in feature_name_dict['sequence_feature_names']:
        sequence_cols.append(features[col])

    return numeric_cols, categorical_cols, sequence_cols


def get_embedding_vec_list(embedding_dict, inputs, feature_names, return_feat_list=()):
    embedding_vec_list = []
    for feat_name, input in zip(feature_names, inputs):
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            embedding_vec = embedding_dict[feat_name](input)
            # embedding_vec = tf.nn.embedding_lookup(embedding_dict[feat_name], input)
            embedding_vec_list.append(embedding_vec)
    return embedding_vec_list


def attention_layer(querys, keys, keys_id):
    """
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]
        keys_id:     [Batchsize, max_seq_len]
        from: https://github.com/qiaoguan/deep-ctr-prediction/blob/master/Din/din.py
    """

    keys_length = tf.shape(keys)[1]  # padded_dim
    embedding_size = querys.get_shape().as_list()[-1]
    #keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
    querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])

    net = tf.concat([keys, keys - querys, querys, keys * querys], axis=-1)
    for units in [32, 16]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)  # shape(batch_size, max_seq_len, 1)
    outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  # shape(batch_size, 1, max_seq_len)

    scores = outputs
    # TODO: 暂时去掉mask，因为keys_id是多个id拼接起来的，换种方式屏蔽
    #   一定要加屏蔽，因为正常的0值id是有别的含义的
    # key_masks = tf.expand_dims(tf.cast(keys_id > 0, tf.bool), axis=1)  # shape(batch_size, 1, max_seq_len) we add 0 as padding
    # key_masks = tf.expand_dims(tf.not_equal(keys_id, 0), axis=1)
    # paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    # scores = tf.where(key_masks, scores, paddings)

    scores = scores / (embedding_size ** 0.5)  # scale
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, keys)  # (batch_size, 1, embedding_size)
    # outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")  # (batch_size, embedding_size) #不要降维

    return outputs


class DINEstimator(BaseModel):
    """
    part of code comes from：https://github.com/shenweichen/DeepCTR, https://github.com/shenweichen/DSIN
    """
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(DINEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_feature_columns(self):
        pass

    def _build_model(self):
        def _model_fn(features, labels, mode, params):
            seed = params['seed']
            task = params['task']
            embedding_dim = params['embedding_dim']
            init_std = params['init_std']
            linear_optimizer = params['linear_optimizer']
            dnn_optimizer = params['dnn_optimizer']
            l2_reg_embedding = params['l2_reg_embedding']
            dnn_hidden_units = params['dnn_hidden_units']
            hidden_activations = params['hidden_activations']
            l2_reg_dnn = params['l2_reg_dnn']
            net_dropout = params['net_dropout']
            batch_norm = params['batch_norm']
            max_seq_len = params['max_seq_len']

            dense_input, sparse_input, sequence_input = get_input(
                    self.feature_name_dict, features, max_seq_len)

            with tf.variable_scope(DNN_SCOPE_NAME): #如果不声明variable_scope好像就不会更新，AUC只有0.5x
                embedding_dict = {feat: tf.keras.layers.Embedding(self.feature_encoder.get_vocab_size(feat), embedding_dim,
                                                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                                  embeddings_regularizer=l2(l2_reg_embedding),
                                                                  name='sparse_emb_' + str(i) + '-' + feat,
                                                                  mask_zero=False) for i, feat in
                                         enumerate(self.feature_name_dict["categorical_feature_names"])}

                # embedding_dict = {feat: tf.get_variable(name=feat, dtype=tf.float32,
                #                                         shape=[self.feature_encoder.get_vocab_size(feat), embedding_dim],
                #                                         trainable=True)
                #                   for i, feat in enumerate(self.feature_name_dict["categorical_feature_names"])}

                return_feature_list = [name.lstrip("hist_") for name in self.feature_name_dict["sequence_feature_names"]]
                query_emb_list = get_embedding_vec_list(embedding_dict, sparse_input, self.feature_name_dict["categorical_feature_names"], return_feature_list)

                keys_emb_list = get_embedding_vec_list(embedding_dict, sequence_input, return_feature_list)

                deep_input_emb_list = get_embedding_vec_list(embedding_dict, sparse_input, self.feature_name_dict["categorical_feature_names"])

                keys_emb = concat_func(keys_emb_list)
                query_emb = concat_func(query_emb_list)
                deep_input_emb = concat_func(deep_input_emb_list)

                # hist = AttentionSequencePoolingLayer((32, 16), att_activation="sigmoid",
                #                                      weight_normalization=False, supports_masking=True)([
                #     query_emb, keys_emb])
                hist = attention_layer(query_emb, keys_emb, sequence_input)

                deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist])
                deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
                if len(dense_input) > 0:
                   deep_input_emb = tf.keras.layers.Concatenate()([deep_input_emb] + dense_input)

                dnn_output = DNN(dnn_hidden_units, hidden_activations, l2_reg_dnn, net_dropout, batch_norm, seed=seed)(
                    deep_input_emb)
            logits = tf.keras.layers.Dense(
                1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

            return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                    training_chief_hooks=None)

        self.estimator = tf.estimator.Estimator(_model_fn, model_dir=self.model_dir, params=self.kwargs)
