from collections import OrderedDict

import tensorflow as tf
from easyctr.estimator.models import BaseModel
from easyctr.estimator.utils import custom_estimator, DNN_SCOPE_NAME, variable_scope
# from easyctr.input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from easyctr.layers.core import DNN, PredictionLayer
# from easyctr.layers.sequence import AttentionSequencePoolingLayer
from easyctr.layers.utils import concat_func, NoMask
# from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
# from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
# from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


# def get_input(feature_specs, features):
#     numeric_cols = []
#     categorical_cols = []
#     sequence_cols = []
#     for feature, feature_spec in feature_specs.items():
#         if feature_spec['type'] == 'numeric':
#             numeric_cols.append(features[feature])
#         elif feature_spec['type'] == 'categorical':
#             categorical_cols.append(features[feature])
#         elif feature_spec['type'] == 'sequence':
#             sequence_cols.append(features[feature])
#     return numeric_cols, categorical_cols, sequence_cols


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
    # TODO: ????????????mask?????????keys_id?????????id????????????????????????????????????
    #   ????????????????????????????????????0???id?????????????????????
    # key_masks = tf.expand_dims(tf.cast(keys_id > 0, tf.bool), axis=1)  # shape(batch_size, 1, max_seq_len) we add 0 as padding
    # key_masks = tf.expand_dims(tf.not_equal(keys_id, 0), axis=1)
    # paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    # scores = tf.where(key_masks, scores, paddings)

    scores = scores / (embedding_size ** 0.5)  # scale
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, keys)  # (batch_size, 1, embedding_size)
    # outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")  # (batch_size, embedding_size) #????????????

    return outputs


def get_feature_names(feature_specs):
    numeric_cols = []
    categorical_cols = []
    sequence_cols = []
    for feature, feature_spec in feature_specs.items():
        if feature_spec['type'] == 'numeric':
            numeric_cols.append(feature)
        elif feature_spec['type'] == 'categorical':
            categorical_cols.append(feature)
        elif feature_spec['type'] == 'sequence':
            sequence_cols.append(feature)
    return numeric_cols, categorical_cols, sequence_cols


def _model_fn(features, labels, mode, params):
    feature_specs = params['feature_specs']
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

    # numeric_input, categorical_input, sequence_input = get_input(feature_specs, features)
    numeric_feature_names, categorical_feature_names, sequence_feature_names = get_feature_names(feature_specs) #TODO: ?????????
    numeric_input = [features[name] for name in numeric_feature_names]
    categorical_input = [features[name] for name in categorical_feature_names]
    sequence_input = [features[name] for name in sequence_feature_names]

    with tf.variable_scope(DNN_SCOPE_NAME):  # ???????????????variable_scope????????????????????????AUC??????0.5x
        embedding_dict = {feat: tf.keras.layers.Embedding(feature_specs[feat]["vocab_size"], embedding_dim,
                                                          embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                                              seed=seed),
                                                          embeddings_regularizer=l2(l2_reg_embedding),
                                                          name='embedding_' + feat,
                                                          mask_zero=False) for i, feat in
                          enumerate(categorical_feature_names)}

        # embedding_dict = {feat: tf.get_variable(name=feat, dtype=tf.float32,
        #                                         shape=[self.feature_encoder.get_vocab_size(feat), embedding_dim],
        #                                         trainable=True)
        #                   for i, feat in enumerate(self.feature_name_dict["categorical_feature_names"])}

        return_feature_list = [feature_specs[name]['cname'] for name in sequence_feature_names]
        query_emb_list = get_embedding_vec_list(embedding_dict, categorical_input,
                                                categorical_feature_names,
                                                return_feature_list)

        keys_emb_list = get_embedding_vec_list(embedding_dict, sequence_input, return_feature_list)

        deep_input_emb_list = get_embedding_vec_list(embedding_dict, categorical_input,
                                                     categorical_feature_names)

        keys_emb = concat_func(keys_emb_list)
        query_emb = concat_func(query_emb_list)
        deep_input_emb = concat_func(deep_input_emb_list)

        # hist = AttentionSequencePoolingLayer((32, 16), att_activation="sigmoid",
        #                                      weight_normalization=False, supports_masking=True)([
        #     query_emb, keys_emb])
        hist = attention_layer(query_emb, keys_emb, sequence_input)

        deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist])
        deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
        if len(numeric_input) > 0:
            deep_input_emb = tf.keras.layers.Concatenate()([deep_input_emb] + numeric_input)

        dnn_output = DNN(dnn_hidden_units, hidden_activations, l2_reg_dnn, net_dropout, batch_norm, seed=seed)(
            deep_input_emb)
        logits = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    return custom_estimator(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer)


class DINEstimator(BaseModel):
    """
    part of code comes from???https://github.com/shenweichen/DeepCTR, https://github.com/shenweichen/DSIN
    """
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(DINEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_feature_columns(self):
        pass

    # TODO: ????????????????????????
    def _build_model(self):
        params = self.kwargs
        params['feature_specs'] = self.feature_specs
        self.estimator = tf.estimator.Estimator(_model_fn, model_dir=self.model_dir, params=self.kwargs)
