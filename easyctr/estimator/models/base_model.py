import os
import logging
from easyctr import datasets, inputs
import tensorflow as tf


class BaseModel(object):
    def __init__(self, feature_encoder, **kwargs):
        self.feature_encoder = feature_encoder
        self.params = kwargs
        self.seed = kwargs['seed']
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.embedding_dim = kwargs['embedding_dim']
        self.train_data = kwargs['train_data']
        self.valid_data = kwargs['valid_data']
        self.test_data = kwargs['test_data']
        self.model_dir = os.path.join(kwargs['model_root'], kwargs['dataset_id'])
        self.estimator = None
        self._init_graph()

    def _init_graph(self):
        self.feature_name_dict = self.feature_encoder.get_feature_name_dict()
        self._build_feature_columns() #非必需
        self._build_input()
        self._build_model()

    # TODO: 可以放在单独模块，返回dict: {'numeric_feature_columns': numeric_feature_columns, ...}
    def _build_feature_columns(self):
        self.numeric_feature_columns = []
        self.categorical_feature_columns = []
        self.sequence_feature_columns = []
        # 连续特征
        for col in self.feature_name_dict['numeric_feature_names']:
            numeric_col = tf.feature_column.numeric_column(col)
            self.numeric_feature_columns.append(numeric_col)
        # 离散特征
        for col in self.feature_name_dict['categorical_feature_names']:
            cate_col = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, self.feature_encoder.feature_map.feature_specs[col]["vocab_size"] + 1), #已经编码为从0开始的连续正整数
                self.embedding_dim)
            self.categorical_feature_columns.append(cate_col)
        # # 序列特征
        # for col in self.feature_name_dict['sequence_columns']:

    def _build_input(self):
        feature_description = {}
        feature_description.update(
            {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in self.feature_name_dict['numeric_feature_names']})
        feature_description.update(
            {k: tf.FixedLenFeature(dtype=tf.int64, shape=1) for k in self.feature_name_dict['categorical_feature_names']})
        feature_description.update(
            {k: tf.FixedLenFeature(dtype=tf.int64, shape=self.params['max_seq_len']) for k in
             self.feature_name_dict['sequence_feature_names']})
        feature_description[self.feature_name_dict['label_name']] = tf.FixedLenFeature(dtype=tf.float32, shape=1)

        self.train_input = inputs.input_fn_tfrecord(self.train_data, feature_description,
                                                    self.feature_name_dict['label_name'],
                                                    batch_size=self.batch_size, num_epochs=1, shuffle_factor=10)
        self.eval_input = inputs.input_fn_tfrecord(self.valid_data, feature_description,
                                                   self.feature_name_dict['label_name'],
                                                   batch_size=2 ** 14, num_epochs=1, shuffle_factor=0)

    def _build_model(self):
        raise NotImplementedError('model must implement build_model methold')

    def train(self):
        for i in range(1, self.epochs + 1):
            logging.info("epoch: " + str(i))
            self.estimator.train(self.train_input)
            self.estimator.evaluate(self.eval_input)

    def evaluate(self):
        self.estimator.evaluate(self.eval_input)
