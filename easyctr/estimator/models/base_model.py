import os
import logging
from easyctr import datasets, inputs
import tensorflow as tf


class BaseModel(object):
    def __init__(self, feature_encoder, **kwargs):
        self.feature_encoder = feature_encoder
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
        self.numeric_columns, self.categorical_columns, self.label_column = self.feature_encoder.get_column_names() #TODO
        self._build_feature_columns()
        self._build_input()
        self._build_model()

    def _build_feature_columns(self):
        numeric_feature_columns = []
        categorical_feature_columns = []
        # 连续特征
        for col in self.numeric_columns:
            numeric_col = tf.feature_column.numeric_column(col)
            numeric_feature_columns.append(numeric_col)
        # 离散特征做embedding
        for col in self.categorical_columns:
            cate_col = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, self.feature_encoder.feature_map.feature_specs[col]["vocab_size"] + 1), #已经编码为从0开始的连续正整数
                self.embedding_dim)
            categorical_feature_columns.append(cate_col)
        self.numeric_feature_columns, self.categorical_feature_columns = numeric_feature_columns, categorical_feature_columns

    def _build_input(self):
        feature_description = {k: tf.FixedLenFeature(dtype=tf.int64, shape=1) for k in self.categorical_columns}
        feature_description.update({k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in self.numeric_columns})
        feature_description[self.label_column] = tf.FixedLenFeature(dtype=tf.float32, shape=1)

        self.train_input = inputs.input_fn_tfrecord(self.train_data, feature_description, self.label_column,
                                                    batch_size=self.batch_size, num_epochs=1, shuffle_factor=10)
        self.eval_input = inputs.input_fn_tfrecord(self.valid_data, feature_description, self.label_column,
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
