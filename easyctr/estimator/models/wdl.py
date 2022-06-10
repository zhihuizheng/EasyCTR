from easyctr.estimator.models import BaseModel
import tensorflow as tf
from tensorflow.estimator import RunConfig

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import custom_estimator, LINEAR_SCOPE_NAME, DNN_SCOPE_NAME
from ...layers.core import DNN
from ...layers.utils import concat_func, combined_dnn_input


class WDLEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(WDLEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_model(self):
        linear_feature_columns = self.numeric_feature_columns + self.categorical_feature_columns
        dnn_feature_columns = self.numeric_feature_columns + self.categorical_feature_columns

        def _model_fn(features, labels, mode, params):
            seed = params['seed']
            task = params['task']
            linear_optimizer = params['linear_optimizer']
            dnn_optimizer = params['dnn_optimizer']
            l2_reg_embedding = params['l2_reg_embedding']
            dnn_hidden_units = params['dnn_hidden_units']
            hidden_activations = params['hidden_activations']
            l2_reg_linear = params['l2_reg_linear']
            l2_reg_dnn = params['l2_reg_dnn']
            net_dropout = params['net_dropout']
            batch_norm = params['batch_norm']

            with tf.variable_scope(LINEAR_SCOPE_NAME):
                linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg_linear=l2_reg_linear)

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                     l2_reg_embedding)

            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_out = DNN(dnn_hidden_units, hidden_activations, l2_reg_dnn, net_dropout, False, seed=seed)(dnn_input)
            dnn_logit = tf.keras.layers.Dense(
                1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

            logits = linear_logit + dnn_logit #add_func([dnn_logit, linear_logit])

            return custom_estimator(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer)

        self.estimator = tf.estimator.Estimator(_model_fn, model_dir=self.model_dir, params=self.kwargs)