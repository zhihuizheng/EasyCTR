from easyctr.estimator.models import BaseModel
import tensorflow as tf
from tensorflow.estimator import RunConfig

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import deepctr_model_fn, LINEAR_SCOPE_NAME, DNN_SCOPE_NAME
from ...layers.core import DNN
from ...layers.interaction import FM, InteractingLayer
from ...layers.utils import add_func, concat_func, combined_dnn_input


class AutoIntEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(AutoIntEstimator, self).__init__(feature_encoder, **kwargs)

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
            l2_reg_dnn = params['l2_reg_dnn']
            net_dropout = params['net_dropout']
            batch_norm = params['batch_norm']
            att_layer_num = params['att_layer_num']
            l2_reg_linear = params['l2_reg_linear']
            embedding_dim = params['embedding_dim']
            att_head_num = params['att_head_num']
            att_res = params['att_res']

            if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
                raise ValueError("Either hidden_layer or att_layer_num must > 0")

            with tf.variable_scope(LINEAR_SCOPE_NAME):
                linear_logit = get_linear_logit(features, linear_feature_columns,
                                                l2_reg_linear=l2_reg_linear)

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                     l2_reg_embedding)

                att_input = concat_func(sparse_embedding_list, axis=1)

                for _ in range(att_layer_num):
                    att_input = InteractingLayer(
                        embedding_dim, att_head_num, att_res)(att_input)
                att_output = tf.keras.layers.Flatten()(att_input)

                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

                if len(dnn_hidden_units) > 0 and att_layer_num > 0:  # Deep & Interacting Layer
                    deep_out = DNN(dnn_hidden_units, hidden_activations, l2_reg_dnn, net_dropout, batch_norm, seed=seed)(
                        dnn_input)
                    stack_out = tf.keras.layers.Concatenate()([att_output, deep_out])
                    dnn_logit = tf.keras.layers.Dense(
                        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(stack_out)
                elif len(dnn_hidden_units) > 0:  # Only Deep
                    deep_out = DNN(dnn_hidden_units, hidden_activations, l2_reg_dnn, net_dropout, batch_norm, seed=seed)(
                        dnn_input, )
                    dnn_logit = tf.keras.layers.Dense(
                        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(deep_out)
                elif att_layer_num > 0:  # Only Interacting Layer
                    dnn_logit = tf.keras.layers.Dense(
                        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(att_output)
                else:  # Error
                    raise NotImplementedError

            logits = add_func([dnn_logit, linear_logit])

            return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                    training_chief_hooks=None)
        self.estimator = tf.estimator.Estimator(_model_fn, model_dir=self.model_dir, params=self.kwargs)
