from easyctr.estimator.models import BaseModel
import tensorflow as tf
from tensorflow.estimator import RunConfig

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope
from ...layers.core import DNN
from ...layers.interaction import FM
from ...layers.utils import concat_func, combined_dnn_input


class DeepFMEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(DeepFMEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_model(self):

        linear_feature_columns = self.numeric_feature_columns #+ self.categorical_feature_columns
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

            train_flag = (mode == tf.estimator.ModeKeys.TRAIN)
            linear_logits = get_linear_logit(features, linear_feature_columns)

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                     l2_reg_embedding=l2_reg_embedding)

                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
                fm_logit = FM()(concat_func(sparse_embedding_list, axis=1))

                dnn_output = DNN(dnn_hidden_units, hidden_activations, l2_reg_dnn, net_dropout, batch_norm, seed=seed)(
                    dnn_input, training=train_flag)
                dnn_logit = tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

            logits = linear_logits + fm_logit + dnn_logit
            return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                    training_chief_hooks=None)
        self.estimator = tf.estimator.Estimator(_model_fn, model_dir=self.model_dir, params=self.kwargs)
