import tensorflow as tf
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2


from easyctr.estimator.models import BaseModel
import tensorflow as tf
from tensorflow.estimator import RunConfig

from ...feature_column import get_linear_logit, input_from_feature_columns
from ...utils import custom_estimator, LINEAR_SCOPE_NAME, DNN_SCOPE_NAME
from ....layers.core import DNN, PredictionLayer
from ....layers.utils import concat_func, combined_dnn_input, reduce_sum
from easyctr.estimator.utils import _eval_metric_ops


def activation_layer(activation):
    if isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


class STAREstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(STAREstimator, self).__init__(feature_encoder, **kwargs)

    # def _build_feature_columns(self):
    #     super(STAREstimator, self)._build_feature_columns()
    #     domain_col = self.kwargs['domain_col']
    #     # 把域Id特征从categorical_feature_columns中去掉（其实不去掉理论上也可以），并且声明成onehot类型输入
    #     for i, (feat, feat_col) in enumerate(
    #             zip(self.feature_name_dict['categorical_feature_names'], self.categorical_feature_columns)):
    #         if feat == domain_col:
    #             self.categorical_feature_columns.pop(i)
    #             break
    #     #self.domain_feature_column = tf.feature_column.indicator_column()

    def _build_model(self):
        feature_columns = self.numeric_feature_columns + self.categorical_feature_columns

        """
        part of code comes from: https://zhuanlan.zhihu.com/p/423557973
        """
        def model_fn(features, labels, mode, params):
            task = params['task']
            seed = params['seed']
            l2_reg_embedding = params['l2_reg_embedding']
            hidden_units = params['hidden_units']
            dnn_activation = params['hidden_activations']
            l2_reg_dnn = params['l2_reg_dnn']
            net_dropout = params['net_dropout']
            batch_norm = params['batch_norm']
            num_domains = params['num_domains']
            domain_col = params['domain_col']

            # 声明domain_indicator
            domain_indicator = tf.one_hot(indices=features[domain_col], depth=num_domains, on_value=1.0, off_value=0.0,
                                          axis=1) # (batch_size, num_domains, 1)

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                                     l2_reg_embedding=l2_reg_embedding)
                # 把domain列拿出来作为一个单独的特征（辅助任务用）
                for i, feature_name in enumerate(self.feature_name_dict['categorical_feature_names']):
                    if feature_name == domain_col:
                        domain_embedding = sparse_embedding_list.pop(i)
                        break

                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
                hidden_units = [dnn_input.shape[1].value] + hidden_units

                shared_kernels = [tf.get_variable(name='shared_kernel_' + str(i),
                                                       shape=(hidden_units[i-1], hidden_units[i]),
                                                       initializer=glorot_normal(seed=seed),
                                                       regularizer=l2(l2_reg_dnn),
                                                       trainable=True) for i in range(1, len(hidden_units))]

                shared_bias = [tf.get_variable(name='shared_bias_' + str(i),
                                                    shape=(hidden_units[i],),
                                                    initializer=Zeros(),
                                                    trainable=True) for i in range(1, len(hidden_units))]
                ## domain-specific 权重
                domain_kernels = [[tf.get_variable(name='domain_kernel_' + str(index) + str(i),
                                                        shape=(hidden_units[i-1], hidden_units[i]),
                                                        initializer=glorot_normal(seed=seed),
                                                        regularizer=l2(l2_reg_dnn),
                                                        trainable=True) for i in range(1, len(hidden_units))] for
                                       index in range(num_domains)]

                domain_bias = [[tf.get_variable(name='domain_bias_' + str(index) + str(i),
                                                     shape=(hidden_units[i],),
                                                     initializer=Zeros(),
                                                     trainable=True) for i in range(1, len(hidden_units))] for index
                                    in range(num_domains)]

                activation_layers = [activation_layer(dnn_activation) for _ in range(1, len(hidden_units))]

            output_list = [dnn_input] * num_domains
            for i in range(len(hidden_units) - 1):
                for j in range(num_domains):
                    # 网络的权重由共享FCN和其domain-specific FCN的权重共同决定
                    output_list[j] = tf.nn.bias_add(tf.tensordot(
                        output_list[j], shared_kernels[i] * domain_kernels[j][i], axes=(-1, 0)), shared_bias[i] + domain_bias[j][i])
                    try:
                        output_list[j] = activation_layers[i](output_list[j])#, training=training)
                    except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                        print("make sure the activation function use training flag properly", e)
                        output_list[j] = activation_layers[i](output_list[j])

            # main network
            output_m = tf.reduce_sum(tf.stack(output_list, axis=1) * domain_indicator, axis=1)
            logits_m = tf.layers.dense(output_m, units=1)

            # auxiliary network
            logits_a = tf.layers.dense(tf.reduce_sum(domain_embedding, axis=1), units=1)

            # total
            logits = logits_m + logits_a
            preds = tf.sigmoid(logits)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'logits': logits,
                    'preds': preds
                }
                export_outputs = {
                    'preds': tf.estimator.export.PredictOutput(predictions['preds'])  # 线上预测需要的
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
            else:
                loss = tf.reduce_sum(tf.losses.log_loss(labels=labels, predictions=preds))
                eval_metric_ops = _eval_metric_ops(labels, preds, loss, task=task)

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

        self.estimator = tf.estimator.Estimator(model_fn, model_dir=self.model_dir, params=self.kwargs)


# class STAR(BaseModel):
#
#     # def __init__(self, hidden_units, num_domains, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
#     #              seed=1024, **kwargs):
#     #     self.hidden_units = hidden_units
#     #     self.num_domains = num_domains
#     #     self.activation = activation
#     #     self.l2_reg = l2_reg
#     #     self.dropout_rate = dropout_rate
#     #     self.use_bn = use_bn
#     #     self.output_activation = output_activation
#     #     self.seed = seed
#     #
#     #     super(STAR, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         input_size = input_shape[-1]
#         hidden_units = [int(input_size)] + list(self.hidden_units)
#         ## 共享FCN权重
#         self.shared_kernels = [self.add_weight(name='shared_kernel_' + str(i),
#                                         shape=(hidden_units[i], hidden_units[i + 1]),
#                                         initializer=glorot_normal(seed=self.seed),
#                                         regularizer=l2(self.l2_reg),
#                                         trainable=True) for i in range(len(self.hidden_units))]
#
#         self.shared_bias = [self.add_weight(name='shared_bias_' + str(i),
#                                      shape=(self.hidden_units[i],),
#                                      initializer=Zeros(),
#                                      trainable=True) for i in range(len(self.hidden_units))]
#         ## domain-specific 权重
#         self.domain_kernels = [[self.add_weight(name='domain_kernel_' + str(index) + str(i),
#                                         shape=(hidden_units[i], hidden_units[i + 1]),
#                                         initializer=glorot_normal(eed=self.seed),
#                                         regularizer=l2(self.l2_reg),
#                                         trainable=True) for i in range(len(self.hidden_units))] for index in range(self.num_domains)]
#
#         self.domain_bias = [[self.add_weight(name='domain_bias_' + str(index) + str(i),
#                                      shape=(self.hidden_units[i],),
#                                      initializer=Zeros(),
#                                      trainable=True) for i in range(len(self.hidden_units))] for index in range(self.num_domains)]
#
#         self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]
#
#         if self.output_activation:
#             self.activation_layers[-1] = activation_layer(self.output_activation)
#
#         super(STAR, self).build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, inputs, domain_indicator, training=None, **kwargs):
#         deep_input = inputs
#         output_list = [inputs] * self.num_domains
#         for i in range(len(self.hidden_units)):
#             for j in range(self.num_domains):
#                 # 网络的权重由共享FCN和其domain-specific FCN的权重共同决定
#                 output_list[j] = tf.nn.bias_add(tf.tensordot(
#                     output_list[j], self.shared_kernels[i] * self.domain_kernels[j][i], axes=(-1, 0)), self.shared_bias[i] + self.domain_bias[j][i])
#                 try:
#                     output_list[j] = self.activation_layers[i](output_list[j], training=training)
#                 except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
#                     print("make sure the activation function use training flag properly", e)
#                     output_list[j] = self.activation_layers[i](output_list[j])
#         output = tf.reduce_sum(tf.stack(output_list, axis=1) * tf.expand_dims(domain_indicator, axis=-1), axis=1)
#
#         return output
#
#     # def compute_output_shape(self, input_shape):
#     #     if len(self.hidden_units) > 0:
#     #         shape = input_shape[:-1] + (self.hidden_units[-1],)
#     #     else:
#     #         shape = input_shape
#     #
#     #     return tuple(shape)
#     #
#     # def get_config(self, ):
#     #     config = {'activation': self.activation, 'hidden_units': self.hidden_units,
#     #               'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
#     #               'output_activation': self.output_activation, 'seed': self.seed}
#     #     base_config = super(STAR, self).get_config()
#     #     return dict(list(base_config.items()) + list(config.items()))
