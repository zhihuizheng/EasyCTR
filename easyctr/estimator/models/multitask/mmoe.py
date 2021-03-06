from easyctr.estimator.models import BaseModel
import tensorflow as tf
from tensorflow.estimator import RunConfig

from ...feature_column import get_linear_logit, input_from_feature_columns
from ...utils import custom_estimator, LINEAR_SCOPE_NAME, DNN_SCOPE_NAME
from ....layers.core import DNN, PredictionLayer
from ....layers.utils import concat_func, combined_dnn_input, reduce_sum
from easyctr.estimator.utils import _eval_metric_ops


class MMoEEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(MMoEEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_model(self):
        feature_columns = self.numeric_feature_columns + self.categorical_feature_columns

        def model_fn(features, labels, mode, params):
            seed = params['seed']
            l2_reg_embedding = params['l2_reg_embedding']
            dnn_activation = params['hidden_activations']
            l2_reg_dnn = params['l2_reg_dnn']
            net_dropout = params['net_dropout']
            batch_norm = params['batch_norm']
            expert_dnn_hidden_units = params['expert_dnn_hidden_units']
            gate_dnn_hidden_units = params['gate_dnn_hidden_units']
            tower_dnn_hidden_units = params['tower_dnn_hidden_units']
            num_experts = params['num_experts']
            num_tasks = params['num_tasks']
            task_names = params['task_names']
            task_types = params['task_types']

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                                     l2_reg_embedding=l2_reg_embedding)

                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

                # build expert layer
                expert_outs = []
                for i in range(num_experts):
                    expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout, batch_norm,
                                         seed=seed,
                                         name='expert_' + str(i))(dnn_input)
                    expert_outs.append(expert_network)

                expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(
                    expert_outs)  # None,num_experts,dim

                mmoe_outs = []
                for i in range(num_tasks):  # one mmoe layer: nums_tasks = num_gates
                    # build gate layers
                    gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout, batch_norm,
                                     seed=seed,
                                     name='gate_' + task_names[i])(dnn_input)
                    gate_out = tf.keras.layers.Dense(num_experts, use_bias=False, activation='softmax',
                                                     name='gate_softmax_' + task_names[i])(gate_input)
                    gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

                    # gate multiply the expert
                    gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                             name='gate_mul_expert_' + task_names[i])(
                        [expert_concat, gate_out])
                    mmoe_outs.append(gate_mul_expert)

                task_outs = []
                for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outs):
                    # build tower layer
                    tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout, batch_norm,
                                       seed=seed,
                                       name='tower_' + task_name)(mmoe_out)

                    logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
                    output = PredictionLayer(task_type, name=task_name)(logit)
                    task_outs.append(output)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = dict(zip(task_names, task_outs))
                export_outputs = {
                    key: tf.estimator.export.PredictOutput(value) for key, value in predictions.items()  # ?????????????????????
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
            else:
                task_losses = []
                eval_metric_ops = {}
                for task_name, task_type, task_out in zip(task_names, task_types, task_outs):
                    task_loss = tf.reduce_sum(tf.losses.log_loss(labels=labels[task_name], predictions=task_out))
                    eval_metric_ops.update(_eval_metric_ops(labels[task_name], task_out, task_loss, task=task_type, name=task_name))
                    task_losses.append(task_loss)
                #loss = tf.add_n(task_losses) # TODO: ??????loss?????????
                loss = 0
                for i, task_loss in enumerate(task_losses):
                    loss += 1.0 * task_loss

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

        self.estimator = tf.estimator.Estimator(model_fn, model_dir=self.model_dir, params=self.kwargs)
