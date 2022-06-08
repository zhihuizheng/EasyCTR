from easyctr.estimator.models import BaseModel
import tensorflow as tf
from tensorflow.estimator import RunConfig

from ...feature_column import get_linear_logit, input_from_feature_columns
from ...utils import custom_estimator, LINEAR_SCOPE_NAME, DNN_SCOPE_NAME
from ....layers.core import DNN, PredictionLayer
from ....layers.utils import concat_func, combined_dnn_input, reduce_sum
from easyctr.estimator.utils import _eval_metric_ops


class PLEEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(PLEEstimator, self).__init__(feature_encoder, **kwargs)

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
            num_tasks = params['num_tasks']
            task_names = params['task_names']
            task_types = params['task_types']
            num_levels = params['num_levels']
            shared_expert_num = params['shared_expert_num']
            specific_expert_num = params['specific_expert_num']

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                                     l2_reg_embedding=l2_reg_embedding)

                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

                # single Extraction Layer
                def cgc_net(inputs, level_name, is_last=False):
                    # inputs: [task1, task2, ... taskn, shared task]
                    specific_expert_outputs = []
                    # build task-specific expert layer
                    for i in range(num_tasks):
                        for j in range(specific_expert_num):
                            expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout,
                                                 batch_norm,
                                                 seed=seed,
                                                 name=level_name + 'task_' + task_names[i] + '_expert_specific_' + str(j))(inputs[i])
                            specific_expert_outputs.append(expert_network)

                    # build task-shared expert layer
                    shared_expert_outputs = []
                    for k in range(shared_expert_num):
                        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout,
                                             batch_norm,
                                             seed=seed,
                                             name=level_name + 'expert_shared_' + str(k))(inputs[-1])
                        shared_expert_outputs.append(expert_network)

                    # task_specific gate (count = num_tasks)
                    cgc_outs = []
                    for i in range(num_tasks):
                        # concat task-specific expert and task-shared expert
                        cur_expert_num = specific_expert_num + shared_expert_num
                        # task_specific + task_shared
                        cur_experts = specific_expert_outputs[
                                      i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs

                        expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

                        # build gate layers
                        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout, batch_norm,
                                         seed=seed,
                                         name=level_name + 'gate_specific_' + task_names[i])(
                            inputs[i])  # gate[i] for task input[i]
                        gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
                                                         name=level_name + 'gate_softmax_specific_' + task_names[i])(
                            gate_input)
                        gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

                        # gate multiply the expert
                        gate_mul_expert = tf.keras.layers.Lambda(
                            lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                            name=level_name + 'gate_mul_expert_specific_' + task_names[i])(
                            [expert_concat, gate_out])
                        cgc_outs.append(gate_mul_expert)

                    # task_shared gate, if the level not in last, add one shared gate
                    if not is_last:
                        cur_expert_num = num_tasks * specific_expert_num + shared_expert_num
                        cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

                        expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

                        # build gate layers
                        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout, batch_norm,
                                         seed=seed,
                                         name=level_name + 'gate_shared')(inputs[-1])  # gate for shared task input

                        gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
                                                         name=level_name + 'gate_softmax_shared')(gate_input)
                        gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

                        # gate multiply the expert
                        gate_mul_expert = tf.keras.layers.Lambda(
                            lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                            name=level_name + 'gate_mul_expert_shared')(
                            [expert_concat, gate_out])

                        cgc_outs.append(gate_mul_expert)
                    return cgc_outs

                # build Progressive Layered Extraction
                ple_inputs = [dnn_input] * (num_tasks + 1)  # [task1, task2, ... taskn, shared task]
                ple_outputs = []
                for i in range(num_levels):
                    if i == num_levels - 1:  # the last level
                        ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
                    else:
                        ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
                        ple_inputs = ple_outputs

                task_outs = []
                for task_type, task_name, ple_out in zip(task_types, task_names, ple_outputs):
                    # build tower layer
                    tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, net_dropout, batch_norm,
                                       seed=seed,
                                       name='tower_' + task_name)(ple_out)
                    logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
                    output = PredictionLayer(task_type, name=task_name)(logit)
                    task_outs.append(output)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = dict(zip(task_names, task_outs))
                export_outputs = {
                    key: tf.estimator.export.PredictOutput(value) for key, value in predictions.items()  # 线上预测需要的
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
            else:
                task_losses = []
                eval_metric_ops = {}
                for task_name, task_type, task_out in zip(task_names, task_types, task_outs):
                    task_loss = tf.reduce_sum(tf.losses.log_loss(labels=labels[task_name], predictions=task_out))
                    eval_metric_ops.update(_eval_metric_ops(labels[task_name], task_out, task_loss, task=task_type, name=task_name))
                    task_losses.append(task_loss)
                #loss = tf.add_n(task_losses) # TODO: 增加loss的权重
                loss = 0
                for i, task_loss in enumerate(task_losses):
                    loss += 1.0 * task_loss

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

        self.estimator = tf.estimator.Estimator(model_fn, model_dir=self.model_dir, params=self.kwargs)
