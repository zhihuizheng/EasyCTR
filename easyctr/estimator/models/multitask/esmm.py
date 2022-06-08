from easyctr.estimator.models import BaseModel
import tensorflow as tf
from ...feature_column import get_linear_logit, input_from_feature_columns
from ...utils import custom_estimator, LINEAR_SCOPE_NAME, DNN_SCOPE_NAME
from ....layers.utils import concat_func, combined_dnn_input
from easyctr.estimator.utils import _eval_metric_ops


class ESMMEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.kwargs = kwargs
        super(ESMMEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_model(self):
        feature_columns = self.numeric_feature_columns + self.categorical_feature_columns

        """
        part of code comes from https://github.com/qiaoguan/deep-ctr-prediction/blob/master/ESMM/esmm.py
        """
        def build_deep_layers(net, params):
            # Build the hidden layers, sized according to the 'hidden_units' param.
            for num_hidden_units in params['hidden_units']:
                net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_uniform_initializer())
            return net

        def model_fn(features, labels, mode, params):
            task = params['task']
            l2_reg_embedding = params['l2_reg_embedding']

            with tf.variable_scope(DNN_SCOPE_NAME):
                sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                                     l2_reg_embedding=l2_reg_embedding)

                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
                last_ctr_layer = build_deep_layers(dnn_input, params)
                last_cvr_layer = build_deep_layers(dnn_input, params)

                ctr_logits = tf.layers.dense(last_ctr_layer, units=1)
                cvr_logits = tf.layers.dense(last_cvr_layer, units=1)
                ctr_preds = tf.sigmoid(ctr_logits)
                cvr_preds = tf.sigmoid(cvr_logits)
                ctcvr_preds = tf.multiply(ctr_preds, cvr_preds)

                ctr_label = labels[params['esmm']['ctr_tower']['label_name']]  # labels['ctr_label']
                cvr_label = labels[params['esmm']['cvr_tower']['label_name']]  # labels['cvr_label']

                # click_label = features['label']
                # conversion_label = features['is_conversion']

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'ctr_preds': ctr_preds,
                    'cvr_preds': cvr_preds,
                    'ctcvr_preds': ctcvr_preds,
                    # 'click_label': click_label,
                    # 'conversion_label': conversion_label
                }
                export_outputs = {
                    'preds': tf.estimator.export.PredictOutput(predictions['cvr_preds'])  # 线上预测需要的
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            else:
                ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits)) #替换成get_loss这样的泛函数
                ctcvr_loss = tf.reduce_sum(tf.losses.log_loss(labels=cvr_label, predictions=ctcvr_preds))
                loss = ctr_loss + ctcvr_loss  # loss这儿可以加一个参数，参考multi-task损失的方法

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                eval_metric_ops = {}
                eval_metric_ops.update(_eval_metric_ops(ctr_label, ctr_preds, ctr_loss, task=task, name='ctr'))
                eval_metric_ops.update(_eval_metric_ops(cvr_label, ctcvr_preds, ctcvr_loss, task=task, name='ctcvr'))

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

        self.estimator = tf.estimator.Estimator(model_fn, model_dir=self.model_dir, params=self.kwargs)
