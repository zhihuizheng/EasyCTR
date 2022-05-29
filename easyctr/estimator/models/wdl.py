from easyctr.estimator.models import BaseModel
from tensorflow.estimator import DNNLinearCombinedClassifier, RunConfig


# def WDLEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_unit=(128, 64),
#                  config=None, **kwargs):
#     def
#
#     model_dir = os.path.join(kwargs['model_root'], kwargs['dataset_id'])
#     return DNNLinearCombinedClassifier(linear_feature_columns=linear_feature_columns,
#                                        dnn_feature_columns=dnn_feature_columns,
#                                        model_dir=model_dir,
#                                        dnn_hidden_units=dnn_hidden_unit,
#                                        config=config,
#                                        dnn_optimizer=kwargs['dnn_optimizer'])


class WDLEstimator(BaseModel):
    def __init__(self, feature_encoder, **kwargs):
        self.dnn_hidden_units = kwargs['dnn_hidden_units']
        self.dnn_optimizer = kwargs['dnn_optimizer']
        super(WDLEstimator, self).__init__(feature_encoder, **kwargs)

    def _build_model(self):
        feature_columns = self.numeric_feature_columns + self.categorical_feature_columns
        self.estimator = DNNLinearCombinedClassifier(linear_feature_columns=feature_columns,
                                                     dnn_feature_columns=feature_columns,
                                                     model_dir=self.model_dir,
                                                     dnn_hidden_units=self.dnn_hidden_units,
                                                     config=None,
                                                     dnn_optimizer=self.dnn_optimizer)
