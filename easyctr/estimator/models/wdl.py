from easyctr.estimator.models import BaseModel
from tensorflow.estimator import DNNLinearCombinedClassifier, RunConfig


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
