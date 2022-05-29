import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import argparse
import logging
import os
import shutil
from easyctr import datasets, inputs
from easyctr.utils import load_config, set_logger, set_checkpoints, print_to_json
from easyctr.features import FeatureMap, FeatureEncoder
from easyctr.estimator import models

import tensorflow as tf
from tensorflow.estimator import DNNLinearCombinedClassifier, RunConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/wdl_criteo.yaml', help='The config file.')

    args = vars(parser.parse_args())
    params = load_config(args['config'])
    set_logger(params)
    set_checkpoints(params)
    logging.info(print_to_json(params))

    # preporcess the datasets
    dataset = params['dataset_id'].split('_')[0].lower()
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    if params.get("data_format") == 'csv':  # load data from csv
        try:
            feature_encoder = getattr(datasets, dataset).FeatureEncoder(**params)
        except:
            feature_encoder = FeatureEncoder(**params)
        if os.path.exists(feature_encoder.json_file) and os.path.exists(feature_encoder.pickle_file):
            feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
            feature_encoder.feature_map.load(feature_encoder.json_file)
        else:  # Build feature_map and transform tfrecords data
            datasets.build_dataset(feature_encoder, **params)
        params["train_data"] = os.path.join(data_dir, 'train.tfrecords')
        params["valid_data"] = os.path.join(data_dir, 'valid.tfrecords')
        params["test_data"] = os.path.join(data_dir, 'test.tfrecords')
        feature_map = feature_encoder.feature_map
    else:
        raise Exception("unsupport data format: {}.".format(params.get("data_format")))

    # initialize model
    model_class = getattr(models, params['model'])
    model = model_class(feature_encoder, **params)

    logging.info("Training")
    model.train()

    logging.info("Evaluating")
    model.evaluate()




    # numeric_columns, categorical_columns, _ = feature_encoder.get_column_names()
    # numeric_feature_columns, categorical_feature_columns = feature_encoder.build_feature_columns(params['embedding_dim'])
    # feature_columns = numeric_feature_columns + categorical_feature_columns

    # # run time config
    # run_config = RunConfig(save_summary_steps=100,
    #                        save_checkpoints_steps=100,
    #                        keep_checkpoint_max=2)
    #
    # estimator_wd_v2 = models.WDLEstimator(feature_columns, feature_columns, dnn_hidden_unit=(64, 32),
    #                                       config=None, **params)

    # print(type(estimator_wd_v2))

    # training
    # TODO:
    #  边训练边评估
    #  打印训练进度
    #  训练比较慢，是卡在读数据还是参数更新
    #  原来pytorch训练也挺慢的，但是电脑没有说声音很大
    # estimator_wd_v2.train(input_fn=create_train_input_fn(params['train_data'], params))
    #
    # train_result_wd_v2 = estimator_wd_v2.evaluate(input_fn=create_test_input_fn(params['train_data'], params))
    # print(train_result_wd_v2)
    #
    # eval_result_wd_v2 = estimator_wd_v2.evaluate(input_fn=create_test_input_fn(params['valid_data'], params))
    # print(eval_result_wd_v2)


    # # TODO: 单独放一个模块
    # feature_description = {k: tf.FixedLenFeature(dtype=tf.int64, shape=1) for k in categorical_columns}
    # feature_description.update({k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in numeric_columns})
    # feature_description['label'] = tf.FixedLenFeature(dtype=tf.float32, shape=1)
    #
    # train_model_input = inputs.input_fn_tfrecord(params['train_data'], feature_description, 'label', batch_size=128,
    #                                              num_epochs=1, shuffle_factor=10)
    # test_model_input = inputs.input_fn_tfrecord(params['valid_data'], feature_description, 'label',
    #                                             batch_size=2 ** 14, num_epochs=1, shuffle_factor=0)
    # train_spec = tf.estimator.TrainSpec(input_fn=train_model_input)
    # eval_spec = tf.estimator.EvalSpec(input_fn=test_model_input)
    # # tf.estimator.train_and_evaluate(estimator_wd_v2, train_spec, eval_spec)
    #
    # for i in range(1, params['epochs'] + 1):
    #     logging.info("epoch: ", i)
    #     estimator_wd_v2.train(train_model_input)
    #     estimator_wd_v2.evaluate(test_model_input)
    #
    # print("11111")
    # estimator_wd_v2.evaluate(train_model_input)
    #
    # print("22222")
    # estimator_wd_v2.evaluate(test_model_input)
