import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import argparse
import logging
import os
from easyctr import datasets
from easyctr.utils import load_config, set_logger, set_checkpoints, print_to_json
from easyctr.features import FeatureEncoder
from easyctr.estimator import models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/star_synthetic.yaml', help='The config file.')

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
        #feature_map = feature_encoder.feature_map
    else:
        raise Exception("unsupport data format: {}.".format(params.get("data_format")))

    # initialize model
    model_class = getattr(models, params['model'])
    model = model_class(feature_encoder, **params)

    logging.info("Training")
    model.train()

    logging.info("Evaluating")
    model.evaluate()
