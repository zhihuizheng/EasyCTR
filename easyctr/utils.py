import os
import logging
import logging.config
import yaml
import json
import shutil
from collections import OrderedDict


def load_config(config_file):
    params = dict()
    if not os.path.exists(config_file):
        raise RuntimeError('config_file={} not exist!'.format(config_file))

    with open(config_file, 'r') as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
        if 'data' in config_dict:
            params.update(config_dict['data'])
        if 'model' in config_dict:
            params.update(config_dict['model'])

    return params


def set_logger(params, log_file=None):
    if log_file is None:
        dataset_id = params['dataset_id']
        model = params['model']
        log_dir = os.path.join(params['log_root'], dataset_id)
        log_file = os.path.join(log_dir, model + '.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        # handlers=[logging.FileHandler(log_file, mode='w'),
                        #           logging.StreamHandler()]) #这样写会在控制台重复打印日志
                        handlers=[logging.FileHandler(log_file, mode='w')])


def set_checkpoints(params):
    model_dir = os.path.join(params['model_root'], params['dataset_id'])
    if os.path.exists(model_dir): # remove old checkpoints
        shutil.rmtree(model_dir)


def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)


def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())


# class Monitor(object):
#     def __init__(self, kv):
#         if isinstance(kv, str):
#             kv = {kv: 1}
#         self.kv_pairs = kv
#
#     def get_value(self, logs):
#         value = 0
#         for k, v in self.kv_pairs.items():
#             value += logs.get(k, 0) * v
#         return value
