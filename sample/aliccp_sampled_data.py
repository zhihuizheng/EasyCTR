# coding: utf-8
import os
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# feature_list = "21090522181.021090645531.021090934451.021691547801.030193516651.020541862221.020683167991.020784162051.050893550390.69315"
# features = feature_list.split("\x01")
# print(features)

df = pd.read_csv("/Users/zhengzhihui/workbench/dataset/Ali-CCP/sample_train/sample.csv",
                 sep=',', header=None, names=['sampleID', 'click', 'conversion', 'common_feature_index', 'feature_num', 'feature_list'])
# print(df.describe())
# print(df.info())

splits = df['feature_list'][0].split('\x01')
print(splits)
