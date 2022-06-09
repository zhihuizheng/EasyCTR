# coding: utf-8
import os
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


df = pd.read_csv("../data/alidisplay_x1/sampled_din.csv")
# print(df.describe())
# print(df.info())

def get_conv(x):
    if x == 0: return 0
    if random.uniform(0, 1) < 0.2: return 1
    return 0

def get_sceneId(x):
    rand = random.uniform(0, 1)
    if rand <= 0.3: return 1
    if rand <= 0.6: return 2
    return 3

df['conv'] = df['clk'].apply(lambda x: get_conv(x))
df['sceneId'] = df['clk'].apply(lambda x: get_sceneId(x))

os.makedirs("../data/multitask_x1")
df.to_csv("../data/multitask_x1/synthetic.csv", index=False)
