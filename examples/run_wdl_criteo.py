import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.estimator import DNNLinearCombinedClassifier, RunConfig


# 一些常用的函数
# 对连续特征做log变换
def log_fn(x, epson=0.000001):
    assert (x >= 0.0)
    if x == 0.0:
        return np.log(epson)
    else:
        return np.log(x)


def column_mean_and_std_fn(df, col):
    not_nan = df[col].notna()
    tmp_df = df[col][not_nan]
    return tmp_df.mean(), tmp_df.std()


# 填充缺失特征
def column_fillna(df, col, value):
    tmp_df = df[col].fillna(value)
    return tmp_df


# input_fn，这函数为模型train提供训练数据
# estimator.train方法接受的训练数据有两种形式：一种是dataset类型的数据，另一种是tensor类的数据
# tensor类型的数据，各种特征的变换需要自己实现，很麻烦。
def input_fn(features, labels, training=True, batch_size=256, num_epochs=5):
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # 如果在训练模式下混淆并重复数据。
    if training:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds


if __name__ == "__main__":
    df = pd.read_csv("../data/criteo_x1/train_sample.csv")

    df = df[0:600000]
    df = shuffle(df)

    columns = df.columns
    numeric_columns = columns[1:14]
    categorical_columns = columns[14:]

    numeric_columns = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
    categorical_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                           'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                           'C22', 'C23', 'C24', 'C25', 'C26']

    # 填充缺失值
    # for numeric
    for col in numeric_columns:
        mean, std = column_mean_and_std_fn(df, col)
        a = column_fillna(df, col, mean)
        # print(mean)
        df[col] = a
        df[col] = df[col].apply(lambda x: log_fn(x))

    # for cate
    for col in categorical_columns:
        value = ("%s_nan" % (col))
        df[col] = df[col].fillna(value)

    # # 构造catrgorical特征对应的字典信息
    # I2 = df['I2']  # 没有nan值
    # I2_info = df.groupby(['I2']).size().sort_values(ascending=False)
    # select_features = I2_info[I2_info >= 150]
    # I2_feature_list = select_features.index.tolist()

    categorical_columns_dict = {}
    omit_columns = ['I2', 'C3', 'C9', 'C12', 'C16', 'C21']
    for col in categorical_columns[0:]:
        if col in omit_columns:
            continue
        tmp = df.groupby([col]).size().sort_values(ascending=False).cumsum() / 600000
        feature_list = tmp[tmp <= 0.95]
        categorical_columns_dict[col] = feature_list.index.tolist()

    # categorical_columns_dict['I2'] = I2_feature_list
    # print(categorical_columns_dict)

    # 嵌入的维度
    embedding_dim = 16

    # 离线特征列，同上，没啥区别，可能丢弃了一些列
    categorical_columns_new = [col for col in categorical_columns if col not in omit_columns]
    # categorical_columns_new.append('I2')

    tf_feature_columns = []

    # for numeric
    for col in numeric_columns:
        numeric_col = tf.feature_column.numeric_column(col)
        tf_feature_columns.append(numeric_col)

    # 对离散特征做embedding后喂给模型
    for col in categorical_columns_new:
        cate_col = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                col
                , categorical_columns_dict[col]
            )
            , embedding_dim)
        tf_feature_columns.append(cate_col)

    # 交叉特征
    # dnn_tf_feature_columns = copy.deepcopy(tf_feature_columns)
    # linear_tf_feature_columns = copy.deepcopy(tf_feature_columns)

    # crossed_features = tf.feature_column.crossed_column(['C2', 'C23'], 6 * 188)
    # linear_tf_feature_columns.append(crossed_features)
    # crossed_features = tf.feature_column.crossed_column(['C18', 'C23'], 6 * 864)
    # linear_tf_feature_columns.append(crossed_features)
    #
    # print(len(dnn_tf_feature_columns))
    # print(len(linear_tf_feature_columns))

    # run time config
    run_config = RunConfig(save_summary_steps=100
                           , save_checkpoints_steps=100000
                           , keep_checkpoint_max=2)

    # define model
    estimator_wd_v2 = DNNLinearCombinedClassifier(linear_feature_columns=tf_feature_columns
                                                  , dnn_feature_columns=tf_feature_columns
                                                  , n_classes=2
                                                  , dnn_hidden_units=[64, 32]
                                                  , config=run_config
                                                  , dnn_optimizer='Adam')

    print(type(estimator_wd_v2))

    split_point = int(600000 * 0.8)
    df_train = df[0:split_point]
    df_test = df[split_point:]

    # train
    train_labels = df_train['label'].to_numpy()
    train_features = df_train.drop(['label'], axis=1)

    # test
    test_labels = df_test['label'].to_numpy()
    test_features = df_test.drop(['label'], axis=1)

    # 注意，调用train方法的时候，我们需要把训练数据送给模型
    # input_fn参数接受一个函数作为输入，我们需要在这个函数里把数据喂给模型。
    # 这里就用到了我们刚开始定义的input_fn函数了，该函数返回一个tf.dataset实例，可以作为estimator的输入
    # 比较奇葩的方式！
    # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/estimator/DNNLinearCombinedClassifier#train
    estimator_wd_v2.train(input_fn=lambda: input_fn(train_features, train_labels, training=True, num_epochs=1),
                          steps=None)

    train_result_wd_v2 = estimator_wd_v2.evaluate(
        input_fn=lambda: input_fn(train_features, train_labels, training=False))
    print(train_result_wd_v2)
    eval_result_wd_v2 = estimator_wd_v2.evaluate(input_fn=lambda: input_fn(test_features, test_labels, training=False))
    print(eval_result_wd_v2)

