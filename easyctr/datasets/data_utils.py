import os
import logging
import numpy as np
import gc
import glob
import tensorflow as tf


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0, 
                     test_size=0, split_type="sequential"):
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None, valid_size=0, 
                  test_size=0, split_type="sequential", **kwargs):
    """ Build feature_map and transform tfrecords data """
    # Load csv data
    train_ddf = feature_encoder.read_csv(train_data)
    valid_ddf = feature_encoder.read_csv(valid_data) if valid_data else None
    test_ddf = feature_encoder.read_csv(test_data) if test_data else None

    # Split data for train/validation/test
    if valid_size > 0 or test_size > 0:
        train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                          valid_size, test_size, split_type)
    # fit and transform train_ddf
    train_ddf = feature_encoder.preprocess(train_ddf)
    feature_encoder.fit(train_ddf, **kwargs)
    #feature_name_dict = feature_encoder.get_feature_name_dict()   ######TODO: 这一行修改

    train_ddf = feature_encoder.transform(train_ddf)
    # block_size = int(kwargs.get("data_block_size", 0))
    # if block_size > 0:
    #     block_id = 0
    #     for idx in range(0, len(train_array), block_size):
    #         save_hdf5(train_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'train_part_{}.h5'.format(block_id)))
    #         block_id += 1
    # else:
    #     save_hdf5(train_array, os.path.join(feature_encoder.data_dir, 'train.h5'))
    # del train_array, train_ddf
    write_tfrecord(os.path.join(feature_encoder.data_dir, 'train.tfrecords'), train_ddf, feature_name_dict)
    gc.collect()

    # Transfrom valid_ddf
    if valid_ddf is not None:
        valid_ddf = feature_encoder.preprocess(valid_ddf)
        valid_ddf = feature_encoder.transform(valid_ddf)
        # valid_array = feature_encoder.transform(valid_ddf)
        # if block_size > 0:
        #     block_id = 0
        #     for idx in range(0, len(valid_array), block_size):
        #         save_hdf5(valid_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'valid_part_{}.h5'.format(block_id)))
        #         block_id += 1
        # else:
        #     save_hdf5(valid_array, os.path.join(feature_encoder.data_dir, 'valid.h5'))
        write_tfrecord(os.path.join(feature_encoder.data_dir, 'valid.tfrecords'), valid_ddf, feature_name_dict)
        # del valid_array, valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_ddf is not None:
        test_ddf = feature_encoder.preprocess(test_ddf)
        test_ddf = feature_encoder.transform(test_ddf)
        # test_array = feature_encoder.transform(test_ddf)
        # if block_size > 0:
        #     block_id = 0
        #     for idx in range(0, len(test_array), block_size):
        #         save_hdf5(test_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'test_part_{}.h5'.format(block_id)))
        #         block_id += 1
        # else:
        #     save_hdf5(test_array, os.path.join(feature_encoder.data_dir, 'test.h5'))
        write_tfrecord(os.path.join(feature_encoder.data_dir, 'test.tfrecords'), test_ddf, feature_name_dict)
        # del test_array, test_ddf
        gc.collect()
    logging.info("Transform csv data to tfrecords done.")


def make_example(line, feature_name_dict):
    features = {}
    features.update(
        {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][feat]])) for feat in
         feature_name_dict['numeric_feature_names']})
    features.update(
        {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[1][feat])])) for feat in
         feature_name_dict['categorical_feature_names']})
    features.update(
        {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=map(int, line[1][feat].split('-')))) for feat in #TODO: 加一个sequence特征的encode和decode
         feature_name_dict['sequence_feature_names']})
    features.update(
        {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][feat]])) for feat in
         feature_name_dict['label_names']})
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(filename, df, feature_name_dict):
    writer = tf.python_io.TFRecordWriter(filename)
    for line in df.iterrows():
        ex = make_example(line, feature_name_dict)
        writer.write(ex.SerializeToString())
    writer.close()
