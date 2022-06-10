import tensorflow as tf

# filenames = ['part-r-00099']
filenames = ['../data/alidisplay_x1/train.tfrecords']
sess = tf.InteractiveSession()
dataset = tf.data.TFRecordDataset(filenames).batch(10).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
iterator = dataset.make_one_shot_iterator()
batch_data = iterator.get_next()

res = sess.run(batch_data)
serialized_example = res[0]
example_proto = tf.train.Example.FromString(serialized_example)
features = example_proto.features
print(features)
