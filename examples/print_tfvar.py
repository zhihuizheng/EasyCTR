# -*- encoding: utf-8 -*-

import os
from tensorflow.python import pywrap_tensorflow
 
checkpoint_path = os.path.join("model20210227.ckpt", "model.ckpt-557912")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
    #if "user_poi_part/fc_0/kernel" == key:
    #if "user_poi_part" in key:
    #    print(reader.get_tensor(key).shape)
    #    for x in reader.get_tensor(key):
    #        print(x)
    #        #raw_input()
