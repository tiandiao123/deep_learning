import os
import argparse
from tvm.relay.frontend.tensorflow_parser import TFParser
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework import graph_util
import collections
from IPython import embed
import numpy as np
import time
from config.config import config
from config._base_params import tuning_params
import re

try:
    tf = tf.compat.v1
except:
    pass


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def load_tensorflow_graph(TF_pb_path, input_names, input_shapes, output_names, dtypes):
    graph_def = TFParser(TF_pb_path).parse()
    with tf.Graph().as_default() as graph:
        # input_map = {}
        # for i in range(len(input_names)):
        #     str_array = input_names[i].split(":")
        #     assert len(str_array) == 2
        #     input_name = str_array[0]
        #     dtype = dtypes[i]

            
        #     if "bool" in str(dtype) or "boolean" in str(dtype) or "is_training" in input_names[i]:
        #         new_input = tf.placeholder(dtype=tf.bool, shape=(), name=input_name)
        #     elif dtype == "int32":
        #         new_input = tf.placeholder(dtype=tf.int32, shape=input_shapes[i], name=input_name)
        #     elif dtype == "float32":
        #         new_input = tf.placeholder(dtype=tf.float32, shape=input_shapes[i], name=input_name)
        #     elif dtype == "float16":
        #         new_input = tf.placeholder(dtype=tf.float16, shape=input_shapes[i], name=input_name)
        #     elif "str" in str(dtype) or "string" in str(dtype):
        #         new_input = tf.placeholder(tf.string, shape=(None), name = input_name)
        #     else:
        #         print("unsupported dtype {}".format(dtype))
        #         exit(-1)

        #     input_map[input_name] = new_input
        # print("output input map info:")
        # for key in input_map:
        #     print("key: {}, val: {} ".format(key, input_map[key]))


        tf.import_graph_def(graph_def, name='')
        #tf.import_graph_def(graph_def, name='')

        return graph



def test_tensorflow_model(graph, feed_dict, output_names):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=tf_config) as sess:
        output_tensors = []
        for ele_name in output_names:
            output_tensor = sess.graph.get_tensor_by_name(ele_name)
            output_tensors.append(output_tensor)
        
        print("warming up")
        for i in range(10):
            sess.run(output_tensors, feed_dict)
        
        total_time = 0
        print("start testing model inference speed......")
        for i in range(600):
            if i%10==0:
                print("still testing......, current finished {} times of tests".format(i))
            start = time.time()
            res = sess.run(output_tensors, feed_dict)
            end = time.time()
            total_time += end - start
        
        avg_time = total_time/100
        avg_time *= 1000
        print("the inference speed of non-tuned model is {} ms".format(avg_time))

        
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(output_tensors, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_frozen.json', 'w') as f:
            f.write(ctf)



def test_tensorflow_model_using_XLA(graph, feed_dict, output_names):
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #sess = tf.Session(config=config)
    with tf.Session(graph=graph, config=config) as sess:
        output_tensors = []
        for ele_name in output_names:
            output_tensor = sess.graph.get_tensor_by_name(ele_name)
            output_tensors.append(output_tensor)

        print("warming up")
        for i in range(10):
            sess.run(output_tensors, feed_dict)

        total_time = 0
        print("start testing model inference speed......")
        for i in range(600):
            if i%10==0:
                print("still testing......, current finished {} times of tests".format(i))
            start = time.time()
            res = sess.run(output_tensors, feed_dict)
            end = time.time()
            total_time += end - start

        avg_time = total_time/100
        avg_time *= 1000
        print("the inference speed of non-tuned model using XLA is {} ms".format(avg_time))


        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(output_tensors, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_frozen_xla.json', 'w') as f:
            f.write(ctf)






def main():
    TF_pb_path = config['model_path']
    input_names = config['input_names']
    input_shapes = config['input_shapes']
    output_names = config['output_names']
    dtypes = config['dtypes']
    
    
    graph = load_tensorflow_graph(TF_pb_path, input_names, input_shapes, output_names, dtypes)


    feed_dict = {}
    
    for i in range(len(input_names)):
        input_shape = input_shapes[i]
        dtype = dtypes[i]

        if input_names[i]!='is_training:0':
            data_np = np.random.uniform(size=input_shape).astype(dtype)
            feed_dict[input_names[i]] = data_np
        elif "is_training" in input_names[i] or "bool" in str(dtype) or "boolean" in str(dtype):
            feed_dict[input_names[i]] = False
        elif "string" in str(dtype) or "str" in str(dtype):
            feed_dict[input_names[i]] = ""
        else:
            raise("un supported dtype : {}".format(str(dtype)))
    


    test_tensorflow_model(graph, feed_dict, output_names)
    test_tensorflow_model_using_XLA(graph, feed_dict, output_names)



if __name__ == "__main__":
    main()
    print("finished testing the model!!!")

    

