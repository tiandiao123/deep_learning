from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt


from tensorflow.python.compiler.tensorrt import trt_convert as trt




import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
import time


FROZEN_MODEL_PATH = '/data00/cuiqing.li/TF_checkpoint/mobilenet_v1_1.0_224_frozen-with-shapes.pb'
model_path = FROZEN_MODEL_PATH

# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
	with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
		graph_def = tf_compat_v1.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def





input_img = np.ones((1,224,224,3), dtype=np.float32)


graph = tf.Graph()
with graph.as_default():
	config = tf_compat_v1.ConfigProto()
	config.graph_options.optimizer_options.global_jit_level = tf_compat_v1.OptimizerOptions.ON_1
	#sess = tf.Session(config=config)
	with tf_compat_v1.Session(config=config) as sess:
		# read TensorRT model
		frozen_graph = read_pb_graph(FROZEN_MODEL_PATH)

		# obtain the corresponding input-output tensor
		tf.import_graph_def(frozen_graph, name='')
		input = sess.graph.get_tensor_by_name('input:0')
		# output = sess.graph.get_tensor_by_name('resnet_model/final_dense:0')
		output = sess.graph.get_tensor_by_name("MobilenetV1/Predictions/Softmax:0")

		# in this case, it demonstrates to perform inference for 50 times
		total_time = 0; n_time_inference = 100
		out_pred = sess.run(output, feed_dict={input: input_img})
		for i in range(n_time_inference):
			t1 = time.time()
			out_pred = sess.run(output, feed_dict={input: input_img})
			t2 = time.time()
			delta_time = t2 - t1
			total_time += delta_time
			print("needed time in inference-" + str(i) + ": ", delta_time)

		avg_time_original_model = total_time / n_time_inference
		print("average inference time: ", avg_time_original_model)











