import helper
import os
from glob import glob
from matplotlib import pyplot
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

print("hello test!")


def model_inputs(image_width, image_height, image_channels, z_dim):
	input_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name="input_real")
	input_z = tf.placeholder(tf.float32, (None, z_dim), name = "input_z")
	learning_rate = tf.placeholder(tf.float32, name = "learning_rate")

	return input_real, input_z, learning_rate

def discriminator(images, reuse=False):
	
	with tf.variable_scope('discriminator', reuse = reuse):
		# input size 28*28*3
		layer1 = tf.layers.conv2d(images, 64, 5, strides = 2, padding='same')
		relu1 = tf.maximum(0.12*layer1, layer1)

		# 14*14*64
		layer2 = tf.layers.conv2d(relu1, 128, 5, strides =2, padding='same')
		layer2 = tf.layer2.batch_normalization(layer2, training = True)
		relu2 = tf.maximum(0.12*layer2, layer2)

		# 7*7*128
		layer3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
		layer3 = tf.layers.batch_normalization(layers, training= True)
		relu3 = tf.maximum(0.12*layer3, layer3)

		flatten_layer = tf.reshape(relu3, (-1, 4*4*256))
		flatten_layer = tf.layers.dropout(flatten_layer, rate = 0.3)
		logits = tf.layers.dense(flatten_layer, 1)

		return out, logits

