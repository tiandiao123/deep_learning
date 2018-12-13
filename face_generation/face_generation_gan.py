import helper
import os
from glob import glob
from matplotlib import pyplot
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

print("hello test!")

show_n_images = 25
count = 0


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
		out = tf.sigmoid(logits)


	return out, logits



def generator(z, out_channel_dim, is_train = True):
	with tf.variable_scope('generator', reuse = not is_train):
		layer1 = tf.layers.dense(z, 7*7*512)
		layer1 = tf.reshape(layer1, (-1, 7, 7, 512))
		relu1 = tf.maximum(0.12*layer1, layer1)

		# input 7*7*512
		layer2 = tf.layers.conv2d_transpose(relu1, 256, 5, strides=1, padding='same')
		layer2 = tf.layer2.batch_normalization(layer2, training = is_train)
		relu2 = tf.maximum(0.12*layer2, layer2)

		#output 7*7*256

		layer3 = tf.layers.conv2d_transpose(relu2, 128, 5, strides=2, padding='same')
		layer3 = tf.layers.batch_normalization(layer3, training = is_train)
		relu3 = tf.maximum(0.12*layer3, layer3)
		# 14 * 14 * 128 

		logits = tf.layers.conv2d_transpose(relu3, out_channel_dim, 5, strides=2. padding='same')
		output = tf.tanh(logits)

	return output


def model_loss(input_real, input_z, out_channel_dim):
	g_model = generator(input_z, out_channel_dim)
	d_model_real, d_logits_real = discriminator(input_real)
	d_model_fake, d_logits_fake = discriminator(g_model, reuse = True)

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels = tf.ones_like(d_model_real)*0.9))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels = tf.zeros_like(d_model_fake)))
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

	d_loss = d_loss_real + d_loss_fake

	return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
	g_vars = [var for vat in t_vars if var.name.startswith('generator')]

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(d_loss, var_list=d_vars)
		g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(g_loss, var_list=g_vars)
	return d_train_opt, g_train_opt



def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.save_config("step_" + str(count) + ".png")




def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
	input_real, input_z, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3])
	d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
	d_optimizer, g_optimizer = model_opt(d_loss, g_loss, lr, beta1)

	with tf.Session() as sess:
		for epch_i in range(len(epoch_count)):
			step = 0
			for batch_images in get_batches(batch_size):
				batch_images = 2*batch_images
				step+=1
				batch_z = np.random.uniform(-1, 1, size = (batch_z, z_dim))
				_ = sess.run(d_optimizer, feed_dict = {input_real:batch_images, input_z: batch_z, lr: learning_rate})

				if step%10 == 0:
					train_loss_d = d_loss.eval({input_real:batch_images, input_z: batch_z})
					train_loss_g = g_loss.eval({input_z: batch_z})

				if step%100 == 0:
					count = step
					show_generator_output(sess, show_n_images, input_z, data_shape[3], data_image_mode)

			_ = sess.run(g_optimizer, feed_dict = {input_real: batch_images, input_z: batch_z, lr: learning_rate})




batch_size = 64
z_dim = 128
learning_rate = 0.0001
beta1 = 0.6

epochs = 2


data_dir = './data'
celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)