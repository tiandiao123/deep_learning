import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers



# classifier_url ="https://tfhub.dev/google/tf2-preview/resnet_v2_101/classification/2" 
IMAGE_SHAPE = (256, 256)

# classifier = tf.keras.Sequential([
#     hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
# ])

import numpy as np
import PIL.Image as Image

# grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
# grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
# grace_hopper = np.array(grace_hopper)/255.0
# result = classifier.predict(grace_hopper[np.newaxis, ...])
# print(result)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

print("create transfer learning model: ")
feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4" 

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(256,256,3))
feature_batch = feature_extractor_layer(image_batch)
feature_extractor_layer.trainable = False
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes)
])

print(model.summary())


model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[batch_stats_callback])



import time
t = time.time()

export_path = "./tmp/saved_models/{}".format(int(t))
tf.saved_model.save(model, "./tmp/inception_v1")







print("got my model!")