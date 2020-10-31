#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from image_change import preprocess_for_train
from fo_inference import inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99

train_files = tf.train.match_filenames_once("/root/PycharmProjects/untitled/test/flo_data_an/data_tfrecords/train_*")
test_files = tf.train.match_filenames_once("/root/PycharmProjects/untitled/test/flo_data_an/data_tfrecords/test_*")


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image' : tf.FixedLenFeature([], tf.string),
            'label' : tf.FixedLenFeature([], tf.int64),
            'height' : tf.FixedLenFeature([], tf.int64),
            'width' : tf.FixedLenFeature([], tf.int64),
            'channels' : tf.FixedLenFeature([], tf.int64)

        }
    )

    decoded_image = tf.decode_raw(features['image'], tf.uint8)
    print([features['height'], features['width'], features['channels']])
    decoded_image.set_shape([features['height'], features['width'], features['channels']])
    label = features['label']
    return decoded_image, label

image_size = 299
batch_size = 100
shuffle_buffer = 10000

dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)
# print(dataset.shape)

dataset = dataset.map(
    lambda image, label :(
        preprocess_for_train(image, image_size, image_size, None), label
    )
)
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

learning_rate = 0.01
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
logit = inference(image_batch, True, regularizer)
global_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.argmax(label_batch, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               10000 / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.control_dependencies([train_step, variable_averages_op]):
    train_op = tf.no_op(name='train')

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))

    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break





