#-*-coding:utf-8-*-
import tensorflow as tf

files = tf.train.match_filenames_once('./TFRecord_data/data.tfrecords-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i' : tf.FixedLenFeature([], tf.int64),
        'j' : tf.FixedLenFeature([], tf.int64)
    }
)

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    print(sess.run(files))
