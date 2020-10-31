#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np


file_names = tf.train.match_filenames_once("/root/PycharmProjects/untitled/test/flo_data_an/data_tfrecords/train_*")

init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(file_names))