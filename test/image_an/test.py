#-*-coding:utf-8-*-
import tensorflow as tf

test_files = tf.train.match_filenames_once("./cat.jpg")
print(test_files)