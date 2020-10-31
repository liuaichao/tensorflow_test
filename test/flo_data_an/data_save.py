#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import os
import glob
import gc
from tensorflow.python.platform import gfile

def _int64_feature(value):
    return tf.train.Feature(int64_list =tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

INPUT_DATA = '/root/PycharmProjects/untitled/data/flower_photos'
TEST_PERCENTAGE = 10

def create_image_lists(sess, testing_percentage):
    i = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    current_label = 0

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        for file_name in file_list:
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)
            #
            #
            # print(image_value.shape[0])
            # print(image_value.shape[1])
            # print(image_value.shape[2])
            chance = np.random.randint(100)

            if chance < testing_percentage:
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            print("No:%d"%i)
            i = i+1


        writer_train = tf.python_io.TFRecordWriter('./train_{}.tfrecords'.format(str(dir_name)))
        writer_test = tf.python_io.TFRecordWriter('./test_{}.tfrecords'.format(str(dir_name)))
        training_images = np.array(training_images)
        testing_images = np.array(testing_images)

        for index in range(len(training_images)):
            image_raw_train = training_images[index].tostring()
            example_train = tf.train.Example(features=tf.train.Features(feature={
                'image' : _bytes_feature(image_raw_train),
                'label' : _int64_feature(training_labels[index]),
                'height' : _int64_feature(training_images[index].shape[0]),
                'width' : _int64_feature(training_images[index].shape[1]),
                'channels' : _int64_feature(training_images[index].shape[2])

            }))

            writer_train.write(example_train.SerializeToString())
            del example_train
        writer_train.close()
        del writer_train

        del training_images
        del training_labels
        training_images = []
        training_labels = []

        for index in range(len(testing_images)):
            image_raw_test = testing_images[index].tostring()
            example_test = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image_raw_test),
                'label': _int64_feature(testing_labels[index]),
                'height': _int64_feature(testing_images[index].shape[0]),
                'width': _int64_feature(testing_images[index].shape[1]),
                'channels': _int64_feature(testing_images[index].shape[2])

            }))
            writer_test.write(example_test.SerializeToString())
            del example_test

        writer_test.close()
        del writer_test

        del testing_images
        del testing_labels
        a = gc.collect()
        print("the rabsh is %d"%a)
        testing_images = []
        testing_labels = []
        print("alrealy saved")

        current_label += 1

    # state = np.random.get_state()
    # np.random.shuffle(training_images)
    # np.random.set_state(state)
    # np.random.shuffle(training_labels)


def main():
    with tf.Session() as sess:
        create_image_lists(sess, TEST_PERCENTAGE)

        print("alrealy saved")

if __name__ == '__main__':
    main()