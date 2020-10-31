import tensorflow as tf
import numpy as np
import pandas as pd
batch_size = 8

data = pd.read_csv('../data/data_multivar.txt')
data = data.values
data2 = []
for i in range(len(data)):
    if data[i,2] <= 1:
        data2.append(data[i])
data = np.array(data2)



w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name = 'y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +(1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)


dataset_size = 128

X = np.array(data[:,0:2], dtype=float)
Y = np.array(data[:,2], dtype=int)
Y = Y.reshape(len(Y), 1)
print(X)
print(Y)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))



    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        # print("%d,%d"%(start, end))
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))