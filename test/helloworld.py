import tensorflow as tf
hello = tf.constant("hello worlf !!!")
sess = tf.Session()
print(sess.run(hello))