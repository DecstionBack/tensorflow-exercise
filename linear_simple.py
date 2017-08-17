import tensorflow as tf
import numpy as np

x = np.random.random([100])
y = 0.3 * x + 0.1


W = tf.Variable(tf.zeros([1]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]),dtype=tf.float32)
Input_x = tf.placeholder(tf.float32)
Input_y = tf.placeholder(tf.float32)

y_pred = W * Input_x + b

loss = tf.reduce_mean(tf.square(Input_y - y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(500):
        sess.run(optimizer, feed_dict={Input_x:x, Input_y:y})
        #这儿的一定要注意，因为loss不仅仅是一个变量，它是由Input_x和Input_y计算出来的，所以在计算loss的时候也需要将具体的数据传入进去
        print('epoch:', epoch, 'loss:', sess.run(loss, feed_dict={Input_x:x, Input_y:y}), 'W', sess.run(W), 'b', sess.run(b))
