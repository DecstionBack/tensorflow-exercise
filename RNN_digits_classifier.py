import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#使用RNN做分类最重要的应该就是确定输入的向量和标签向量，具体是哪儿的数据不重要，都是ndarray类型的就可以了

# hyperparameters
lr = 0.001 # learning rate Adam算法会用到
batch_size = 128
n_inputs = 28 # 28 cl
n_steps = 28 # 28 rows -> time stamps
n_hidden_unins = 128 # hidden units
n_classes = 10
# tf input
xs =tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="inputs")
ys =tf.placeholder(tf.float32, [None, n_classes], name="outputs")
# W & b
weights = {
    'in': tf.Variable(tf.random_uniform([n_inputs, n_hidden_unins], -1.0, 1.0), name="in_w"),  #Wxh
    'out': tf.Variable(tf.random_uniform([n_hidden_unins, n_classes], -1.0, 1.0), name="out_w"), #Why
}
b = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unins]), name="in_bias"),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name="out_bias"),
}

#这儿不定义RNN这个函数也可以
# def RNN(X, weights, bias):
#     # hidden_layer for input
#     # X : (128, 28, 28)
#     with tf.name_scope("inlayer"):
#         X = tf.reshape(X, [-1, n_inputs])  #X为N*n_inputs维度的矩阵
#         X_in = tf.matmul(X, weights['in']) + b['in']
#         X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unins])
#     # RNN cell
#     with tf.name_scope("RNN_CELL"):
#         lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unins)
#         # _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
#         # ouputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state)
#         outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
#     # out layer
#     with tf.name_scope('outlayer'):
#         results = tf.matmul(states[1], weights['out']) + b['out']
#     return results
# pred = RNN(xs, weights, b)


with tf.name_scope("inlayer"):
    X = tf.reshape(xs, [-1, n_inputs])  #X为N*n_inputs维度的矩阵
    X_in = tf.matmul(X, weights['in']) + b['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unins])
# RNN cell
with tf.name_scope("RNN_CELL"):
    #lstm_cell只是声明了cell的类型，下面的dynamic_rnn会实际去运行，因为一个cell需要有输入、cell、输出，
    # BasicLSTMCell只是制定了cell类型，然后输入数据以后运行得到结果，后面的过程就由dynamic_rnn来承担了
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unins)

    # _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # ouputs, states = tf.nn.dynamic_rnn(lstm_cell, xs_in, initial_state=_init_state)
    #outputs是最后的输出结果，而states是最后的状态C
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
# out layer
with tf.name_scope('outlayer'):
    results = tf.matmul(states[1], weights['out']) + b['out']
pred=results


# cost
#这里logits和labels需要明确指明，新版本更改
#tf.nn.softmax_cross_entropy_with_logits计算交叉熵，logits是softmax以前的结果，ys是标准答案，计算时会先对logits变量使用softmax然后再计算交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
# accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# run
import time
init = tf.global_variables_initializer()
epochs = 15
st = time.time()
print(type(mnist.test.labels))

#前面只是在搭建计算图并没有运行，即使调用RNN，甚至是计算cost，只有在下面tf.Session()才真正计算、
#和线性回归一样只是搭建起来了而已，并没有计算，这里可以想前面只是构建图
with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)
    batch = mnist.train.num_examples / batch_size
    for epoch in range(epochs):
        for i in range(int(batch)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([batch_size, n_inputs, n_steps])
            sess.run(train_op, feed_dict={xs: batch_x, ys: batch_y})
        print ('epoch:', epoch+1, 'accuracy:', sess.run(accuracy, feed_dict={xs: mnist.test.images.reshape([-1, n_steps, n_inputs]), ys: mnist.test.labels}))
    end = time.time()
    print ('*' * 30)
    print ('training finish.\ncost time:',int(end-st), 'seconds\naccuracy:', sess.run(accuracy, feed_dict={xs: mnist.test.images.reshape([-1, n_steps, n_inputs]), ys: mnist.test.labels}))