import tensorflow as tf
import numpy as np

xs = np.linspace(-3,3,100)
ys = np.sin(xs) + np.random.uniform(-0.5,0.5,100)

graph = tf.Graph()
with graph.as_default():
    input_X = tf.placeholder(tf.float32, name='input_X')
    input_Y = tf.placeholder(tf.float32,name='input_Y')
    W = tf.Variable(tf.random_normal([1]),name='weight')
    b = tf.Variable(tf.random_normal([1]),name='bias')
    Y_pred = tf.add(tf.multiply(input_X, W, name='mul'),b, name='add')
    cost = tf.reduce_sum(tf.pow(Y_pred - input_Y,2))/(100-1)
    tf.summary.scalar('cost',cost)
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    merged = tf.summary.merge_all()

n_epoches = 500
with tf.Session(graph=graph) as sess:
    train_writer = tf.summary.FileWriter('linear/train',sess.graph)
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for epoch_i in range(n_epoches):
        for (x,y) in zip(xs,ys):
            sess.run(optimizer,feed_dict={input_X:x,input_Y:y})
        summary, training_cost = sess.run([merged,cost],feed_dict={input_X:xs, input_Y:ys})
        train_writer.add_summary(summary, epoch_i)
        print(training_cost)
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
    print('W:')
    print(sess.run(W))
    print('b:')
    print(sess.run(b))