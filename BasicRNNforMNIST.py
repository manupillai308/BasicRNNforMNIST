import tensorflow as tf
from sklearn.datasets import fetch_mldata
import numpy as np

data = fetch_mldata('mnist original')
x, _y = data.data, data.target

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit()
for train, test in sss.split(x,_y):
    x_train, y_train = x[train], _y[train]
    x_test, y_test = x[test], _y[test]
    
print(x_train.shape, y_train.shape)
x_train, x_test = np.reshape(x_train, newshape=(-1,28,28)), np.reshape(x_test, newshape=(-1,28,28))
print(x_train.shape, y_train.shape)

n_steps = 28
n_inputs = 28
learning_rate = 0.001
n_epoch = 40

def next_batch(batch_size = 30, x_train = x_train, y_train = y_train):
    prev = 0
    for i in range(batch_size , x_train.shape[0], batch_size):
        yield x_train[prev:i], y_train[prev:i]
        prev = i
tf.reset_default_graph()
X = tf.placeholder(shape=[None, n_steps, n_inputs], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int32)

bcell = tf.contrib.rnn.BasicRNNCell(num_units=50, activation=tf.nn.elu, dtype=tf.float32)

outputs, states = tf.nn.dynamic_rnn(cell=bcell, dtype=tf.float32, inputs=X)
logits = tf.layers.dense(units=10, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),inputs=states)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels=y)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(k=1, targets=y, predictions=tf.nn.softmax(logits))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        batch = next_batch()
        for x_batch, y_batch in batch:
            sess.run(training_op, feed_dict = {X:x_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict = {X:x_train, y:y_train})
        acc_test = accuracy.eval(feed_dict = {X:x_test, y:y_test})
        print('epoch',epoch,'Training accuracy:',acc_train, 'Testing accuracy:',acc_test)