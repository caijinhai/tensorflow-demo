import tensorflow as tf



m = tf.Variable(0.)
b = tf.Variable(0.)

input_x = tf.placeholder(tf.int32)
input_y = tf.placeholder(tf.int32)

caculate_y = m * input_x + b
loss = tf.square(input_y - caculate_y)

optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(init)


import random

random_m = random.random()
random_b = random.random()

for i in range(1000):
    x = random.random()
    y = random_m * x + random_b
    
    _loss, _ = sess.run([loss, train_op], feed_dict={input_x: x, input_y: y})
    print(i, _loss)

print("True parameter: m=%.4f, b=%.4f" % (random_m, random_b))
print("Learn parameter: m=%.4f, b=%.4f" % (tuple(sess.run(m, b))))