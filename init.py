import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy
import time

if __name__ == '__main__':
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	train_accuracies = []
	train_entropies = []

	test_accuracies = []
	test_entropies = []
	batches = []

	fig = plt.figure()


	plt.style.use('ggplot')


	X = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	init = tf.initialize_all_variables()

	Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

	Y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

	is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

	optimizer = tf.train.GradientDescentOptimizer(0.003)
	train_step = optimizer.minimize(cross_entropy)


	sess = tf.Session()

	sess.run(init)

	for i in range(1000):
		batch_X, batch_Y = mnist.train.next_batch(100)
		train_data = {X: batch_X, Y_: batch_Y}

		sess.run(train_step, feed_dict=train_data)

		a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
		a *= 100
		print "Training accuracy : ", a
		print "Training entropy  : ", c

		batches.append(i)
		train_accuracies.append(a)
		train_entropies.append(c)

		test_data = {X: mnist.test.images, Y_: mnist.test.labels}

		a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

		a *= 100
		test_accuracies.append(a)
		test_entropies.append(c)


		ax1 = fig.add_subplot(121)
		ax1.plot(batches, train_accuracies, 'r-')
		ax1.plot(batches, test_accuracies, 'g-')

		ax2 = fig.add_subplot(122)
		ax2.plot(batches, train_entropies, 'r-')
		ax2.plot(batches, test_entropies, 'g-')

		plt.tight_layout()

		plt.draw()
		plt.pause(1e-17)

		print "Testing accuracy : ", a
		print "Testing entropy  : ", c
plt.show()




