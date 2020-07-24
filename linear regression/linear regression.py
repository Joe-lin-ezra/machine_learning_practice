
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

# hyper-parameter
learning_rate = 0.01
epochs = 1000

# data
train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
sample_len = train_X.shape[0]

# placeholder (X, Y) for training data
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# training data, initial_value set a random number
w = tf.Variable(initial_value=1., name="weight")
b = tf.Variable(initial_value=1., name="bias")

# linear model and cost function
predict = tf.add(tf.multiply(w, X), b)
cost = tf.reduce_sum(tf.pow(predict - Y, 2)) / (2 * sample_len)

# training method
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init)
    # train
    training_cost = 0.
    for epoch in range(epochs):
        for (x, y) in zip(train_X, train_Y):
            session.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = session.run(cost, feed_dict={X: train_X, Y: train_Y})
        # to know if it's normal working
        if (epoch + 1) % 100 == 0:
            print("after %d optimize:\n\tcost:%f\n\t   w:%f\n\t   b:%f" % (epoch + 1, training_cost,
                                                                      session.run(w), session.run(b)))

    # set text color
    print("\033[33mOptimization Complete \033[0m")
    print("\033[34mtraining cost:%f \033[0m" % training_cost)


    # draw the data
    plt.plot(train_X, train_Y, "ro",label="training data")
    plt.plot(train_X, session.run(w) * train_X + session.run(b), label="fitting line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("training data.png")
    plt.show()

    # -------------------------------------not personal code---------------------------------------------------------
    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = session.run(
        tf.reduce_sum(tf.pow(predict - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("\033[34mTesting cost= %f\033[0m" % testing_cost)
    print("Absolute mean square loss \033[31mdifference:%f\033[0m" % abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, session.run(w) * train_X + session.run(b), label='Fitted line')
    plt.legend()
    plt.savefig('test data.png')
    plt.show()
