
import math

learning_rate = 0.01
epochs = 1000

# data
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
m = len(train_X)

w = 1
b = 1
minimum = 0
for epoch in range(epochs):
    # y = f(x) = wx + b
    cost = 0
    gradient_w = 0
    gradient_b = 0

    for (x, y) in zip(train_X, train_Y):
        cost += math.pow(((w * x + b) - y), 2) / (2 * m)
        gradient_w += ((w * x + b) - y) * x / m
        gradient_b += ((w * x + b) - y) / m

    # print('gradient w: %lf\ngradient b: %lf' %(gradient_w, gradient_b))
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    minimum = cost
    if (epoch + 1) % 100 == 0:
        print("after %d optimize:\n\tcost:%f\n\t   w:%f\n\t   b:%f" % (epoch + 1, cost,
                                                                       w, b))

print('cost:', minimum, '\ny = %lfx %+lf' % (w, b))


'''
the answer of another code that use tensorflow module
after 1000 optimize:
cost:0.077039
w:0.245932
b:0.827807 
'''