#!/usr/bin/python

########################################################
# CS5783: Machine Learning #
# Assignment 2#
# Problem 3: More Nearest Neighbours #
# Sk. Mashfiqur Rahman (CWID: A20102717) #
# collect data from: http://yann.lecun.com/exdb/mnist/#
#######################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold


def dist(xi, xj):
    c = [np.sqrt(np.sum((xi - xjd) ** 2.)) for xjd in xj]
    return np.array(c)

# Training images
training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images = bytearray(training_images)
training_images = training_images[16:]
training_images_file.close()

training_images = np.array(training_images,"float64")
d = 28 * 28
n = training_images.shape[0] / d
training_images = training_images.reshape(int(n), d)

# Training labels
training_labels_file = open('train-labels.idx1-ubyte','rb')
training_labels = training_labels_file.read()
training_labels = bytearray(training_labels)
training_labels = training_labels[8:]
training_labels_file.close()

training_labels = np.array(training_labels)
training_labels = training_labels.reshape(training_labels.shape[0], 1)

class1 = training_images[(training_labels == 1).reshape(-1)]
class2 = training_images[(training_labels == 2).reshape(-1)]
class3 = training_images[(training_labels == 7).reshape(-1)]

x = np.concatenate((class1[:200], class2[:200], class3[:200]), axis=0)
y = np.concatenate((np.ones((200, 1), 'int64'), np.ones((200, 1), 'int64') + 1, np.ones((200, 1), 'int64') + 6), axis=0)

K = np.array([1, 3, 5, 7, 9])  # array of different nearest neighbor classifiers
m = K.shape[0]
avg_performance = np.zeros(m)
performance = np.zeros(m)
for r in range(m):
    for i in range(m):
        kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
        for train_index, test_index in kf.split(x):  # split and shuffle training and test data into folds
            train_x, test_x = x[train_index], x[test_index]
            train_y, test_y = y[train_index], y[test_index]  # combination of labels 1, 2 and 7
        validation_y = np.zeros(test_y.shape)
        for j in range(test_x.shape[0]):
            d = np.argsort(dist(test_x[j], train_x))
            d = d[:K[r]]  # take range up to respective model number
            count = np.zeros(10)
            for p in d:
                count[train_y[p]] += 1   # count Pth (nearest distances) number of elements in train_y (1/7/9)
            validation_y[j] = np.argmax(count)  # choose the respective best model
        performance[i] = float(sum(validation_y == test_y)) / test_y.shape[0]  # performance analysis on test and training y for folds 1 to 5
    avg_performance[r] = (float(sum(performance)) / m)*100.
    print("Candidate model", K[r], ":", "average performance", avg_performance[r], "%")

best_model = K[np.argmax(avg_performance)]
print("best candidate model: k =", best_model)

# Test images
test_images_file = open('t10k-images.idx3-ubyte','rb')
test_images = test_images_file.read()
test_images = bytearray(test_images)
test_images = test_images[16:]
test_images_file.close()

test_images = np.array(test_images,"float64")
d = 28 * 28
n = test_images.shape[0] / d
test_images = test_images.reshape(int(n), d)

# Test labels
test_labels_file = open('t10k-labels.idx1-ubyte','rb')
test_labels = test_labels_file.read()
test_labels = bytearray(test_labels)
test_labels = test_labels[8:]
test_labels_file.close()

test_labels = np.array(test_labels)
test_labels_1 = test_labels.reshape(test_labels.shape[0], 1)

class1 = test_images[(test_labels == 1).reshape(-1)]
class2 = test_images[(test_labels == 2).reshape(-1)]
class3 = test_images[(test_labels == 7).reshape(-1)]

test_x = np.concatenate((class1[:50], class2[:50], class3[:50]), axis=0)
test_y = np.concatenate((np.ones((50, 1), 'int64'), np.ones((50, 1), 'int64') + 1, np.ones((50, 1), 'int64') + 6), axis=0)

r = np.arange(3*50)
p = np.arange(3*200)
np.random.shuffle(r)
test_x = test_x[r.reshape(-1)]
test_y = test_y[r.reshape(-1)]
x = x[p.reshape(-1)]
test_y_hat = np.zeros(test_y.shape)
performance = 0.

for j in range(test_x.shape[0]):
    d = np.argsort(dist(test_x[j], x))
    d = d[:best_model]
    count = np.zeros(10)
    for p in d:
        count[y[p]] += 1
    test_y_hat[j] = np.argmax(count)
performance = (float(sum(test_y_hat == test_y)) / test_y.shape[0])*100.
print("New performance:", performance, "%")

fig = plt.figure()
correctly_classified_one = test_x[np.logical_and((test_y_hat == 1).reshape(-1), (test_y == 1).reshape(-1))]
plt.imshow(correctly_classified_one[1].reshape(28, 28), cmap=plt.cm.bone)
plt.title("Correct classification of one")
plt.show()

fig = plt.figure()
incorrectly_classified_one = test_x[np.logical_and((test_y_hat == 1).reshape(-1), (test_y != 1).reshape(-1))]
plt.imshow(incorrectly_classified_one[1].reshape(28, 28), cmap=plt.cm.bone)
plt.title("Incorrect classification of one")
plt.show()

fig = plt.figure()
correctly_classified_two = test_x[np.logical_and((test_y_hat == 2).reshape(-1), (test_y == 2).reshape(-1))]
plt.imshow(correctly_classified_two[1].reshape(28, 28), cmap=plt.cm.bone)
plt.title("Correct classification of two")
plt.show()

fig = plt.figure()
correctly_classified_seven = test_x[np.logical_and((test_y_hat == 7).reshape(-1), (test_y == 7).reshape(-1))]
plt.imshow(correctly_classified_seven[1].reshape(28, 28), cmap=plt.cm.bone)
plt.title("Correct classification of seven")
plt.show()

fig = plt.figure()
incorrectly_classified_seven = test_x[np.logical_and((test_y_hat == 7).reshape(-1), (test_y != 7).reshape(-1))]
plt.imshow(incorrectly_classified_seven[1].reshape(28, 28), cmap=plt.cm.bone)
plt.title("Incorrect classification of seven")
plt.show()

fig = plt.figure()
incorrectly_classified_two = test_x[np.logical_and((test_y_hat == 2).reshape(-1), (test_y != 2).reshape(-1))]
print(incorrectly_classified_two.shape)
plt.imshow(incorrectly_classified_two[1].reshape(28, 28), cmap=plt.cm.bone)
plt.title("Incorrect classification of two")
plt.show()
