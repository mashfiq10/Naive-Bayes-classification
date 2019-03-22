#!/usr/bin/python

#######################################################
# CS5783: Machine Learning #
# Assignment 2#
# Problem 3: Naive Bayes #
# Sk. Mashfiqur Rahman #
# collect data from: http://yann.lecun.com/exdb/mnist/#
#######################################################


import numpy as np
import matplotlib.pyplot as plt

# Training images
training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images = bytearray(training_images)
training_images = training_images[16:]
training_images_file.close()

training_images = np.array(training_images,"float64") > 100.
d = 28 * 28
n = training_images.shape[0] / d

# plotting samples
i = 1
fig = plt.figure()
for k in range(20*20):
    a = fig.add_subplot(20, 20, k + 1)  # cols, rows, in
    plt.imshow(training_images[(i-1)*d: i*d].reshape(28,28), cmap=plt.cm.bone)  # shape 784 into 28,28
    a.set_axis_off()
    i += 1
plt.show()
training_images = training_images.reshape(int(n), d)

# Training labels
training_labels_file = open('train-labels.idx1-ubyte','rb')
training_labels = training_labels_file.read()
training_labels = bytearray(training_labels)
training_labels = training_labels[8:]
training_labels_file.close()

training_labels = np.array(training_labels)
training_labels_1 = training_labels.reshape(training_labels.shape[0], 1)

k = 10
counts = np.bincount(training_labels)
prior = counts / n
log_prior = np.log10(prior)
theta = np.zeros((k, d), "float64")
for i in range(k):
        mask = (training_labels == i)
        #  (nk + 1)/(Nk+k)
        theta[i] += (np.sum(training_images[mask], axis=0, dtype="float64") + 1.) / (counts[i] + k)
log_theta = np.log10(theta)
log_complement = np.log10(1. - theta)

# Test images
test_images_file = open('t10k-images.idx3-ubyte','rb')
test_images = test_images_file.read()
test_images = bytearray(test_images)
test_images = test_images[16:]
test_images_file.close()

test_images = np.array(test_images,"float64") > 100.
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

test_labels_hat = np.zeros(test_labels_1.shape)
for i in range(test_images.shape[0]):
    log_likelihood = np.sum(log_theta[:, test_images[i].reshape(-1)], axis=1) + np.sum(log_complement[:, np.logical_not(test_images[i].reshape(-1))], axis=1)
    log_posterior = log_prior + log_likelihood
    test_labels_hat[i] = np.argmax(log_posterior)

c = np.sum(test_labels_1 == test_labels_hat)
print("Naive Bayes classifier classification accuracy:", float(c / len(test_labels_1))*100., '%')
