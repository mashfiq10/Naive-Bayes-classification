#!/usr/bin/python

########################################################
# Naive Bayes Gaussian #
# Sk. Mashfiqur Rahman (CWID: A20102717) #
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

mask = np.random.random(size=(training_images.shape[0])) < 0.9
x_training = training_images[mask.reshape(-1)]
y_training = training_labels[mask.reshape(-1)]
mask = np.logical_not(mask)
x_test = training_images[mask.reshape(-1)]
y_test = training_labels[mask.reshape(-1)]

k = 2
N = 1000
class1 = x_training[(y_training == 5).reshape(-1)]
class2 = x_training[(y_training != 5).reshape(-1)]

x_training = np.concatenate((class1[:N], class2[:N]), axis=0)
y_training = np.concatenate((np.ones((N, 1), 'int64'), np.zeros((N, 1), 'int64')), axis=0)

mean = np.zeros(shape=(k, x_training.shape[1]))
mean[0] = np.mean(x_training[:N], axis=0, dtype="float64")
mean[1] = np.mean(x_training[N:], axis=0, dtype="float64")
std0 = np.std(x_training[:N], dtype="float64")
std1 = np.std(x_training[N:], dtype="float64")
var1 = float(std0)**2
var2 = float(std1)**2
pi = 3.1415926

mask = (y_test == 5)
y_test[mask] = 1
y_test[np.logical_not(mask)] = 0
y_hat = np.zeros(y_test.shape)

log_likelihood_ratio = np.zeros(y_test.shape)
tau = np.array([5., 2., 1., 0.5, .2])
tau1 = np.array([55., 45., 35., 25., 15., 5., 2., 1., 0.5, .1, -1., -10., -40., -60., -100.])

for i in range(x_test.shape[0]):
    pdf1 = np.array(np.exp(-(x_test[i].reshape(-1)-mean[0,:])**2/(2*var1)) / (2*pi*var1)**.5)
    pdf2 = np.array(np.exp(-(x_test[i].reshape(-1)-mean[1,:])**2/(2*var2)) /(2*pi*var2)**.5)
    log_likelihood_ratio[i] = np.sum(np.log10(pdf1)) - np.sum(np.log10(pdf2))

r = 0
x = np.zeros(5)
y = np.zeros(5)

for k in tau:
    mask = (log_likelihood_ratio >= k)
    y_hat[mask] = 1
    y_hat[np.logical_not(mask)] = 0

    TP = np.logical_and(y_test.reshape(-1) == 1, y_hat.reshape(-1) == 1)
    TN = np.logical_and(y_test.reshape(-1) == 0, y_hat.reshape(-1) == 0)
    FP = np.logical_and(y_test.reshape(-1) == 0, y_hat.reshape(-1) == 1)
    FN = np.logical_and(y_test.reshape(-1) == 1, y_hat.reshape(-1) == 0)

    FPR = float(sum(FP)) / (sum(TN) + sum(FP))
    TPR = float(sum(TP)) / (sum(TP) + sum(FN))

    print(r+1, '.', 'FPR:', FPR, ',', 'TPR:', TPR)
    x[r] = FPR
    y[r] = TPR
    r += 1

plt.figure(figsize=(16, 12))
plt.plot(x, y, 'b-')
# plt.ylim(ymax = 1.0, ymin = 0.0)
# plt.xlim(xmax = 1.0, xmin = 0.0)
plt.title("ROC curve for given situations", fontsize=24)
plt.xlabel("FPR", fontsize=22)
plt.ylabel("TPR", fontsize=22)
plt.show()

r = 0
x = np.zeros(15)
y = np.zeros(15)
for k in tau1:
    mask = (log_likelihood_ratio >= k)
    y_hat[mask] = 1
    y_hat[np.logical_not(mask)] = 0

    TP = np.logical_and(y_test.reshape(-1) == 1, y_hat.reshape(-1) == 1)
    TN = np.logical_and(y_test.reshape(-1) == 0, y_hat.reshape(-1) == 0)
    FP = np.logical_and(y_test.reshape(-1) == 0, y_hat.reshape(-1) == 1)
    FN = np.logical_and(y_test.reshape(-1) == 1, y_hat.reshape(-1) == 0)

    FPR = float(sum(FP)) / (sum(TN) + sum(FP))
    TPR = float(sum(TP)) / (sum(TP) + sum(FN))

    x[r] = FPR
    y[r] = TPR
    r += 1

plt.figure(figsize=(16, 12))
plt.plot(x, y, 'b-')
plt.title("Model ROC curve", fontsize=24)
plt.xlabel("FPR", fontsize=22)
plt.ylabel("TPR", fontsize=22)
plt.show()
