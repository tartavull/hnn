from tqdm import *
import numpy

import dataset
from nets import simpleNet


mnist = dataset.load_mnist()

nn = simpleNet(architecture=numpy.array([784 ,100, 10]))

for epoch in range(10):
	success = numpy.zeros(shape=(mnist[0][0].shape[0],))
	for example in tqdm(range(mnist[0][0].shape[0])):  #
		
		input = mnist[0][0][example,:]
		target = mnist[0][1][example]
		
		output = nn.forward(input)

		reward = -0.00001
		if output == target:
			reward = 0.0001
			success[example] += 1

		nn.backward(reward)

	print numpy.sum(success) / success.shape[0]