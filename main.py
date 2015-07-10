from tqdm import *
import numpy

import dataset
from nets import simpleNet
from ma import movingAverage


from sklearn.metrics import classification_report
y_true = []
y_pred = []

def monitorMA(example, movingAverage):
	if example not in monitor:
		monitor[example] = list()
	else:
		monitor[example].append(movingAverage)

data = dataset.load_dictionary()

nn = simpleNet(architecture=numpy.array([15 , data.shape[0]]))

success_rate = dict()
monitor = dict()

for epoch in tqdm(range(50000)):
	for example in range(data.shape[0]):

		input = data[example][1]
		target = data[example][0]
		y_true.append(target)

		output = nn.forward(input)
		y_pred.append(output)

		if example not in success_rate:
			success_rate[example] = movingAverage(10000)
		else:
			success_rate[example].append(float(output == target))


		if output == target:
			#Positive reward
		 	reward = 1 * (1.0 - success_rate[example].mean) 
		else:
			#Negative reward
			reward = -1 *  success_rate[example].mean  

		nn.backward(reward)

		# if epoch % 100 == 0:
		# 	monitorMA(example, success_rate[example].mean)

# for example in range(len(monitor)):
# 	print monitor[example]

# print nn.layers[1].excitatory

# for example in range(data.shape[0]):

# 	input = data[example][1]
# 	target = data[example][0]
# 	output = nn.forward(input)
# 	print target , output

print classification_report(y_true, y_pred)