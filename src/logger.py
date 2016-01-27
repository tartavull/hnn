import numpy
import matplotlib.pyplot as plt

class Logger:

	def __init__(self):

		self.weights = []
		self.rewards = []
		self.ma = []
	
	def logWeight(self, weight):

		self.weights.append(weight.flatten().tolist())


	def logReward(self, reward):

		self.rewards.append(reward)

	def logMovingAverage(self, ma):

		self.ma.append(ma)

	def plotWeights(self):

		f, axes = plt.subplots(4, 1)

		for i in range(4):
			axes[i].plot(numpy.asarray(self.weights)[:,i])

		plt.show()

	def plotRewards(self):

		f, axes = plt.subplots(1,1)

		axes.plot(self.rewards)

		plt.show()

	def plot(self):

		subplots = len(self.weights[0]) + 1 
		f, axes = plt.subplots(subplots, 1, sharex=True)

		axes[0].plot(self.rewards, 'r*')


		for i in range(len(self.weights[0])):
			print i
			axes[i+1].plot(numpy.asarray(self.weights)[:,i])

		plt.show()


if __name__ == '__main__':
	logger = Logger()
	logger.plotWeights()