from ma import *
from tqdm import *
from sklearn.metrics import classification_report
from logger import *


class Trainer:

	def __init__( self , nn , dataset , log , moving_average_window = 100 , positive_reward = .01 , negative_reward = -.02):

		self.nn = nn
		self.dataset = dataset

		self.number_of_examples = dataset.shape[0]

		self.ma_success_rate = []
		for label in range(self.number_of_examples):
			self.ma_success_rate.append(MovingAverage(moving_average_window))



		self.positive_reward = positive_reward
		self.negative_reward = negative_reward

		self.log = log
		if self.log:
			self.logger = Logger()


	def train (self, epochs ):

		for epoch in tqdm(range(epochs), mininterval=5.0):

			for example in range( self.number_of_examples ):

				desiredTarget, actualTarget = self.processExample( example )

				if actualTarget == None:
					continue

				self.updateSucessRate ( desiredTarget , actualTarget )
				
				reward = self.getReward( desiredTarget , actualTarget )

				self.nn.backward(reward)

				if self.log and  example == 0:
					self.logger.logWeight(self.nn.layers[1].excitatory)
					self.logger.logReward(reward)

	def processExample (self, example ):

		input = self.getInput( example )
		desiredTarget = self.getTarget( example )

		actualTarget = self.nn.forward( input )

		return desiredTarget, actualTarget

	def getTarget( self, example):

		return self.dataset[example][0]

	def getInput ( self, example):

		return self.dataset[example][1]

	def updateSucessRate(self, desiredTarget , actualTarget ):

		if  desiredTarget == actualTarget:
			self.ma_success_rate[desiredTarget].append(1.0)

		else :
			self.ma_success_rate[desiredTarget].append(0.0)

	def getReward(self, desiredTarget , actualTarget ):

		if desiredTarget == actualTarget:
			# Positive reward
		 	return self.positive_reward * (0.9 *  (1.0 - self.ma_success_rate[desiredTarget].mean) + 0.1) 
		else:
			#Negative reward
			return self.negative_reward * (0.9 * self.ma_success_rate[desiredTarget].mean + 0.1)


	def test(self):

		self.actualTargets = []
		self.desiredTargets = []

		for epoch in range(100):
			for example in range( self.number_of_examples ):

				desiredTarget, actualTarget = self.processExample( example )
				if actualTarget == None:
					continue
				
				self.actualTargets.append(actualTarget)
				self.desiredTargets.append(desiredTarget)

		self.report()

	def report(self):

		for label in range(self.number_of_examples):

			print 'label ', label , ' % correct predictions ', self.ma_success_rate[label].mean

		print classification_report (self.desiredTargets, self.actualTargets)
		from sklearn.metrics import confusion_matrix
		print confusion_matrix (self.desiredTargets, self.actualTargets )