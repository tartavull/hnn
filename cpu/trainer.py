from ma import *
from tqdm import *
from sklearn.metrics import classification_report


class Trainer:

	def __init__( self , nn , dataset , moving_average_window = 10 , positive_reward = 1.0 , negative_reward = -1.0):

		self.nn = nn
		self.dataset = dataset

		self.number_of_examples = dataset.shape[0]

		self.ma_success_rate = []
		for label in range(self.number_of_examples):
			self.ma_success_rate.append(MovingAverage(moving_average_window))

		self.positive_reward = positive_reward
		self.negative_reward = negative_reward

		self.debug = True
		if self.debug:
			self.actualTargets = []
			self.desiredTargets = []



	def train (self, epochs ):

		for epoch in tqdm(range(epochs)):

			for example in range( self.number_of_examples ):

				self.processExample( example )


		if self.debug:
			self.report()


	def processExample (self, example ):

		input = self.getInput( example )
		desiredTarget = self.getTarget( example )

		actualTarget = self.nn.forward( input )

		self.updateSucessRate ( desiredTarget , actualTarget )
		reward = self.getReward( desiredTarget , actualTarget )

		#print 'desired' ,desiredTarget , 'actual', actualTarget , 'reward', reward


		self.nn.backward(reward)

		if self.debug:
			self.actualTargets.append(actualTarget)
			self.desiredTargets.append(desiredTarget)



	def getTarget( self, example):

		return self.dataset[example][0]

	def getInput ( self, example):

		return self.dataset[example][1]

	def updateSucessRate(self, desiredTarget , actualTarget ):

		if  desiredTarget == actualTarget:
			self.ma_success_rate[actualTarget].append(1.0)

		else :
			self.ma_success_rate[actualTarget].append(0.0)

	def getReward(self, desiredTarget , actualTarget ):

		if desiredTarget == actualTarget:
			# Positive reward
		 	return self.positive_reward * (1.0 - self.ma_success_rate[actualTarget].mean) 
		else:
			#Negative reward
			return self.negative_reward *  self.ma_success_rate[actualTarget].mean

	def report(self):

		for label in range(self.number_of_examples):

			print 'label ', label , ' % correct predictions ', self.ma_success_rate[label].mean

		print classification_report (self.desiredTargets, self.actualTargets)