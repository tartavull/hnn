import numpy

class inputLayer:

	def __init__(self, net, input_len):
		
		self.net = net
		self.input_len = input_len


		return
	
	def forward(self, input):
		#Generate random array
		random = numpy.random.rand(self.input_len)

		#Consider a neuron was active if the generated random number
		#Is smaller than the input.

		#This means if the input is close to 1, most of the time it will spiked
		#Otherwise, is unprobable it will spike.

		#We could apply a function to the input, to make it more close to .5
		#This way it would be less determnistic 
		self.spiked = random < input

		return self.spiked

	def backward(self, reward):
		#Because this layers hasn't have any learnable parameter
		#We don't implement any backward pass
		pass

class fullyConectedLayer:

	def __init__(self, net , input_len , layer_len ):

		self.net = net
		self.input_len = input_len
		self.layer_len = layer_len

		#This atribute is explained in the backward 
		#if max weight is too large  the exponential  to overflow
		self.max_weight = 100.0
		self.min_weight = 0.0
		self.mid_weight = (self.max_weight - self.min_weight) / 2.0

		##Initialize weights

		#Excitary weights goes from input to the output.
		#Excitatory weights describes the probability that a neuron in this layer will fired
		#given that an input neuron layered fired.
		self.excitatory = numpy.random.uniform(self.mid_weight * 0.1 , self.mid_weight * 0.2 ,size=(self.input_len, self.layer_len))
		#
		# self.excitatory = numpy.array([[ 5100., -1000., -1000., -1000., -1000., -1000.],
		# 															 [-1000.,  5100., -1000., -1000., -1000., -1000.],
		# 															 [-1000., -1000.,  5100., -1000., -1000., -1000.],
		# 															 [-1000., -1000., -1000.,  5100., -1000., -1000.]]).transpose()
	def forward(self, input):

		self.input = input
		#We allow excitatory weights to be between 0 and 10
		#we then substract 5.0 element wise, shifting the range from -5.0 to 5.0
		#after that we apply a sigmoid function so it goes from ~0.0 to ~1.0
		#this is then consider as the probabillity of firing, base on this
		#We randomly make it fired.

		output = numpy.dot(self.input, self.excitatory) - self.mid_weight

		#larger tempeature makes the output smaller, 
		#making the probability of firing closer to 0.5, 
		#point where the variance is larger.
		#This makes our network have a more random behavior, which is good for 
		#exploring possible solutions. 
		#Temperature should drop to 1.0 as trainning progress.
		probabillity = 1/(1 + numpy.exp(-1 * output / self.net.temperature))

		#Consider a neuron was active if the generated random number
		#Is smaller than the input.
		random = numpy.random.rand(self.layer_len)

		#This means if the input is close to 1, most of the time it will spiked
		#Otherwise, is unprobable it will spike.
		self.spiked = random < probabillity

	

		spiked = numpy.where(self.spiked == True)[0]
		if spiked.shape[0] == 0:
			#self.excitatory += (self.max_weight - self.excitatory) * 0.1
			random_pick = numpy.random.randint(0, self.layer_len)
			self.spiked[random_pick] = True

		if spiked.shape[0] > 1:
			random_pick = spiked[ numpy.random.randint(0, spiked.shape[0]) ]
			self.spiked = numpy.zeros( shape= (self.layer_len))
			self.spiked[random_pick] = True


		return self.spiked

	def backward(self, reward): 
		#If the reward it positive in means the neuron which spiked made a good job.
		#and the strength of the weights of the neurons which spiked should be increased.
		#the change in the weights gets smaller, the closer it gets to the limits
		if reward > 0.0:
			#We will update it proportionally to the difference to the max weight, this proportionality is set with the reward.
			#If the reward is 1, our weights will get the self.max_weight in only one run, which is probably too fast.
	
			#Given that only some of the cells in the input spiked (self.input), causing some of the output cells to spike (self.spiked),
			#We want to increase the strength of the weight which connects the spiked input cells to the spiked output cells.
			#We call this weights active_weights
			active_weights = self.input[:, None] * self.spiked

			self.excitatory += (self.max_weight - numpy.sum( self.excitatory , axis=0)[None, :]) * reward * active_weights

			#We also want to decrease the strenght of the weight which connects to neurons which fired in the input 
			#to neurons which didn't fired in the output
			inactive_weights =  self.input[:, None] * (self.spiked == False)
			self.excitatory -= (self.excitatory + numpy.sum( self.excitatory , axis=0)[None, :]) * reward * inactive_weights

		if reward < 0.0:
			#If the reward it's negative, we want to make the weights responsable of firing smaller.We do this proportional to 
			#the difference to the mimun weight and to the reward
			active_weights = self.input[:, None] * self.spiked
			self.excitatory += ( self.excitatory + numpy.sum( self.excitatory , axis=0)[None, :]) * reward * active_weights
			#self.excitatory =  self.excitatory + (self.max_weight - self.excitatory) * (-1 * reward)  * self.input[:, None] * (self.spiked == False)

		return

class outputLayer:

	def __init__(self, net ,input_len):

		self.input_len = input_len

		self.net = net

		return

	def forward(self, input):

		#The ouput of the above layer might have many spiked neurons (true values)
		#in that case we will return only one random value
		spiked = numpy.where(input == True)[0]

		#There were no spikes, return a random label
		if spiked.shape[0] == 0: 
			#return numpy.random.randint(0, input.shape[0])
 			self.net.increaseTemperature()

			return None

		#Just one spiked, ideal case
		if spiked.shape[0] == 1:
			return spiked[0]

		#More than one spiked, pick a random from this subset
		if spiked.shape[0] > 1:
			#random_pick = numpy.random.randint(0, spiked.shape[0])
			#return spiked[random_pick]
			self.net.decreaseTemperature()

			return None

	def backward(self, reward):
		#Because this layers hasn't have any learnable parameter
		#We don't implement any backward pass
		pass