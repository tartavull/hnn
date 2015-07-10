import numpy

class inputLayer:

	def __init__(self, input_len):
		self.input_len = input_len;
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

	def __init__(self, input_len , layer_len ):
		self.input_len = input_len;
		self.layer_len = layer_len;

		##Initialize weights

		#Excitary weights goes from input to the output.
		#Excitatory weights describes the probability that a neuron in this layer will fired
		#given that an input neuron layered fired.
		self.excitatory = numpy.random.uniform(4.0,5.0,size=(self.input_len, self.layer_len))

		#Inhibitory weights connect neurons from within this layers.
		#This inhibitory acts as a competition between neurons in the layer.
		#self.inhibitory = numpy.random.uniform(5.0,5.0,size=(self.layer_len,self.layer_len))

		#this atribute is explain in forward method
		self.temperature = 1.0

		#This atribute is explained in the backawrd method
		self.max_weight = 10.0
		self.min_weight = 0.0
		self.mid_weight = (self.max_weight - self.min_weight) / 2.0

	def forward(self, input):

		self.input = input
		#We allow excitatory weights to be between 0 and 10
		#we then substract 5.0 element wise, shifting the range from -5.0 to 5.0
		#after that we apply a sigmoid function so it goes from ~0.0 to ~1.0
		#this is then consider as the probabillity of firing, base on this
		#We randomly make it fired.
		#print 'self.excitatory' , self.excitatory
		output_ex = numpy.dot(input, self.excitatory) - self.mid_weight
		# print 'output_ex' , output_ex

		#Inhibitory weights makes neurons which large output inhibit other neurons.
		#Making them less probable to spike
		#output_inh = numpy.dot(output_ex, self.inhibitory ) - self.mid_weight
		output_inh = 0
		#print 'output_inh' , output_inh

		#We  sum both, the excitatory and inhibitory behaviors to get the final output
		output = output_ex - output_inh
		#print 'output', output

		#Tempeature makes the output smaller, make the probability closer to 0.5
		#being 0.5 the value where the variance is larger.
		#This makes our network have a more random behavior, which is good for 
		#exploring possible solutions. 
		#Temperature should drop to 1.0 as trainning progress.
		probabillity = 1/(1 + numpy.exp(-1 * output_ex / self.temperature))
		#print 'probabillity', probabillity

		#Consider a neuron was active if the generated random number
		#Is smaller than the input.
		random = numpy.random.rand(self.layer_len)

		#This means if the input is close to 1, most of the time it will spiked
		#Otherwise, is unprobable it will spike.
		self.spiked = random < probabillity

		return self.spiked

	def backward(self, reward): #TODO update the inhibitory

		#If the reward it positive in means the neuron which spiked made a good job.
		#and the strength of the weights of the neurons which spiked should be increased.
		#the change in the weights gets smaller, the closer it gets to the limits
		if reward > 0.0:
			#When we multiply self.excitatory * self.spiked, we get an array with the same size as self.excitatory
			#where all weights which connect to neuron which didn't fired are 0, otherwise the value in self.excitatory.
			#We will update it proportionally to the difference to the max weight, this proportionality is set with the reward.
			#If the reward is 1, our weights will get the self.max_weight in only one run, which is undesired.
			# print 'before updating weights with positive reward \n', self.excitatory
			self.excitatory =  self.excitatory + (self.max_weight - self.excitatory) * reward  * self.input[:, None] * self.spiked
			self.excitatory =  self.excitatory + (self.min_weight + self.excitatory) * (-1 * reward)  * self.input[:, None] * (self.spiked == False)
			# print 'after updating weights with postive reward \n', self.excitatory
			# print 'with self.spiked = \n' , self.spiked



			#When updating the inhibitory weight, we want neurons which succesfully inhibited other neurons to increase it weights,
			#To make it more probable that it will inhibit them again, but also, we want to decrease weight between neurons which 
			#couldn't inhibit each other.


		if reward < 0.0:
			#If the reward it's negative, we want to make the weights responsable of firing smaller.We do this proportional to 
			#the difference to the mimun weight and to the reward
			#print 'before updating weights with negative reward', self.excitatory
			self.excitatory =  self.excitatory + (self.min_weight + self.excitatory) * reward * self.input[:, None]  * self.spiked
			#self.excitatory =  self.excitatory + (self.max_weight - self.excitatory) * (-1 * reward)  * self.input[:, None] * (self.spiked == False)

			#print 'after updating weights with negative reward', self.excitatory
		return

class outputLayer:

	def __init__(self, input_len):
		self.input_len = input_len;
		return

	def forward(self, input):

		#The ouput of the above layer might have many spiked neurons (true values)
		#in that case we will return only one random value
		spiked = numpy.where(input == True)[0]

		#There were no spikes, return a random label
		if spiked.shape[0] == 0: 
			return numpy.random.randint(0, input.shape[0])

		#Just one spiked, ideal case
		if spiked.shape[0] == 1:
			return spiked[0]

		#More than one spiked, pick a random from this subset
		if spiked.shape[0] > 1:
			random_pick = numpy.random.randint(0, spiked.shape[0])
			return spiked[random_pick]
		

	def backward(self, reward):
		#Because this layers hasn't have any learnable parameter
		#We don't implement any backward pass
		pass