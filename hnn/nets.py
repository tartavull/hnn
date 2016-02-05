from layers import *
import numpy

class simpleNet:

	def __init__(self, architecture):

		self.layers = list()

		#Create input layer
		self.layers.append(inputLayer( self , architecture[0]))

		#Create the fully conected layers based on the architecture
		for i in range(len(architecture)-1):
			self.layers.append(fullyConectedLayer( self, architecture[i], architecture[i+1]))

		#Create the output layer
		self.layers.append(outputLayer( self, architecture[-1]))

		self.temperature = 1.0

		self.max_temperature = 20.0
		self.min_temperature = 1.0


	def forward(self, input):

		for layer in self.layers:
			input = layer.forward(input)
			
		return input

	def backward(self, reward):

		for layer in self.layers:
			layer.backward(reward)


	def increaseTemperature(self):

		self.temperature +=  0.1 * ( self.max_temperature - self.temperature ) 


	def decreaseTemperature(self):

		self.temperature -= 0.1 * ( self.temperature - self.min_temperature ) 