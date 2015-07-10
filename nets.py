from layers import *
import numpy

class simpleNet:

	def __init__(self, architecture):

		self.layers = list()

		#Create input layer
		self.layers.append(inputLayer(architecture[0]))

		#Create the fully conected layers based on the architecture
		for i in range(len(architecture)-1):
			self.layers.append(fullyConectedLayer(architecture[i], architecture[i+1]))

		#Create the output layer
		self.layers.append(outputLayer(architecture[-1]))


	def forward(self, input):

		for layer in self.layers:
			input = layer.forward(input)
			
		return input

	def backward(self, reward):

		for layer in self.layers:
			layer.backward(reward)