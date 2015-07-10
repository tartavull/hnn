import numpy

import dataset
from nets import *
from trainer import *

def main_train_naive():

	data = dataset.load_naive()

	input_length = data[0][1].shape[0]

	#there is only of example for every label in this datast
	number_of_labels = len(data)

	# We create an archicture with now hidden layers.
	architecture =  (( input_length , number_of_labels ))

	nn = simpleNet( architecture )

	trainer = Trainer ( nn, data )

	trainer.train(100000)
	


if __name__ == '__main__':
	main_train_naive()
