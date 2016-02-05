# -*- coding: utf-8 -*-
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

  trainer = Trainer ( nn, data , log = True)

  trainer.train(100000)

  nn.temperature = 1.0
  trainer.test()

  print numpy.round(nn.layers[1].excitatory, decimals=2)
  print nn.temperature
  # trainer.logger.plot()

  
  


if __name__ == '__main__':
  main_train_naive()
