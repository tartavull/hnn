#This class is reponsable of loading diferent datasets
#it's always returns 3 datasets,   Train, Validate, Test
import cPickle
import gzip
import os
import urllib
import numpy

def load_mnist():
    

    filepath = './mnist.pkl.gz'

    # Download the MNIST dataset if it is not present
    if not os.path.isfile(filepath):

        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, filepath)

    # Load the dataset
    print '... loading data'

    #We have to descompress it
    f = gzip.open(filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

def load_dictionary():

    dictionary = numpy.zeros(26, dtype='u1, (15,)u1')
    dictionary[:] = [(0,[1, 0, 1, 0, 1, 0 , 0, 0, 0 , 0, 1, 0 , 0, 1, 0 ]),
                     (1,[0, 0, 1, 0, 1, 0 , 0, 0, 1 , 0, 1, 0 , 0, 0, 1 ]),
                     (2,[0, 0, 0, 0, 1, 1 , 0, 1, 1 , 0, 1, 1 , 0, 0, 0 ]),
                     (3,[0, 0, 1, 0, 1, 0 , 0, 1, 0 , 0, 1, 0 , 0, 0, 1 ]),
                     (4,[0, 0, 0, 0, 1, 1 , 0, 0, 1 , 0, 1, 1 , 0, 0, 0 ]),
                     (5,[0, 0, 0, 0, 1, 1 , 0, 0, 1 , 0, 1, 1 , 0, 1, 1 ]),
                     (6,[0, 0, 0, 0, 1, 1 , 0, 1, 1 , 0, 1, 0 , 0, 0, 0 ]),
                     (7,[0, 1, 0, 0, 1, 0 , 0, 0, 0 , 0, 1, 0 , 0, 1, 0 ]),
                     (8,[0, 0, 0, 1, 0, 1 , 1, 0, 1 , 1, 0, 1 , 0, 0, 0 ]),
                     (9,[0, 0, 0, 1, 1, 0 , 1, 1, 0 , 0, 1, 0 , 0, 0, 0 ]),
                     (10,[0, 1, 0, 0, 0, 1 , 0, 1, 1 , 0, 0, 1 , 0, 1, 0 ]),
                     (11,[0, 1, 1, 0, 1, 1 , 0, 1, 1 , 0, 1, 1 , 0, 0, 0 ]),
                     (12,[0, 1, 0, 0, 0, 0 , 0, 1, 0 , 0, 1, 0 , 0, 1, 0 ]),
                     (13,[0, 1, 0, 0, 0, 0 , 0, 0, 0 , 0, 0, 0 , 0, 1, 0 ]),
                     (14,[1, 0, 1, 0, 1, 0 , 0, 1, 0 , 0, 1, 0 , 1, 0, 1 ]),
                     (15,[0, 0, 0, 0, 1, 0 , 0, 0, 0 , 0, 1, 1 , 0, 1, 1 ]),
                     (16,[1, 0, 1, 0, 1, 0 , 0, 1, 0 , 1, 0, 1 , 1, 1, 0 ]),
                     (17,[0, 0, 1, 0, 1, 0 , 0, 0, 0 , 0, 0, 1 , 0, 1, 0 ]),
                     (18,[1, 0, 0, 0, 1, 1 , 1, 0, 1 , 1, 1, 0 , 0, 0, 1 ]),
                     (19,[0, 0, 0, 1, 0, 1 , 1, 0, 1 , 1, 0, 1 , 1, 0, 1 ]),
                     (20,[0, 1, 0, 0, 1, 0 , 0, 1, 0 , 0, 1, 0 , 0, 0, 0 ]),
                     (21,[0, 1, 0, 0, 1, 0 , 0, 1, 0 , 0, 1, 0 , 1, 0, 1 ]),
                     (22,[0, 1, 0, 0, 1, 0 , 0, 1, 0 , 0, 0, 0 , 0, 1, 0 ]),
                     (23,[0, 1, 0, 0, 1, 0 , 1, 0, 1 , 0, 1, 0 , 0, 1, 0 ]),
                     (24,[0, 1, 0, 0, 1, 0 , 1, 0, 1 , 1, 0, 1 , 1, 0, 1 ]),
                     (25,[0, 0, 0, 1, 1, 0 , 1, 0, 1 , 0, 1, 1 , 0, 0, 0 ])]
    return dictionary

def load_naive():
    
    dictionary = numpy.zeros(4, dtype='u1, (6,)u1')
    dictionary[:] = [(0,[1, 0, 0, 1, 1, 1]),
                     (1,[0, 1, 0, 0, 1, 1]),
                     (2,[0, 0, 1, 0, 1, 1]),
                     (3,[0, 0, 1, 1, 1, 1])]
    return dictionary

