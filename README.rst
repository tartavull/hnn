===============================
hnn
===============================

.. image:: https://img.shields.io/pypi/v/hnn.svg
        :target: https://pypi.python.org/pypi/hnn

.. image:: https://img.shields.io/travis/tartavull/hnn.svg
        :target: https://travis-ci.org/tartavull/hnn

.. image:: https://readthedocs.org/projects/hnn/badge/?version=latest
        :target: https://readthedocs.org/projects/hnn/?badge=latest
        :alt: Documentation Status


Hebbian Neural Network

* Free software: ISC license
* Documentation: https://hnn.readthedocs.org.

Features
--------

* Everything

Some ideas we were talking about:

    1) Dopamine is globally applied and strengthens Hebbian connections that have been recently potentiated
    2) Neurons require a baseline firing rate to create tenuous Hebbian connections that can be reenforced when the right network forms
    3) If a connection is 0 weighted, hebbian learning cannot take place (this allows genes to provide structure)
    4) Not connecting is more powerful than connecting. This provides regularization.


### Something I wrote long time ago and I've no idea what it means
one example is show at a time in this case we only have two examples

[1, 0, 0, 0] -> 0
[0, 1, 0, 0] -> 1

the input layer has no effect into it.
the fullyConnectedLayer which has 2 cols and 4 rows of a weight matrix of rand number within 5 and 15.

[[14.09    11.22]
 [12.42    9.15 ]
 [8.7      12.44]
 [15.      5.54 ]]

when the first example is shown, the values [14.09 11.22] will be the sum for the first and the second neuron to fire.

if only one neuron fires , it is an easy case:
if the result is right we have a positive reward, otherwise a negative.

if no neuron fires, we should increment all weights by a constant to make the net more active. or do nothing.

if two or more neuron fires at the same time we should have a negative reward for one of them.


Credits
---------
