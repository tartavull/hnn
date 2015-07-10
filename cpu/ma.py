import numpy

class MovingAverage:
	def __init__(self, periods):
		self.periods = periods
		self.window = list()
		self.mean = 0.0

	def append(self, new_value):
		assert type(new_value) is float

		self.window.append(new_value)

		if len(self.window) <= self.periods:
			self.mean = numpy.mean( self.window )
		else:
			self.mean = self.mean + (new_value  - self.window[0]) / self.periods
			del self.window[0]

		return self.mean

	
if __name__ == '__main__':
	
	"""Test this module"""

	ma = MovingAverage(5)
	ma.append(1.0)
	assert ma.mean == 1.0
	ma.append(2.0)
	ma.append(2.0)
	ma.append(3.0)
	assert ma.mean == 2.0
	ma.append(1.0)
	ma.append(1.0)
	ma.append(1.0)
	ma.append(1.0)
	ma.append(1.0)
	ma.append(1.0)
	assert ma.mean == 1.0
	
