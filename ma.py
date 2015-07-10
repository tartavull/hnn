import numpy

class movingAverage:
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

	