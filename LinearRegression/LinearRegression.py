import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def main():
	# Initialize Dataset
	X = 10*np.random.rand(50)
	y = 8*X + 1 + 2.5*np.random.randn(50)
	fig = plt.gcf()
	plt.scatter(X, y)
	plt.show()
	model = LinearRegression()
	model.train(X,y)


class LinearRegression():
	# Using Gradient Descent for Linear Regression
	def __init__(self, learning_rate=0.01, epochs=1000):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.a_0 = 0
		self.a_1 = 0

	def animate(self,frame):
		pass

	def train(self, X, y):
		n = X.shape[0]

		for i in range(self.epochs):
			y_train = self.a_0 + self.a_1 * X
			error = y - y_train        # Whether you use y_train - y or y - y_train will make a difference
			mse = np.sum(error ** 2) / n
			self.a_0 -= -2/n * np.sum(error) * self.learning_rate
			self.a_1 -= -2/n * np.sum(error * X) * self.learning_rate

			if i%10 == 0:
				print("MSE:", mse)
		# Plot out the line of best fit
		y_plot = []
		plot_range = range(int(min(X))-1,int(max(X))+3)
		for i in plot_range:
			y_plot.append(self.a_0 + self.a_1*i)
		plt.plot(plot_range,y_plot,color ="red", label = "Best Fit")

if __name__ == "__main__":
	main()


