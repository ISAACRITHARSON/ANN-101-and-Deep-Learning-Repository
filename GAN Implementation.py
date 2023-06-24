from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
def define_generator(latent_dim, n_outputs=2):
model = Sequential()
model.add(Dense(15,activation='relu',kernel_initializer='he_uniform',input_dim=latent_dim))
model.add(Dense(n_outputs, activation='linear'))
return model
def generate_latent_points(latent_dim, n):
x_input = randn(latent_dim * n)
x_input = x_input.reshape(n, latent_dim)
return x_input
def generate_fake_samples(generator, latent_dim, n):
x_input = generate_latent_points(latent_dim, n)
X = generator.predict(x_input)
pyplot.scatter(X[:, 0], X[:, 1])
pyplot.show()
latent_dim = 5
model = define_generator(latent_dim)
generate_fake_samples(model, latent_dim, 100)


from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

def define_generator(latent_dim, n_outputs=2):
model = Sequential()
model.add(Dense(15,activation='relu',kernel_initializer='he_uniform',input_dim=latent_dim))
model.add(Dense(n_outputs, activation='linear'))
return model

def generate_latent_points(latent_dim, n):
# generate points in the latent space
x_input = randn(latent_dim * n)
# reshape into a batch of inputs for the network
x_input = x_input.reshape(n, latent_dim)
return x_input

def generate_fake_samples(generator, latent_dim, n):
# generate points in latent space
x_input = generate_latent_points(latent_dim, n)
# predict outputs
X = generator.predict(x_input)
# plot the results
pyplot.scatter(X[:, 0], X[:, 1]**2)
pyplot.show()
latent_dim = 5
model = define_generator(latent_dim)
generate_fake_samples(model, latent_dim, 100)
